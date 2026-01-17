#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTK + IMU406 地下管道定位（带大地2000 + CSV + 管道深度）

功能：
1. 解析 RTK MQTT 中的 GGA，得到 WGS-84 经纬度；
2. 解析 IMU406（命令 0x84）的 9 轴数据，做一维里程计积分；
3. 以 RTK 首次解作为锚点，把 IMU 位移投影到 ENU 平面，得到融合后 WGS-84 经纬度；
4. 将融合后的 WGS-84 坐标转换为“大地2000(CGCS2000)”坐标（目前采用 WGS≈CGCS2000 近似）；
5. 利用前进距离 + 俯仰角估算“相对入口的管道深度”；
6. 所有关键数据写入 CSV 文件 rtk_imu_log.csv，便于后处理 / 画轨迹。
"""

import time
import math
import csv
import os
import serial
import paho.mqtt.client as mqtt
from datetime import datetime

# ================== 基本配置 ==================

# ---- RTK MQTT ----
BROKER_ADDRESS = "mqtt.wit-motion.cn"
BROKER_PORT = 31883
MQTT_USERNAME = "CYL"
MQTT_PASSWORD = "FDA4235JHKBFSJF541KDO3NFK4"

RTK_IMEI = "867920077213354"      # 你的移动站 IMEI
RTK_TOPIC = f"device/{RTK_IMEI}/upload"

# ---- IMU 串口 ----
IMU_PORT = "/dev/ttyUSB0"         # 你的 IMU 当前端口
IMU_BAUDRATE = 115200

# ---- 物理常数 ----
G = 9.80665                        # m/s^2
EARTH_R = 6378137.0                # 地球半径

# 零速检测 + 速度衰减
ACC_ZERO_THRESHOLD_G = 0.02        # 小于 0.02g 认为无加速度
VEL_DAMPING_WHEN_ZERO_ACC = 0.9    # 无加速度时给速度打折

# 零偏标定采样次数
BIAS_SAMPLE_COUNT = 50

# IMU406 帧长（固定 32 字节：68 ... 校验和）
IMU_FRAME_LEN = 32

# ---- CSV 日志 ----
CSV_PATH = "rtk_imu_log.csv"

# ================== 全局状态 ==================

# RTK 当前值（原始 + 锚点）
rtk_lat = None         # 当前 RTK 纬度 (WGS-84)
rtk_lon = None         # 当前 RTK 经度 (WGS-84)
rtk_fix_q = None       # 当前解状态数字
rtk_gga_raw = None     # 最近一条原始 GGA
rtk_utc_raw = None     # GGA 里的 UTC 时间字符串

anchor_lat = None      # RTK 锚点纬度
anchor_lon = None      # RTK 锚点经度

# IMU 里程计状态
imu_bias_calibrated = False
imu_forward_bias_g = 0.0
bias_samples = []

forward_vel = 0.0      # 前进方向速度 m/s
forward_dist = 0.0     # 累计前进距离 m
enu_east = 0.0         # 东向位移 m
enu_north = 0.0        # 北向位移 m

# 管道深度（相对入口，向下为正）
pipe_depth = 0.0       # m

last_imu_time = None   # 上一帧 IMU 时间戳（monotonic）

# CSV writer 句柄
csv_file = None
csv_writer = None

# =======================================================
#                  坐标转换相关
# =======================================================

def wgs84_to_cgcs2000(lat_deg: float, lon_deg: float):
    """
    WGS-84 -> CGCS2000（大地2000）转换。

    说明：
    - 实际工程中需要使用对应区域的七参数 / 四参数精确转换；
    - 没有具体参数时，常规做法是认为 WGS-84 与 CGCS2000
      在厘米级以内非常接近，可以近似相等。

    目前默认直接返回原值，方便你后续替换为精确模型。
    """
    return lat_deg, lon_deg

# =======================================================
#                   RTK 解析部分
# =======================================================

def nmea_to_decimal(coord_str, hemi):
    """NMEA ddmm.mmmm -> 十进制度"""
    if not coord_str or coord_str == "0":
        return None
    val = float(coord_str)
    deg = int(val // 100)
    minutes = val - deg * 100
    decimal = deg + minutes / 60.0
    if hemi in ("S", "W"):
        decimal = -decimal
    return decimal


def parse_gga(sentence):
    """解析 GGA，返回 (lat, lon, fix_q, utc_raw) 或 None"""
    parts = sentence.strip().split(',')
    if len(parts) < 10:
        return None
    if parts[0] not in ("$GNGGA", "$GPGGA"):
        return None

    utc_raw = parts[1]
    lat = nmea_to_decimal(parts[2], parts[3])
    lon = nmea_to_decimal(parts[4], parts[5])
    fix_q = parts[6]

    if not fix_q or fix_q == "0":
        return None

    return lat, lon, fix_q, utc_raw


def fix_quality_desc(fix_q: str) -> str:
    mapping = {
        "0": "无定位",
        "1": "单点定位",
        "2": "差分定位(DGPS)",
        "4": "RTK浮点解",
        "5": "RTK固定解",
        "6": "推算(DR)",
        "7": "厂商扩展高精度解",
    }
    return mapping.get(fix_q, "未知")


def format_utc(utc_raw: str) -> str:
    """GGA 里的 UTC 时间转成 hh:mm:ss 字符串"""
    if not utc_raw or len(utc_raw) < 6:
        return "未知"
    try:
        h = int(utc_raw[0:2])
        m = int(utc_raw[2:4])
        s = float(utc_raw[4:])
        return f"{h:02d}:{m:02d}:{int(s):02d} (UTC)"
    except Exception:
        return utc_raw


def handle_rtk_payload(payload: bytes):
    """MQTT 收到 RTK payload 时调用：解析 GGA、更新当前 RTK + 锚点"""
    global rtk_lat, rtk_lon, rtk_fix_q, rtk_gga_raw, rtk_utc_raw
    global anchor_lat, anchor_lon

    text = payload.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("$GNGGA") or line.startswith("$GPGGA")):
            continue

        rtk_gga_raw = line  # 保存原始 GGA

        res = parse_gga(line)
        if not res:
            continue

        lat, lon, fix_q, utc_raw = res
        rtk_lat = lat
        rtk_lon = lon
        rtk_fix_q = fix_q
        rtk_utc_raw = utc_raw

        # 首次拿到有效 RTK 时，设置锚点
        if anchor_lat is None or anchor_lon is None:
            anchor_lat = lat
            anchor_lon = lon
            print(
                f"[RTK] 锚点设置成功: 经度={lon:.8f}, 纬度={lat:.8f}, "
                f"解状态={fix_q}({fix_quality_desc(fix_q)})"
            )


def on_rtk_connect(client, userdata, flags, rc):
    print(f"[RTK] Connected with result code {rc}")
    if rc == 0:
        client.subscribe(RTK_TOPIC)
        print(f"[RTK] Subscribed to: {RTK_TOPIC}")
    else:
        print("[RTK] 连接失败，请检查账号密码/端口")


def on_rtk_message(client, userdata, msg):
    handle_rtk_payload(msg.payload)


def start_rtk_mqtt():
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_rtk_connect
    client.on_message = on_rtk_message
    print("[RTK] Connecting MQTT...")
    client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
    client.loop_start()
    return client

# =======================================================
#                    IMU 解析部分
# =======================================================

def bcd_to_float_angle(b0, b1, b2):
    """IMU406 角度 BCD -> degree"""
    sign = -1.0 if (b0 & 0xF0) == 0x10 else 1.0
    d0 = b0 & 0x0F
    temp = d0 * 100.0

    h_d = (b1 >> 4) & 0x0F
    l_d = b1 & 0x0F
    temp += h_d * 10.0 + l_d

    h_d = (b2 >> 4) & 0x0F
    l_d = b2 & 0x0F
    temp += h_d * 0.10 + l_d * 0.01

    return sign * temp


def bcd_to_float_acc(b0, b1, b2):
    """IMU406 加速度 BCD -> g"""
    sign = -1.0 if (b0 & 0xF0) == 0x10 else 1.0
    d0 = b0 & 0x0F
    temp = d0 * 10.0

    h_d = (b1 >> 4) & 0x0F
    l_d = b1 & 0x0F
    temp += h_d + l_d * 0.1

    h_d = (b2 >> 4) & 0x0F
    l_d = b2 & 0x0F
    temp += h_d * 0.01 + l_d * 0.001

    return sign * temp


def parse_imu_9axis_frame(frame: bytes):
    """
    解析 IMU406 一帧 9 轴数据（命令 0x84）
    帧长固定 32 字节：68 1F 00 84 ...
    返回 (roll_deg, pitch_deg, yaw_deg, accx_g, accy_g, accz_g) 或 None
    """
    if len(frame) != IMU_FRAME_LEN:
        return None

    # 校验和：从 frame[1] 到 frame[-2] 之和
    calc_ck = sum(frame[1:-1]) & 0xFF
    if calc_ck != frame[-1]:
        return None

    if frame[0] != 0x68 or frame[1] != 0x1F:
        return None

    # payload 从命令字开始：frame[3] == 0x84
    payload = frame[3:-1]
    if payload[0] != 0x84:
        return None

    roll = bcd_to_float_angle(payload[1], payload[2], payload[3])
    pitch = bcd_to_float_angle(payload[4], payload[5], payload[6])
    yaw = bcd_to_float_angle(payload[7], payload[8], payload[9])

    accx = bcd_to_float_acc(payload[10], payload[11], payload[12])
    accy = bcd_to_float_acc(payload[13], payload[14], payload[15])
    accz = bcd_to_float_acc(payload[16], payload[17], payload[18])

    return roll, pitch, yaw, accx, accy, accz


def imu_loop():
    """
    IMU 主循环：
    - 解析 68 1F 00 84 的 32 字节帧
    - 标定前向加速度零偏（X轴）
    - 积分得到前进距离 + ENU 位移 + 相对深度
    - 和 RTK 锚点融合，并打印 / 记录 CSV
    """
    global last_imu_time
    global imu_bias_calibrated, imu_forward_bias_g, bias_samples
    global forward_vel, forward_dist, enu_east, enu_north, pipe_depth
    global csv_writer, csv_file

    ser = serial.Serial(IMU_PORT, IMU_BAUDRATE, timeout=0.1)
    print(f"[IMU] 打开串口 {IMU_PORT} @ {IMU_BAUDRATE}")

    buf = bytearray()

    while True:
        data = ser.read(256)
        if data:
            buf.extend(data)

        # 从缓冲区连续拆帧
        while True:
            if len(buf) < IMU_FRAME_LEN:
                break

            # 找 0x68 作为帧头
            if buf[0] != 0x68:
                buf.pop(0)
                continue

            # 取固定 32 字节一帧
            frame = bytes(buf[:IMU_FRAME_LEN])
            del buf[:IMU_FRAME_LEN]

            res = parse_imu_9axis_frame(frame)
            if not res:
                continue

            roll_deg, pitch_deg, yaw_deg, accx_g, accy_g, accz_g = res

            # 计算 dt
            now = time.monotonic()
            if last_imu_time is None:
                last_imu_time = now
                continue
            dt = now - last_imu_time
            last_imu_time = now

            # ========= 零偏标定阶段（只做一次） =========
            if not imu_bias_calibrated:
                bias_samples.append(accx_g)
                print(
                    f"[IMU] 标定前向零偏中 ({len(bias_samples)}/{BIAS_SAMPLE_COUNT}) "
                    f"当前 accx={accx_g:+.4f} g"
                )
                if len(bias_samples) >= BIAS_SAMPLE_COUNT:
                    imu_forward_bias_g = sum(bias_samples) / len(bias_samples)
                    imu_bias_calibrated = True
                    print(f"[IMU] 前向加速度零偏标定完成: bias = {imu_forward_bias_g:.5f} g")
                continue

            # ========= 一维里程计（前进方向） =========
            # 前进方向 = IMU X 轴，减去零偏
            acc_forward_g = accx_g - imu_forward_bias_g

            # 零速检测：加速度极小则认为在匀速/静止 → 速度打折，防止积分发散
            if abs(acc_forward_g) < ACC_ZERO_THRESHOLD_G:
                forward_vel *= VEL_DAMPING_WHEN_ZERO_ACC
            else:
                acc_forward_ms2 = acc_forward_g * G
                forward_vel += acc_forward_ms2 * dt

            ds = forward_vel * dt
            forward_dist += ds

            # 根据航向角 yaw 把 ds 分解到 ENU 平面
            yaw_rad = math.radians(yaw_deg)
            de = ds * math.sin(yaw_rad)   # East
            dn = ds * math.cos(yaw_rad)   # North
            enu_east += de
            enu_north += dn

            # ========= 根据俯仰角估算管道深度 =========
            # 约定：pitch > 0 头抬起往上爬，深度变浅；pitch < 0 往下钻，深度变深
            # 取“向下为正”的深度定义：
            pitch_rad = math.radians(pitch_deg)
            d_depth = -ds * math.sin(pitch_rad)   # pitch<0 → sin<0 → d_depth>0（变深）
            pipe_depth += d_depth

            # ========= RTK + IMU 融合经纬度 (WGS-84) =========
            fused_lat = fused_lon = None
            fused_lat_2000 = fused_lon_2000 = None
            if anchor_lat is not None and anchor_lon is not None:
                lat0_rad = math.radians(anchor_lat)
                dlat = (enu_north / EARTH_R) * 180.0 / math.pi
                dlon = (enu_east / (EARTH_R * math.cos(lat0_rad))) * 180.0 / math.pi
                fused_lat = anchor_lat + dlat
                fused_lon = anchor_lon + dlon

                # WGS-84 -> CGCS2000
                fused_lat_2000, fused_lon_2000 = wgs84_to_cgcs2000(
                    fused_lat, fused_lon
                )

            # ========= 输出一帧完整信息 =========
            local_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gnss_ts = format_utc(rtk_utc_raw) if rtk_utc_raw else "未知"
            rtk_desc = fix_quality_desc(rtk_fix_q) if rtk_fix_q else "无RTK"

            print("================================================================")
            print(f"[本机时间] {local_ts}    [GNSS时间] {gnss_ts}")

            # 1. 原始 RTK
            if rtk_lat is not None and rtk_lon is not None:
                print(
                    f"[RTK 原始] 经度={rtk_lon:.8f}, 纬度={rtk_lat:.8f}, "
                    f"解状态={rtk_fix_q}({rtk_desc})"
                )
            else:
                print("[RTK 原始] 暂无有效 RTK 位置")

            # 2. IMU 积分
            print(
                f"[IMU 积分] 前进距离 = {forward_dist:+7.3f} m, "
                f"East = {enu_east:+7.3f} m, North = {enu_north:+7.3f} m"
            )

            # 管道深度
            print(
                f"[管道深度] 相对入口 = {pipe_depth:+7.3f} m （向下为正）"
            )

            # 姿态信息
            print(
                f"[IMU 姿态] Roll = {roll_deg:+7.2f}°   "
                f"Pitch = {pitch_deg:+7.2f}°   "
                f"Yaw = {yaw_deg:+7.2f}°"
            )

            # 3. 融合坐标（WGS-84 & 大地2000）
            if fused_lat is not None and fused_lon is not None:
                print(
                    f"[融合坐标-WGS84]  经度={fused_lon:.8f}, "
                    f"纬度={fused_lat:.8f} (相对 RTK 锚点偏移)"
                )
                print(
                    f"[融合坐标-大地2000] 经度={fused_lon_2000:.8f}, "
                    f"纬度={fused_lat_2000:.8f}"
                )
            else:
                print("[融合坐标] 暂无（等待 RTK 锚点）")

            # 4. 原始 GGA
            if rtk_gga_raw:
                print(f"[原始 GGA] {rtk_gga_raw}")

            # ========= 写入 CSV =========
            if csv_writer is not None:
                def safe(v):
                    return "" if v is None else v

                csv_writer.writerow([
                    local_ts,                 # 本机时间
                    gnss_ts,                  # GNSS 时间(字符串)
                    safe(rtk_lat),            # RTK 纬度 (WGS84)
                    safe(rtk_lon),            # RTK 经度 (WGS84)
                    safe(rtk_fix_q),          # RTK 解状态码
                    rtk_desc,                 # RTK 解状态描述
                    forward_dist,             # IMU 前进距离
                    enu_east,                 # ENU East
                    enu_north,                # ENU North
                    pipe_depth,               # 管道深度（相对入口，向下为正）
                    roll_deg, pitch_deg, yaw_deg,
                    safe(fused_lat),          # 融合纬度 WGS84
                    safe(fused_lon),          # 融合经度 WGS84
                    safe(fused_lat_2000),     # 融合纬度 CGCS2000
                    safe(fused_lon_2000),     # 融合经度 CGCS2000
                    rtk_gga_raw or ""         # 原始 GGA
                ])
                csv_file.flush()

# =======================================================
#                        主函数
# =======================================================

def init_csv():
    """
    初始化 CSV 文件：若不存在则写表头；存在则追加。
    """
    global csv_file, csv_writer

    new_file = not os.path.exists(CSV_PATH)
    csv_file = open(CSV_PATH, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    if new_file:
        csv_writer.writerow([
            "local_time",
            "gnss_time_str",
            "rtk_lat_wgs84",
            "rtk_lon_wgs84",
            "rtk_fix_q",
            "rtk_fix_desc",
            "imu_forward_dist_m",
            "enu_east_m",
            "enu_north_m",
            "pipe_depth_m",          # 新增：管道深度
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "fused_lat_wgs84",
            "fused_lon_wgs84",
            "fused_lat_cgcs2000",
            "fused_lon_cgcs2000",
            "raw_gga"
        ])
        csv_file.flush()
    print(f"[CSV] 日志输出到: {os.path.abspath(CSV_PATH)}")


def main():
    # 初始化 CSV 日志
    init_csv()
    # 先启动 RTK MQTT
    start_rtk_mqtt()
    # 再进入 IMU 主循环
    try:
        imu_loop()
    finally:
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
