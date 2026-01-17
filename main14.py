# 导入操作系统相关模块
import os
import sys
import time
import json
import random
import math
import requests
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# 修复高分辨率屏 DPI 缩放问题
os.environ["QT_FONT_DPI"] = "96"

# 尝试导入YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("成功导入YOLO库")
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: 未找到YOLO库，目标检测功能不可用")

# 引入MQTT客户端库
import paho.mqtt.client as mqtt

# 导入PySide6 GUI组件
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtCharts import *
from PySide6.QtMultimedia import *
from PySide6.QtMultimediaWidgets import *
from PySide6.QtWebEngineWidgets import QWebEngineView

# 全局配置
CURRENT_USER = "12ljf"
CURRENT_DATE = "2025-06-22 14:01:55"

# MQTT配置
MQTT_BROKER = "47.107.36.182"
MQTT_PORT = 1883
MQTT_CLIENT_ID = "USER001"
MQTT_USERNAME = "public"
MQTT_PASSWORD = "UQU92K77cpxc2Tm"
MQTT_TOPIC_PUBLISH = "USER001"
MQTT_TOPIC_SUBSCRIBE = "USER002"

# 地图配置
DEFAULT_LATITUDE =  22.902542
DEFAULT_LONGITUDE = 113.875019

# 模式和颜色配置
GAIT_MODES = ["蠕动模式", "蜿蜒模式", "复位模式"]
GAIT_COLORS = {
    "蠕动模式": "#00FF88",
    "蜿蜒模式": "#FF8C00", 
    "复位模式": "#FF6B6B"
}
DIRECTIONS = {"前进": "↑", "后退": "↓", "左转": "←", "右转": "→", "复位": "↺"}

# YOLO配置
YOLO_MODEL_PATH = "yolov8n.pt"  # 默认模型路径
YOLO_CONFIDENCE = 0.2 # 极低的置信度阈值，确保能检测到更多物体
YOLO_DEBUG = False  # 是否启用调试


# 资源管理器 - 策略和工厂模式
class ResourceManager:
    """资源管理器 - 使用策略模式和工厂模式管理硬件资源"""
    _instance = None
    
    def __new__(cls):
        # 单例模式确保整个应用使用同一个资源管理器
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.cv_processors = {}
        self.ai_processors = {}
        self.resources_info = self._detect_resources()
        
        # 初始化处理器
        self._init_processors()
        
    def _detect_resources(self):
        """检测系统可用资源"""
        resources = {
            "gpu_available": False,
            "gpu_info": [],
            "cpu_count": os.cpu_count(),
            "camera_api": self._detect_camera_api(),
            "memory_available": self._get_available_memory()
        }
        
        # 检测GPU (CUDA)
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                resources["gpu_available"] = True
                for i in range(cv2.cuda.getCudaEnabledDeviceCount()):
                    dev_info = {
                        "index": i,
                        "name": f"GPU-{i}",  # 简化名称，实际应使用CUDA API获取
                        "compute_capability": "Unknown"  # 简化，实际应使用CUDA API获取
                    }
                    resources["gpu_info"].append(dev_info)
        except Exception:
            # 如果cv2.cuda不可用，尝试检查CUDA_VISIBLE_DEVICES环境变量
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES']:
                resources["gpu_available"] = True
                resources["gpu_info"].append({
                    "index": 0,
                    "name": "GPU (from env)",
                    "compute_capability": "Unknown"
                })
            
        return resources
    
    def _detect_camera_api(self):
        """检测最佳摄像头API"""
        # 按优先级检测可用的摄像头API
        apis = [
            (cv2.CAP_DSHOW, "DirectShow"),  # Windows
            (cv2.CAP_V4L2, "V4L2"),         # Linux
            (cv2.CAP_AVFOUNDATION, "AVFoundation"),  # macOS
            (cv2.CAP_ANY, "Auto")           # 自动
        ]
        
        for api_id, api_name in apis:
            try:
                # 尝试使用此API打开摄像头（不实际打开，只检测可用性）
                if sys.platform == 'win32' and api_id == cv2.CAP_DSHOW:
                    return api_id  # Windows优先使用DirectShow
                elif sys.platform == 'linux' and api_id == cv2.CAP_V4L2:
                    return api_id  # Linux优先使用V4L2
                elif sys.platform == 'darwin' and api_id == cv2.CAP_AVFOUNDATION:
                    return api_id  # macOS优先使用AVFoundation
            except Exception:
                continue
                
        return cv2.CAP_ANY  # 默认自动选择
    
    def _get_available_memory(self):
        """获取可用内存（GB）"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 4.0  # 默认假设有4GB可用内存
    
    def _init_processors(self):
        """初始化各种处理器"""
        # 视频处理器
        self.cv_processors = {
            "cpu": CPUVideoProcessor(),
            "gpu": GPUVideoProcessor() if self.resources_info["gpu_available"] else CPUVideoProcessor()
        }
        
        # AI推理处理器
        self.ai_processors = {
            "cpu": CPUInferenceProcessor(),
            "gpu": GPUInferenceProcessor() if self.resources_info["gpu_available"] else CPUInferenceProcessor()
        }
    
    def get_video_processor(self):
        """获取最佳视频处理器"""
        if self.resources_info["gpu_available"]:
            return self.cv_processors["gpu"]
        return self.cv_processors["cpu"]
    
    def get_inference_processor(self):
        """获取最佳AI推理处理器"""
        if self.resources_info["gpu_available"]:
            return self.ai_processors["gpu"]
        return self.ai_processors["cpu"]
    
    def get_camera_api(self):
        """获取摄像头API"""
        return self.resources_info["camera_api"]
    
    def get_resources_summary(self):
        """获取资源概要"""
        summary = {
            "gpu_available": self.resources_info["gpu_available"],
            "cpu_count": self.resources_info["cpu_count"],
            "video_processor": "GPU" if self.resources_info["gpu_available"] else "CPU",
            "inference_processor": "GPU" if self.resources_info["gpu_available"] else "CPU"
        }
        return summary


# 视频处理器策略接口
class VideoProcessor:
    """视频处理器接口"""
    def process_frame(self, frame):
        """处理单帧图像"""
        raise NotImplementedError
    
    def convert_to_qt(self, frame):
        """转换为Qt图像格式"""
        raise NotImplementedError


# CPU视频处理器
class CPUVideoProcessor(VideoProcessor):
    """CPU视频处理器实现"""
    def process_frame(self, frame):
        """CPU处理图像帧"""
        if frame is None:
            return None
            
        # 简单的CPU图像处理 - 可根据需要添加更多处理
        # 这里仅执行缩放和简单的色彩调整
        try:
            # 确保帧数据有效
            if frame.size == 0:
                return None
                
            # 可选的处理步骤（提高性能）
            h, w = frame.shape[:2]
            if w > 1280:  # 如果尺寸太大，调整大小以提高性能
                scale = 1280 / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
            return frame
            
        except Exception as e:
            print(f"CPU视频处理错误: {e}")
            return frame
    
    def convert_to_qt(self, frame):
        """转换为Qt图像格式 - CPU实现"""
        try:
            if frame is None or frame.size == 0:
                return None
                
            # 确保是BGR格式
            if len(frame.shape) == 3:
                height, width, channel = frame.shape
                if channel == 3:
                    # BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    bytes_per_line = 3 * width
                    qt_image = QImage(rgb_frame.data, width, height, 
                                    bytes_per_line, QImage.Format_RGB888)
                    return qt_image
            
            return None
            
        except Exception as e:
            print(f"图像转换错误: {e}")
            return None


# GPU视频处理器
class GPUVideoProcessor(VideoProcessor):
    """GPU视频处理器实现"""
    def __init__(self):
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if not self.gpu_available:
            print("警告: CUDA不可用，将使用CPU处理视频")
            
        # 尝试预热GPU
        if self.gpu_available:
            try:
                test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(test_frame)
                gpu_frame.download()
            except Exception as e:
                print(f"GPU预热失败: {e}")
                self.gpu_available = False
    
    def process_frame(self, frame):
        """GPU处理图像帧"""
        if frame is None or not self.gpu_available:
            # 如果帧为空或GPU不可用，使用CPU处理
            cpu_processor = CPUVideoProcessor()
            return cpu_processor.process_frame(frame)
            
        try:
            # 将帧上传到GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # GPU处理
            # 示例：调整大小或应用滤镜
            h, w = frame.shape[:2]
            if w > 1280:  # 如果尺寸太大，调整大小以提高性能
                scale = 1280 / w
                gpu_resized = cv2.cuda.resize(gpu_frame, 
                                            (int(w * scale), int(h * scale)))
                # 下载处理后的帧
                return gpu_resized.download()
            
            # 如果不需要调整大小，直接下载
            return gpu_frame.download()
            
        except Exception as e:
            print(f"GPU视频处理错误: {e}")
            # 出错时回退到CPU处理
            cpu_processor = CPUVideoProcessor()
            return cpu_processor.process_frame(frame)
    
    def convert_to_qt(self, frame):
        """转换为Qt图像格式 - 与CPU相同，因为需要在主内存中操作"""
        cpu_processor = CPUVideoProcessor()
        return cpu_processor.convert_to_qt(frame)


# 推理处理器策略接口
class InferenceProcessor:
    """AI推理处理器接口"""
    def setup_model(self, model_path):
        """设置模型"""
        raise NotImplementedError
    
    def infer(self, frame, confidence=0.5):
        """执行推理"""
        raise NotImplementedError


# CPU推理处理器
class CPUInferenceProcessor(InferenceProcessor):
    """CPU推理处理器实现"""
    def setup_model(self, model_path):
        """在CPU上设置模型"""
        if not YOLO_AVAILABLE:
            return None
            
        try:
            return YOLO(model_path, task='detect', device='cpu')
        except Exception as e:
            print(f"CPU模型设置错误: {e}")
            return None
    
    def infer(self, frame, model, confidence=0.25):  # 降低默认置信度阈值
        """在CPU上执行推理"""
        if frame is None or model is None:
            return None, []
            
        try:
            # 推理前调整图像大小以提高性能
            h, w = frame.shape[:2]
            
            # 执行推理
            results = model(frame, conf=confidence, verbose=False)
            
            # 解析结果
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'class': cls,
                            'name': result.names[cls],
                            'confidence': float(conf),
                            'box': [int(x1), int(y1), int(x2), int(y2)]
                        })
                        
                    print(f"检测到 {len(detections)} 个物体")
            
            return frame, detections
            
        except Exception as e:
            print(f"CPU推理错误: {e}")
            return frame, []


# GPU推理处理器
class GPUInferenceProcessor(InferenceProcessor):
    """GPU推理处理器实现"""
    def __init__(self):
        self.gpu_available = False
        
        # 检测CUDA是否可用
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.device = f"cuda:{torch.cuda.current_device()}"
            else:
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"
            
    def setup_model(self, model_path):
        """在GPU上设置模型"""
        if not YOLO_AVAILABLE:
            return None
            
        try:
            return YOLO(model_path, task='detect', device=self.device)
        except Exception as e:
            print(f"GPU模型设置错误: {e}")
            # 失败时尝试CPU
            try:
                return YOLO(model_path, task='detect', device='cpu')
            except Exception as e2:
                print(f"CPU备用模型设置错误: {e2}")
                return None
    
    def infer(self, frame, model, confidence=0.25):  # 降低默认置信度阈值
        """在GPU上执行推理"""
        if frame is None or model is None:
            return None, []
            
        if not self.gpu_available:
            # 如果GPU不可用，使用CPU处理器
            cpu_processor = CPUInferenceProcessor()
            return cpu_processor.infer(frame, model, confidence)
            
        try:
            # 执行推理
            results = model(frame, conf=confidence, verbose=False)
            
            # 解析结果
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'class': cls,
                            'name': result.names[cls],
                            'confidence': float(conf),
                            'box': [int(x1), int(y1), int(x2), int(y2)]
                        })
                        
                    print(f"GPU检测到 {len(detections)} 个物体")
            
            return frame, detections
            
        except Exception as e:
            print(f"GPU推理错误: {e}")
            # 失败时尝试CPU
            cpu_processor = CPUInferenceProcessor()
            return cpu_processor.infer(frame, model, confidence)


class ArrowButton(QPushButton):
    """自定义方向按钮，绘制无填充的箭头形状"""
    def __init__(self, direction, parent=None):
        super().__init__("", parent)
        self.direction = direction
        self.setFixedSize(70, 70)  # 调整大小，使按钮更小
        self.setCheckable(True)
        self.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
            }
            QPushButton:checked {
                background: rgba(0, 212, 255, 40);
                border-radius: 35px;
            }
        """)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 根据按钮状态确定画笔颜色
        if self.isChecked():
            pen = QPen(QColor("#00D4FF"), 2.5)
        else:
            pen = QPen(QColor("#00D4FF"), 2)
        
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)  # 无填充
        
        w, h = self.width(), self.height()
        center_x, center_y = w//2, h//2
        
        # 根据方向绘制不同的箭头形状
        if self.direction == "前进":
            # 三角形箭头指向上
            points = [
                QPoint(center_x, center_y-20),  # 顶点
                QPoint(center_x-20, center_y+10),  # 左下角
                QPoint(center_x+20, center_y+10)   # 右下角
            ]
            painter.drawPolygon(QPolygon(points))
            
        elif self.direction == "后退":
            # 三角形箭头指向下
            points = [
                QPoint(center_x, center_y+20),  # 底点
                QPoint(center_x-20, center_y-10),  # 左上角
                QPoint(center_x+20, center_y-10)   # 右上角
            ]
            painter.drawPolygon(QPolygon(points))
            
        elif self.direction == "左转":
            # 三角形箭头指向左
            points = [
                QPoint(center_x-20, center_y),  # 左点
                QPoint(center_x+10, center_y-20),  # 右上角
                QPoint(center_x+10, center_y+20)   # 右下角
            ]
            painter.drawPolygon(QPolygon(points))
            
        elif self.direction == "右转":
            # 三角形箭头指向右
            points = [
                QPoint(center_x+20, center_y),  # 右点
                QPoint(center_x-10, center_y-20),  # 左上角
                QPoint(center_x-10, center_y+20)   # 左下角
            ]
            painter.drawPolygon(QPolygon(points))
            
        elif self.direction == "复位":
            # 简单空心圆
            radius = 20
            painter.drawEllipse(center_x-radius, center_y-radius, radius*2, radius*2)


class VideoStreamWidget(QLabel):
    """优化的视频流显示控件 - 使用资源管理器处理帧"""
    frame_ready = Signal(np.ndarray)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignCenter)
        
        # 初始化资源管理器
        self.resource_manager = ResourceManager()
        self.video_processor = self.resource_manager.get_video_processor()
        
        # 视频处理配置
        self.fps_limit = 30
        self.last_frame_time = 0
        self.frame_skip_count = 0
        self.skip_frames = 2  # 每3帧处理1帧
        
        # 样式设置
        self.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1A1F2E, stop:1 #0F1419);
                border: 3px solid #00D4FF;
                border-radius: 15px;
                box-shadow: 0 0 20px rgba(0, 212, 255, 100);
            }
        """)
        
        # 显示默认图像
        self.show_default_image()
        
        # YOLO检测器
        self.yolo_detector = None
        self.detection_enabled = False
        
        # 性能监控
        self.fps_counter = 0
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)
        self.current_fps = 0
        
        # 视频捕获
        self.capture = None
        self.is_capturing = False
        self.capture_thread = None
        self.frame_queue = Queue(maxsize=10)  # 限制队列大小避免内存溢出
        
        # 当前显示的帧
        self.current_frame = None
        
    def setup_camera(self, camera_id=0):
        """设置摄像头捕获"""
        try:
            # 获取推荐的摄像头API
            api = self.resource_manager.get_camera_api()
            
            # 尝试打开摄像头
            self.capture = cv2.VideoCapture(camera_id, api)
            
            # 设置摄像头属性
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps_limit)
            
            if self.capture.isOpened():
                print(f"摄像头已连接: {camera_id}，使用API: {api}")
                return True
            else:
                print(f"摄像头连接失败: {camera_id}")
                return False
                
        except Exception as e:
            print(f"摄像头设置错误: {e}")
            return False
            
    def start_capture(self):
        """开始捕获视频"""
        if self.capture is None:
            if not self.setup_camera():
                return False
                
        if self.is_capturing:
            return True
            
        self.is_capturing = True
        
        # 启动捕获线程
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        return True
    
    def stop_capture(self):
        """停止捕获视频"""
        self.is_capturing = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
            
        if self.capture:
            self.capture.release()
            self.capture = None
            
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
                
        # 显示默认图像
        self.show_default_image()
        
    def _capture_loop(self):
        """视频捕获循环 - 在单独线程中运行"""
        while self.is_capturing and self.capture and self.capture.isOpened():
            try:
                ret, frame = self.capture.read()
                
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                    
                # 使用帧跳过减少CPU/GPU负载
                if self.frame_skip_count < self.skip_frames:
                    self.frame_skip_count += 1
                    continue
                    
                self.frame_skip_count = 0
                
                # 避免队列满时阻塞线程
                if not self.frame_queue.full():
                    self.frame_queue.put(frame, block=False)
                    
                # 添加小延迟以限制帧率
                time.sleep(1.0 / (self.fps_limit * 1.5))  # 略高于目标帧率以考虑处理时间
                
            except Exception as e:
                print(f"视频捕获错误: {e}")
                time.sleep(0.1)
                
        print("视频捕获线程已结束")
        
    def process_frame_queue(self):
        """处理帧队列 - 在UI线程中调用"""
        if self.frame_queue.empty():
            return
            
        try:
            # 获取最新帧
            frame = self.frame_queue.get_nowait()
            
            # 实际处理帧
            self.update_frame(frame)
            
        except Exception as e:
            print(f"帧队列处理错误: {e}")
        
    def show_default_image(self):
        """显示默认图像"""
        pixmap = QPixmap(640, 480)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 渐变背景
        gradient = QLinearGradient(0, 0, pixmap.width(), pixmap.height())
        gradient.setColorAt(0, QColor(26, 31, 46))
        gradient.setColorAt(1, QColor(15, 20, 25))
        painter.fillRect(pixmap.rect(), gradient)
        
        # 边框
        painter.setPen(QPen(QColor("#00D4FF"), 3))
        painter.drawRect(10, 10, pixmap.width()-20, pixmap.height()-20)
        
        # 文字
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, 
                        "🎥 等待视频流...\n\n连接摄像头后将显示实时画面")
        
        # 网格线
        painter.setPen(QPen(QColor("#00D4FF"), 1, Qt.DotLine))
        for i in range(0, pixmap.width(), 50):
            painter.drawLine(i, 0, i, pixmap.height())
        for i in range(0, pixmap.height(), 50):
            painter.drawLine(0, i, pixmap.width(), i)
            
        painter.end()
        self.setPixmap(pixmap)
        self.current_frame = None
    
    def update_frame(self, cv_frame):
        """更新视频帧 - 优化性能"""
        current_time = time.time()
        
        # 限制帧率
        if current_time - self.last_frame_time < 1.0 / self.fps_limit:
            return
        
        self.last_frame_time = current_time
        
        try:
            # 保存当前帧
            self.current_frame = cv_frame.copy()
            
            # 使用资源管理器处理帧
            processed_frame = self.video_processor.process_frame(cv_frame.copy())
            
            # YOLO检测
            if self.detection_enabled and self.yolo_detector:
                processed_frame = self.yolo_detector.process_frame(processed_frame)
            
            # 转换为Qt格式
            qt_image = self.video_processor.convert_to_qt(processed_frame)
            if qt_image:
                pixmap = QPixmap.fromImage(qt_image)
                self.setPixmap(pixmap)
                
                # 发射信号
                self.frame_ready.emit(cv_frame)
                
                # 更新FPS计数
                self.fps_counter += 1
                
        except Exception as e:
            print(f"视频帧处理错误: {e}")
    
    def update_fps(self):
        """更新FPS显示"""
        self.current_fps = self.fps_counter
        self.fps_counter = 0
    
    def enable_yolo_detection(self, enable=True):
        """启用/禁用YOLO检测"""
        if not YOLO_AVAILABLE:
            return False
            
        if enable and not self.yolo_detector:
            try:
                self.yolo_detector = YOLODetector()
                self.detection_enabled = True
                return True
            except Exception as e:
                print(f"YOLO初始化失败: {e}")
                return False
        else:
            self.detection_enabled = enable
            return True
    
    def get_current_frame(self):
        """获取当前显示的帧"""
        return self.current_frame


class YOLODetector:
    """YOLO目标检测器 - 简化版"""
    def __init__(self):
        """初始化YOLO检测器"""
        # 检查YOLO是否可用
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
            print("成功导入YOLO库")
        except ImportError:
            print("警告: 未找到YOLO库，目标检测功能不可用")
            raise ImportError("YOLO not available - 请安装ultralytics库")
        
        # 直接加载模型，不使用资源管理器中转
        self.model = None
        self.load_model(YOLO_MODEL_PATH)
        
        # 检测结果存储
        self.last_detections = []
    
    def load_model(self, model_path):
        """加载YOLO模型"""
        try:
            print(f"正在加载YOLO模型: {model_path}")
            
            # 直接加载模型，指定使用CPU
            self.model = self.YOLO(model_path)
            
            # 检查模型是否正确加载
            if self.model is None:
                print("模型加载失败!")
                return False
                
            print(f"YOLO模型加载成功: {model_path}")
            return True
            
        except Exception as e:
            print(f"模型加载错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def detect(self, frame, draw=True):
        """执行目标检测"""
        if frame is None or self.model is None:
            print("帧为空或模型未加载，无法执行检测")
            return frame, []
        
        try:
            # 记录原始帧尺寸
            if YOLO_DEBUG:
                h, w = frame.shape[:2]
                print(f"执行检测: 帧尺寸={w}x{h}")
                
                # 保存原始帧用于调试
                timestamp = datetime.now().strftime("%H%M%S")
                cv2.imwrite(f"debug_input_{timestamp}.jpg", frame)
            
            # 直接使用模型进行预测，设置低置信度阈值
            results = self.model.predict(frame, conf=YOLO_CONFIDENCE, verbose=False)
            
            # 解析结果
            detections = []
            
            if results and len(results) > 0:
                result = results[0]  # 获取第一帧结果
                
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # 获取边界框
                        box = boxes[i]
                        
                        try:
                            # 获取坐标 (XYXY格式 - 左上右下)
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            
                            # 获取置信度和类别ID
                            conf = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            
                            # 获取类别名称
                            cls_name = result.names[cls_id]
                            
                            # 添加到检测列表
                            detections.append({
                                'class': cls_id,
                                'name': cls_name,
                                'confidence': conf,
                                'box': [x1, y1, x2, y2]
                            })
                        except Exception as e:
                            print(f"解析单个检测结果时出错: {e}")
                            continue
            
            # 保存最近的检测结果
            self.last_detections = detections
            
            if YOLO_DEBUG:
                print(f"检测到 {len(detections)} 个物体")
                for i, det in enumerate(detections):
                    print(f"  物体 {i+1}: {det['name']} (置信度: {det['confidence']:.2f})")
            
            # 在图像上绘制检测结果
            if draw and detections:
                output_frame = frame.copy()
                for det in detections:
                    try:
                        x1, y1, x2, y2 = det['box']
                        name = det['name']
                        conf = det['confidence']
                        
                        # 绘制边界框
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 绘制标签背景
                        label_size = cv2.getTextSize(f'{name} {conf:.2f}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(output_frame, (x1, y1-25), (x1+label_size[0]+10, y1), (0, 255, 0), -1)
                        
                        # 绘制标签文本
                        cv2.putText(output_frame, f'{name} {conf:.2f}', (x1+5, y1-7),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    except Exception as e:
                        print(f"绘制检测框时出错: {e}")
                
                if YOLO_DEBUG:
                    # 保存结果帧用于调试
                    timestamp = datetime.now().strftime("%H%M%S")
                    cv2.imwrite(f"debug_output_{timestamp}.jpg", output_frame)
                
                return output_frame, detections
            
            return frame, detections
            
        except Exception as e:
            print(f"检测过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame, []
    
    def get_detections(self):
        """获取最新的检测结果"""
        return self.last_detections
        
    def process_frame(self, frame):
        """为保持接口兼容，提供process_frame方法"""
        processed_frame, _ = self.detect(frame)
        return processed_frame


class ResponsiveMapWidget(QWebEngineView):
    """响应式地图控件"""
    location_updated = Signal(float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_lat = DEFAULT_LATITUDE
        self.current_lng = DEFAULT_LONGITUDE
        self.setMinimumSize(400, 300)
        
        # 样式
        self.setStyleSheet("""
            QWebEngineView {
                border: 3px solid #00D4FF;
                border-radius: 15px;
                background: #1A1F2E;
                box-shadow: 0 0 20px rgba(0, 212, 255, 100);
            }
        """)
        
        self.load_map()
        
    def load_map(self):
        """加载优化的地图"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>机器蛇位置监控</title>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <style>
                body {{ 
                    margin: 0; 
                    padding: 0; 
                    background: #1A1F2E; 
                    font-family: 'Microsoft YaHei', sans-serif;
                }}
                #map {{ 
                    height: 100vh; 
                    width: 100%; 
                    border-radius: 15px;
                }}
                .info-panel {{
                    position: absolute;
                    top: 15px;
                    right: 15px;
                    background: rgba(0, 212, 255, 0.95);
                    color: white;
                    padding: 12px;
                    border-radius: 10px;
                    font-weight: bold;
                    z-index: 1000;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                    min-width: 180px;
                }}
                .status-indicator {{
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    background: #00FF88;
                    border-radius: 50%;
                    margin-right: 8px;
                    animation: pulse 2s infinite;
                }}
                @keyframes pulse {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0.5; }}
                    100% {{ opacity: 1; }}
                }}
                .coords {{ font-family: 'Consolas', monospace; font-size: 11px; }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <div class="info-panel">
                <div><span class="status-indicator"></span>机器蛇实时位置</div>
                <div class="coords" id="coordinates">
                    经度: {self.current_lng:.6f}<br>
                    纬度: {self.current_lat:.6f}
                </div>
                <div style="font-size: 10px; margin-top: 8px;" id="lastUpdate">
                    更新时间: {datetime.now().strftime('%H:%M:%S')}
                </div>
            </div>
            
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <script>
                // 初始化地图
                var map = L.map('map', {{
                    center: [{self.current_lat}, {self.current_lng}],
                    zoom: 16,
                    zoomControl: true,
                    scrollWheelZoom: true
                }});
                
                // 使用高清地图瓦片
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '© OpenStreetMap contributors',
                    maxZoom: 19
                }}).addTo(map);
                
                // 机器蛇图标
                var robotIcon = L.divIcon({{
                    className: 'robot-marker',
                    html: `<div style="
                        background: radial-gradient(circle, #00D4FF 0%, #0088CC 100%);
                        width: 24px; height: 24px; border-radius: 50%;
                        border: 3px solid white;
                        box-shadow: 0 0 15px #00D4FF;
                        position: relative;
                    ">
                        <div style="
                            position: absolute; top: 50%; left: 50%;
                            transform: translate(-50%, -50%);
                            color: white; font-size: 12px; font-weight: bold;
                        ">🐍</div>
                    </div>`,
                    iconSize: [30, 30],
                    iconAnchor: [15, 15]
                }});
                
                // 位置标记
                var robotMarker = L.marker([{self.current_lat}, {self.current_lng}], {{
                    icon: robotIcon
                }}).addTo(map);
                
                // 轨迹线
                var pathPoints = [[{self.current_lat}, {self.current_lng}]];
                var pathLine = L.polyline(pathPoints, {{
                    color: '#FF8C00',
                    weight: 4,
                    opacity: 0.8,
                    dashArray: '10, 5'
                }}).addTo(map);
                
                // 更新位置函数
                function updateRobotPosition(lat, lng) {{
                    robotMarker.setLatLng([lat, lng]);
                    pathPoints.push([lat, lng]);
                    
                    // 保持最近100个点
                    if (pathPoints.length > 100) {{
                        pathPoints.shift();
                    }}
                    pathLine.setLatLngs(pathPoints);
                    
                    // 更新信息面板
                    document.getElementById('coordinates').innerHTML = 
                        '经度: ' + lng.toFixed(6) + '<br>纬度: ' + lat.toFixed(6);
                    document.getElementById('lastUpdate').innerHTML = 
                        '更新时间: ' + new Date().toLocaleTimeString();
                    
                    // 平滑移动地图中心
                    map.panTo([lat, lng], {{animate: true, duration: 1}});
                }}
                
                // 模拟移动（可替换为实际GPS数据）
                let moveCounter = 0;
                setInterval(function() {{
                    moveCounter++;
                    var newLat = {self.current_lat} + Math.sin(moveCounter * 0.1) * 0.0005;
                    var newLng = {self.current_lng} + Math.cos(moveCounter * 0.1) * 0.0005;
                    updateRobotPosition(newLat, newLng);
                }}, 3000);
                
                // 添加比例尺
                L.control.scale({{position: 'bottomleft'}}).addTo(map);
            </script>
        </body>
        </html>
        """
        
        self.setHtml(html_content)
    
    def update_position(self, lat, lng):
        """更新机器蛇位置"""
        self.current_lat = lat
        self.current_lng = lng
        script = f"updateRobotPosition({lat}, {lng});"
        self.page().runJavaScript(script)
        self.location_updated.emit(lat, lng)


class DashboardMetricCard(QFrame):
    """大屏风格数据卡片"""
    def __init__(self, title, icon, unit="", color="#00D4FF", parent=None):
        super().__init__(parent)
        self.title = title
        self.icon = icon
        self.unit = unit
        self.color = color
        
        self.setFixedHeight(140)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.setup_ui()
        self.update_value("N/A")
        
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(10)
        
        # 顶部：图标和标题
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        # 图标
        icon_label = QLabel(self.icon)
        icon_label.setStyleSheet(f"""
            font-size: 28pt; 
            color: {self.color};
            text-shadow: 0 0 10px {self.color};
        """)
        icon_label.setFixedSize(50, 50)
        
        # 标题
        title_label = QLabel(self.title)
        title_label.setStyleSheet("""
            color: #FFFFFF;
            font-size: 13pt;
            font-weight: bold;
            letter-spacing: 1px;
        """)
        
        header_layout.addWidget(icon_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # 数值显示
        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet(f"""
            color: {self.color};
            font-size: 24pt;
            font-weight: bold;
            font-family: 'Consolas', 'Monaco', monospace;
            text-shadow: 0 0 15px {self.color};
            padding: 8px;
        """)
        
        layout.addLayout(header_layout)
        layout.addWidget(self.value_label, 1)
        
        # 卡片样式
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(30, 35, 50, 220),
                    stop:1 rgba(20, 25, 40, 220));
                border: 2px solid {self.color};
                border-radius: 15px;
                box-shadow: 0 0 20px rgba(0, 212, 255, 80);
            }}
            QFrame:hover {{
                border: 3px solid {self.color};
                box-shadow: 0 0 25px rgba(0, 212, 255, 120);
            }}
        """)
    
    def update_value(self, value, status_color=None):
        """更新数值"""
        if status_color is None:
            status_color = self.color
            
        display_text = f"{value} {self.unit}" if self.unit else str(value)
        self.value_label.setText(display_text)
        
        # 更新颜色
        self.value_label.setStyleSheet(f"""
            color: {status_color};
            font-size: 24pt;
            font-weight: bold;
            font-family: 'Consolas', 'Monaco', monospace;
            text-shadow: 0 0 15px {status_color};
            padding: 8px;
        """)


class AdvancedGaugeWidget(QWidget):
    """高级仪表盘控件"""
    def __init__(self, title="", min_val=0, max_val=100, unit="", parent=None):
        super().__init__(parent)
        self.title = title
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit
        self.current_value = min_val
        self.target_value = min_val
        
        self.setMinimumSize(300, 300)
        self.setMaximumSize(400, 400)
        
        # 动画效果
        self.animation = QPropertyAnimation(self, b"value")
        self.animation.setDuration(1000)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # 定义颜色段
        self.color_segments = [
            {"range": (0, 20), "color": QColor("#00FF88"), "label": "优秀"},
            {"range": (21, 40), "color": QColor("#00D4FF"), "label": "良好"},
            {"range": (41, 60), "color": QColor("#FFD700"), "label": "一般"},
            {"range": (61, 80), "color": QColor("#FF8C00"), "label": "较差"},
            {"range": (81, 100), "color": QColor("#FF6B6B"), "label": "危险"}
        ]
        
    @Property(float)
    def value(self):
        return self.current_value
    
    @value.setter 
    def value(self, val):
        self.current_value = val
        self.update()
        
    def set_value(self, value, animated=True):
        """设置数值"""
        self.target_value = max(self.min_val, min(self.max_val, value))
        
        if animated:
            self.animation.setStartValue(self.current_value)
            self.animation.setEndValue(self.target_value)
            self.animation.start()
        else:
            self.current_value = self.target_value
            self.update()
    
    def get_current_color(self):
        """获取当前数值对应的颜色"""
        normalized_value = (self.current_value - self.min_val) / (self.max_val - self.min_val) * 100
        
        for segment in self.color_segments:
            min_seg, max_seg = segment["range"]
            if min_seg <= normalized_value <= max_seg:
                return segment["color"], segment["label"]
        
        return QColor("#AAAAAA"), "未知"
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) / 2 - 30
        
        # 绘制外圈装饰
        painter.setPen(QPen(QColor("#00D4FF"), 4))
        outer_rect = QRectF(center.x() - radius - 10, center.y() - radius - 10,
                           (radius + 10) * 2, (radius + 10) * 2)
        painter.drawEllipse(outer_rect)
        
        # 绘制仪表盘背景
        gauge_rect = QRectF(center.x() - radius * 0.8, center.y() - radius * 0.8,
                           radius * 1.6, radius * 1.6)
        
        start_angle = 225  # 起始角度
        span_angle = 270   # 跨度角度
        
        # 绘制颜色段
        current_angle = start_angle
        segment_width = radius * 0.2
        
        for segment in self.color_segments:
            min_seg, max_seg = segment["range"]
            color = segment["color"]
            
            seg_span = (max_seg - min_seg) / 100 * span_angle
            
            painter.setPen(QPen(color, segment_width))
            painter.drawArc(gauge_rect, int(current_angle * 16), int(seg_span * 16))
            current_angle += seg_span
        
        # 绘制刻度
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        for i in range(0, 101, 10):
            angle = start_angle + (i / 100) * span_angle
            inner_radius = radius * 0.7
            outer_radius = radius * 0.75
            
            start_x = center.x() + inner_radius * math.cos(math.radians(angle))
            start_y = center.y() + inner_radius * math.sin(math.radians(angle))
            end_x = center.x() + outer_radius * math.cos(math.radians(angle))
            end_y = center.y() + outer_radius * math.sin(math.radians(angle))
            
            painter.drawLine(QPointF(start_x, start_y), QPointF(end_x, end_y))
        
        # 绘制指针
        current_color, current_label = self.get_current_color()
        value_percentage = (self.current_value - self.min_val) / (self.max_val - self.min_val)
        pointer_angle = start_angle + value_percentage * span_angle
        
        pointer_length = radius * 0.6
        pointer_x = center.x() + pointer_length * math.cos(math.radians(pointer_angle))
        pointer_y = center.y() + pointer_length * math.sin(math.radians(pointer_angle))
        
        # 指针阴影
        painter.setPen(QPen(QColor(0, 0, 0, 100), 6))
        painter.drawLine(QPointF(center.x() + 2, center.y() + 2), 
                        QPointF(pointer_x + 2, pointer_y + 2))
        
        # 指针主体
        painter.setPen(QPen(current_color, 4))
        painter.drawLine(center, QPointF(pointer_x, pointer_y))
        
        # 中心圆
        center_radius = 12
        gradient = QRadialGradient(center, center_radius)
        gradient.setColorAt(0, current_color.lighter(150))
        gradient.setColorAt(1, current_color.darker(120))
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.drawEllipse(center, center_radius, center_radius)
        
        # 绘制数值文本
        painter.setPen(QColor("#FFFFFF"))
        value_font = QFont("Consolas", 20, QFont.Bold)
        painter.setFont(value_font)
        
        value_text = f"{self.current_value:.1f}"
        value_rect = QRectF(center.x() - radius * 0.6, center.y() + radius * 0.2,
                           radius * 1.2, 30)
        painter.drawText(value_rect, Qt.AlignCenter, value_text)
        
        # 单位
        painter.setPen(current_color)
        unit_font = QFont("Microsoft YaHei", 12, QFont.Bold)
        painter.setFont(unit_font)
        unit_rect = QRectF(center.x() - radius * 0.6, center.y() + radius * 0.4,
                          radius * 1.2, 25)
        painter.drawText(unit_rect, Qt.AlignCenter, self.unit)
        
        # 状态标签
        painter.setPen(QColor("#FFFFFF"))
        label_font = QFont("Microsoft YaHei", 11, QFont.Bold)
        painter.setFont(label_font)
        label_rect = QRectF(center.x() - radius * 0.6, center.y() + radius * 0.6,
                           radius * 1.2, 25)
        painter.drawText(label_rect, Qt.AlignCenter, current_label)
        
        # 标题
        painter.setPen(QColor("#00D4FF"))
        title_font = QFont("Microsoft YaHei", 14, QFont.Bold)
        painter.setFont(title_font)
        title_rect = QRectF(center.x() - radius * 0.8, center.y() - radius * 0.9,
                           radius * 1.6, 30)
        painter.drawText(title_rect, Qt.AlignCenter, self.title)


class MQTTThread(QThread):
    """MQTT通信线程"""
    sensor_data_signal = Signal(dict)
    connection_signal = Signal(bool)
    video_frame_signal = Signal(np.ndarray)  # 新增视频帧信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_connected = False
        self.client = None
        
    def run(self):
        try:
            self.client = mqtt.Client(client_id=MQTT_CLIENT_ID)
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.on_disconnect = self.on_disconnect
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            
            print(f"尝试连接MQTT服务器: {MQTT_BROKER}:{MQTT_PORT}")
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_forever()
            
        except Exception as e:
            print(f"MQTT连接错误: {e}")
            self.connection_signal.emit(False)
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.is_connected = True
            self.connection_signal.emit(True)
            client.subscribe(MQTT_TOPIC_SUBSCRIBE)
            print(f"已连接MQTT并订阅: {MQTT_TOPIC_SUBSCRIBE}")
        else:
            print(f"MQTT连接失败，错误码: {rc}")
            self.connection_signal.emit(False)
    
    def on_disconnect(self, client, userdata, rc):
        self.is_connected = False
        print("MQTT连接已断开")
        self.connection_signal.emit(False)
    
    def on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            print(f"收到MQTT消息: {msg.topic}")
            
            # 处理视频帧
            if "camera_frame" in data:
                try:
                    # Base64编码的图像数据
                    import base64
                    img_bytes = base64.b64decode(data["camera_frame"])
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    cv_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if cv_frame is not None and cv_frame.size > 0:
                        # 保存接收到的帧到文件用于调试（如果启用了调试模式）
                        if YOLO_DEBUG:
                            timestamp = datetime.now().strftime("%H%M%S")
                            debug_filename = f"mqtt_frame_{timestamp}.jpg"
                            cv2.imwrite(debug_filename, cv_frame)
                            print(f"保存MQTT帧到: {debug_filename}, shape={cv_frame.shape}")
                        
                        # 发射视频帧信号
                        self.video_frame_signal.emit(cv_frame)
                    else:
                        print(f"无效的MQTT帧数据: size={0 if cv_frame is None else cv_frame.size}")
                except Exception as e:
                    print(f"视频帧解析错误: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 发送传感器数据信号
            self.sensor_data_signal.emit(data)
            
        except Exception as e:
            print(f"消息解析错误: {e}")
    
    def publish_command(self, command):
        """发布控制命令"""
        if self.is_connected and self.client:
            try:
                payload = json.dumps(command)
                self.client.publish(MQTT_TOPIC_PUBLISH, payload)
                print(f"已发布命令: {command}")
                return True
            except Exception as e:
                print(f"发布命令错误: {e}")
        else:
            print("MQTT未连接，无法发送命令")
        return False


class MainDashboard(QMainWindow):
    """主仪表板界面 - 响应式设计"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"BIRobot 智能控制系统 v2.0 - 用户: {CURRENT_USER}")
        self.setMinimumSize(1800, 1200)  # 适配大屏幕
        
        # 初始化资源管理器
        self.resource_manager = ResourceManager()
        
        # 数据存储
        self.sensor_data = []
        self.chart_data = {"temperature": [], "pressure": [], "air_quality": [], "timestamps": []}
        self.max_data_points = 50
        
        # 当前状态
        self.current_mode = None
        self.current_direction = None
        
        # 视频录制
        self.is_recording = False
        self.video_writer = None
        self.recording_file_path = ""
        self.recording_fps = 20
        self.recording_frame_size = (640, 480)
        
        # YOLO检测
        self.yolo_detector = None
        self.mqtt_detection_enabled = False
        self.last_mqtt_frame = None  # 保存最近的MQTT帧用于检测
        
        # 设置界面
        self.setup_ui()
        self.setup_connections()
        
        # 启动MQTT
        self.mqtt_thread = MQTTThread()
        self.mqtt_thread.sensor_data_signal.connect(self.handle_sensor_data)
        self.mqtt_thread.connection_signal.connect(self.update_connection_status)
        self.mqtt_thread.video_frame_signal.connect(self.process_mqtt_frame)
        self.mqtt_thread.start()
        
        # 定时器
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui_time)
        self.ui_timer.start(1000)
        
        # 视频处理定时器
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.process_video_frames)
        self.video_timer.setInterval(33)  # 约30fps
        
        # MQTT视频检测定时器
        self.mqtt_detection_timer = QTimer()
        self.mqtt_detection_timer.timeout.connect(self.process_mqtt_detection)
        self.mqtt_detection_timer.setInterval(100)  # 每100ms处理一次
        
        # 打印资源信息
        resources = self.resource_manager.get_resources_summary()
        print(f"系统资源信息: {resources}")
        
        self.showMaximized()  # 全屏显示
    
    def setup_ui(self):
        """设置响应式UI"""
        # 全局样式
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0A0E14, stop:0.5 #1A1F2E, stop:1 #0F1419);
                color: #FFFFFF;
            }
            QGroupBox {
                font-size: 16pt;
                font-weight: bold;
                color: #00D4FF;
                border: 3px solid #00D4FF;
                border-radius: 15px;
                margin-top: 30px;
                padding-top: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(30, 35, 50, 180),
                    stop:1 rgba(20, 25, 40, 180));
                box-shadow: 0 0 20px rgba(0, 212, 255, 50);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 25px;
                padding: 0 20px;
                color: #00D4FF;
                text-shadow: 0 0 15px #00D4FF;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局 - 垂直布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # 顶部状态栏
        self.create_top_header(main_layout)
        
        # 主内容区 - 三列布局
        content_layout = QHBoxLayout()
        content_layout.setSpacing(25)
        
        # 左侧控制面板
        left_panel = self.create_control_panel()
        left_panel.setFixedWidth(400)
        
        # 中间监控区域
        center_panel = self.create_monitoring_panel()
        
        # 右侧数据面板
        right_panel = self.create_data_panel()
        right_panel.setFixedWidth(450)
        
        content_layout.addWidget(left_panel)
        content_layout.addWidget(center_panel, 1)
        content_layout.addWidget(right_panel)
        
        main_layout.addLayout(content_layout, 1)
        
        # 底部状态栏
        self.create_status_bar()
    
    def create_top_header(self, layout):
        """创建顶部标题栏 - 已简化"""
        header_frame = QFrame()
        header_frame.setFixedHeight(100)
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 212, 255, 120),
                    stop:0.5 rgba(0, 255, 136, 120),
                    stop:1 rgba(255, 140, 0, 120));
                border: 3px solid #00D4FF;
                border-radius: 20px;
                box-shadow: 0 0 30px rgba(0, 212, 255, 100);
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(30, 15, 30, 15)
        
        # 主标题，居中显示
        title_label = QLabel("🐍 BIRobot 机器蛇智能控制系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
                       font-size: 32pt;
            font-weight: bold;
            color: #FFFFFF;
            text-shadow: 0 0 20px #00D4FF;
            letter-spacing: 2px;
        """)
        
        header_layout.addWidget(title_label)
        layout.addWidget(header_frame)
    
    def create_control_panel(self):
        """创建控制面板"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(25)
        
        # 模式选择
        mode_group = QGroupBox("🎮 运动模式选择")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(15)
        
        self.mode_buttons = []
        for mode in GAIT_MODES:
            btn = QPushButton(mode)
            btn.setCheckable(True)
            btn.setFixedHeight(60)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(30, 35, 50, 200),
                        stop:1 rgba(20, 25, 40, 200));
                    color: #FFFFFF;
                    border: 3px solid {GAIT_COLORS.get(mode, '#00D4FF')};
                    border-radius: 15px;
                    font-size: 16pt;
                    font-weight: bold;
                    padding: 15px;
                }}
                QPushButton:checked {{
                    background: {GAIT_COLORS.get(mode, '#00D4FF')};
                    color: #000000;
                    box-shadow: 0 0 25px {GAIT_COLORS.get(mode, '#00D4FF')};
                }}
                QPushButton:hover:!checked {{
                    background: rgba(50, 60, 80, 220);
                    box-shadow: 0 0 15px rgba(50, 60, 80, 100);
                }}
            """)
            btn.clicked.connect(lambda checked, m=mode: self.select_mode(m))
            mode_layout.addWidget(btn)
            self.mode_buttons.append(btn)
        
        # 方向控制 - 使用自定义箭头按钮
        direction_group = QGroupBox("🎯 方向控制")
        direction_widget = QWidget()
        direction_layout = QGridLayout(direction_widget)
        direction_layout.setSpacing(20)
        direction_layout.setContentsMargins(40, 50, 40, 40)
        
        self.direction_buttons = {}
        positions = {
            "前进": (0, 1), "左转": (1, 0), "复位": (1, 1),
            "右转": (1, 2), "后退": (2, 1)
        }
        
        for direction, (row, col) in positions.items():
            btn = ArrowButton(direction)
            btn.clicked.connect(lambda checked, d=direction: self.select_direction(d))
            direction_layout.addWidget(btn, row, col, Qt.AlignCenter)
            self.direction_buttons[direction] = btn
        
        direction_group_layout = QVBoxLayout(direction_group)
        direction_group_layout.addWidget(direction_widget)
        
        # 状态显示
        status_group = QGroupBox("📊 当前状态")
        status_layout = QVBoxLayout(status_group)
        
        self.mode_status_label = QLabel("模式: 未选择")
        self.mode_status_label.setStyleSheet("""
            background: rgba(0, 212, 255, 30);
            border: 2px solid rgba(0, 212, 255, 150);
            border-radius: 12px;
            padding: 15px;
            font-size: 16pt;
            font-weight: bold;
            color: #00D4FF;
            text-shadow: 0 0 10px #00D4FF;
        """)
        
        self.data_status_label = QLabel("数据: 0 条记录")
        self.data_status_label.setStyleSheet(self.mode_status_label.styleSheet())
        
        # 控制按钮
        control_btn_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 重置")
        self.clear_btn = QPushButton("🗑️ 清空")
        
        for btn in [self.reset_btn, self.clear_btn]:
            btn.setFixedHeight(50)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 107, 107, 120);
                    border: 3px solid #FF6B6B;
                    border-radius: 12px;
                    font-size: 14pt;
                    font-weight: bold;
                    color: white;
                    padding: 10px;
                }
                QPushButton:hover {
                    background: rgba(255, 107, 107, 180);
                    box-shadow: 0 0 20px rgba(255, 107, 107, 100);
                }
            """)
        
        control_btn_layout.addWidget(self.reset_btn)
        control_btn_layout.addWidget(self.clear_btn)
        
        status_layout.addWidget(self.mode_status_label)
        status_layout.addWidget(self.data_status_label)
        status_layout.addLayout(control_btn_layout)
        
        control_layout.addWidget(mode_group, 1)
        control_layout.addWidget(direction_group, 1)
        control_layout.addWidget(status_group, 1)
        
        return control_widget
    
    def create_monitoring_panel(self):
        """创建监控面板"""
        monitor_widget = QWidget()
        monitor_layout = QVBoxLayout(monitor_widget)
        monitor_layout.setSpacing(20)
        
        # 监控标题
        monitor_title = QLabel("📹 实时监控中心")
        monitor_title.setAlignment(Qt.AlignCenter)
        monitor_title.setFixedHeight(70)
        monitor_title.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(0, 212, 255, 100),
                stop:1 rgba(0, 255, 136, 100));
            border: 3px solid #00D4FF;
            border-radius: 15px;
            font-size: 22pt;
            font-weight: bold;
            color: #FFFFFF;
            text-shadow: 0 0 15px #FFFFFF;
        """)
        
        # 监控区域分割器
        monitor_splitter = QSplitter(Qt.Horizontal)
        monitor_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #00D4FF;
                width: 4px;
                border-radius: 2px;
            }
        """)
        
        # 视频监控
        video_group = QGroupBox("📺 视频监控")
        video_layout = QVBoxLayout(video_group)
        
        self.video_widget = VideoStreamWidget()
        self.video_widget.setMinimumSize(700, 500)
        
        # 视频控制按钮
        video_controls = QHBoxLayout()
        video_controls.setSpacing(15)
        
        self.play_btn = QPushButton("▶️ 开始")
        self.stop_btn = QPushButton("⏹️ 停止")
        self.record_btn = QPushButton("🔴 录制")
        self.yolo_btn = QPushButton("🎯 检测")
        
        for btn in [self.play_btn, self.stop_btn, self.record_btn, self.yolo_btn]:
            btn.setFixedHeight(45)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(0, 212, 255, 120);
                    border: 2px solid #00D4FF;
                    border-radius: 10px;
                    font-size: 12pt;
                    font-weight: bold;
                    color: white;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background: rgba(0, 212, 255, 180);
                    box-shadow: 0 0 15px rgba(0, 212, 255, 100);
                }
            """)
        
        video_controls.addWidget(self.play_btn)
        video_controls.addWidget(self.stop_btn)
        video_controls.addWidget(self.record_btn)
        video_controls.addWidget(self.yolo_btn)
        video_controls.addStretch()
        
        video_layout.addWidget(self.video_widget)
        video_layout.addLayout(video_controls)
        
        # 地图监控
        map_group = QGroupBox("🗺️ 位置追踪")
        map_layout = QVBoxLayout(map_group)
        
        self.map_widget = ResponsiveMapWidget()
        self.map_widget.setMinimumSize(500, 500)
        
        map_layout.addWidget(self.map_widget)
        
        monitor_splitter.addWidget(video_group)
        monitor_splitter.addWidget(map_group)
        monitor_splitter.setSizes([700, 500])
        
        # 图表区域 - 修改为更大的尺寸
        chart_group = QGroupBox("📈 传感器数据趋势")
        chart_group.setMinimumHeight(450)  # 增加最小高度
        chart_layout = QVBoxLayout(chart_group)
        
        # 创建图表
        self.chart = QChart()
        self.chart.setTitle("实时传感器数据")
        self.chart.setTitleFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        self.chart.setTitleBrush(QColor("#00D4FF"))
        self.chart.setBackgroundVisible(False)
        
        # 数据系列
        self.temp_series = QLineSeries()
        self.temp_series.setName("温度 (°C)")
        self.temp_series.setPen(QPen(QColor("#FF6B6B"), 4))
        
        self.pressure_series = QLineSeries()
        self.pressure_series.setName("气压 (hPa/10)")
        self.pressure_series.setPen(QPen(QColor("#00D4FF"), 4))
        
        self.air_series = QLineSeries()
        self.air_series.setName("空气质量 (/10)")
        self.air_series.setPen(QPen(QColor("#00FF88"), 4))
        
        self.chart.addSeries(self.temp_series)
        self.chart.addSeries(self.pressure_series)
        self.chart.addSeries(self.air_series)
        
        # 坐标轴 - 颜色修改为蓝色
        self.axis_x = QValueAxis()
        self.axis_x.setTitleText("时间序列")
        self.axis_x.setRange(0, self.max_data_points)
        self.axis_x.setTickCount(6)
        self.axis_x.setLabelsBrush(QColor("#00D4FF"))  # 横坐标字体颜色设为蓝色
        self.axis_x.setTitleBrush(QColor("#00D4FF"))   # 标题颜色设为蓝色
        
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("传感器数值")
        self.axis_y.setRange(-10, 100)
        self.axis_y.setLabelsBrush(QColor("#00D4FF"))  # 纵坐标字体颜色设为蓝色
        self.axis_y.setTitleBrush(QColor("#00D4FF"))   # 标题颜色设为蓝色
        
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        
        for series in [self.temp_series, self.pressure_series, self.air_series]:
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
        
        # 图表视图
        chart_view = QChartView(self.chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setMinimumHeight(400)  # 增加图表高度
        chart_view.setStyleSheet("""
            QChartView {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1A1F2E, stop:1 #0F1419);
                border: 3px solid #00D4FF;
                border-radius: 15px;
            }
        """)
        
        chart_layout.addWidget(chart_view)
        
        # 图例颜色与数据线对应
        legend = self.chart.legend()
        legend.setVisible(True)
        legend.setAlignment(Qt.AlignBottom)
        legend.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        
        # 需要在数据系列都添加到图表后设置图例样式
        markers = legend.markers()
        for i, marker in enumerate(markers):
            if i == 0:  # 温度
                marker.setLabelBrush(QColor("#FF6B6B"))
            elif i == 1:  # 气压
                marker.setLabelBrush(QColor("#00D4FF"))
            elif i == 2:  # 空气质量
                marker.setLabelBrush(QColor("#00FF88"))
        
        monitor_layout.addWidget(monitor_title)
        monitor_layout.addWidget(monitor_splitter, 2)
        monitor_layout.addWidget(chart_group, 1)
        
        return monitor_widget
    
    def create_data_panel(self):
        """创建数据面板"""
        data_widget = QWidget()
        data_layout = QVBoxLayout(data_widget)
        data_layout.setSpacing(25)
        
        # 实时数据卡片
        cards_group = QGroupBox("📊 实时传感器数据")
        cards_layout = QVBoxLayout(cards_group)
        cards_layout.setSpacing(20)
        
        # 数据卡片
        self.temp_card = DashboardMetricCard("温度", "🌡️", "°C", "#FF6B6B")
        self.pressure_card = DashboardMetricCard("气压", "📊", "hPa", "#00D4FF")
        self.humidity_card = DashboardMetricCard("湿度", "💧", "%", "#00FF88")
        self.gps_card = DashboardMetricCard("GPS信号", "🛰️", "", "#FFD700")
        
        cards_layout.addWidget(self.temp_card)
        cards_layout.addWidget(self.pressure_card)
        cards_layout.addWidget(self.humidity_card)
        cards_layout.addWidget(self.gps_card)
        
        # 空气质量仪表盘
        air_group = QGroupBox("🌬️ 空气质量监测")
        air_layout = QVBoxLayout(air_group)
        
        self.air_gauge = AdvancedGaugeWidget("空气质量指数", 0, 9999, "VOC")
        air_layout.addWidget(self.air_gauge, 0, Qt.AlignCenter)
        
        # 物体检测结果
        detection_group = QGroupBox("🎯 AI物体识别")
        detection_layout = QVBoxLayout(detection_group)
        
        self.detection_info = QLabel("🎯 检测状态: 待启动")
        self.detection_info.setStyleSheet("""
            background: rgba(0, 212, 255, 30);
            border: 2px solid rgba(0, 212, 255, 100);
            border-radius: 10px;
            padding: 12px;
            font-size: 14pt;
            font-weight: bold;
            color: #00D4FF;
        """)
        
        self.detection_table = QTableWidget()
        self.detection_table.setColumnCount(3)
        self.detection_table.setHorizontalHeaderLabels(["物体", "置信度", "位置"])
        self.detection_table.horizontalHeader().setStretchLastSection(True)
        self.detection_table.setMaximumHeight(200)
        self.detection_table.setStyleSheet("""
            QTableWidget {
                background: rgba(30, 35, 50, 150);
                border: 2px solid #00D4FF;
                border-radius: 10px;
                gridline-color: #00D4FF;
                font-size: 11pt;
            }
            QHeaderView::section {
                background: #00D4FF;
                color: black;
                font-weight: bold;
                padding: 10px;
                border: none;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid rgba(0, 212, 255, 50);
            }
        """)
        
        detection_layout.addWidget(self.detection_info)
        detection_layout.addWidget(self.detection_table)
        
        data_layout.addWidget(cards_group, 1)
        data_layout.addWidget(air_group, 1)
        data_layout.addWidget(detection_group, 1)
        return data_widget
    
    def create_status_bar(self):
        """创建状态栏"""
        status_bar = QStatusBar()
        status_bar.setFixedHeight(50)
        status_bar.setStyleSheet("""
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 212, 255, 80),
                    stop:1 rgba(0, 150, 200, 80));
                color: white;
                border-top: 3px solid #00D4FF;
                font-weight: bold;
                font-size: 12pt;
                padding: 8px;
            }
        """)
        
        # 显示资源管理信息
        resources = self.resource_manager.get_resources_summary()
        gpu_info = "GPU" if resources["gpu_available"] else "CPU"
        status_bar.showMessage(f"🚀 系统已启动 - 使用{gpu_info}处理视频 - 等待传感器数据连接...")
        
        self.setStatusBar(status_bar)
    
    def setup_connections(self):
        """设置信号连接"""
        # 按钮连接
        self.export_btn = QPushButton("📊 导出数据")
        self.export_btn.setFixedSize(140, 60)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 255, 136, 150),
                    stop:1 rgba(0, 200, 100, 150));
                color: white;
                border: 3px solid #00FF88;
                border-radius: 15px;
                font-size: 12pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(0, 255, 136, 200);
                box-shadow: 0 0 20px rgba(0, 255, 136, 150);
            }
        """)
        self.export_btn.clicked.connect(self.export_sensor_data)
        self.reset_btn.clicked.connect(self.reset_charts)
        self.clear_btn.clicked.connect(self.clear_all_data)
        
        # 视频控制
        self.play_btn.clicked.connect(self.start_video_stream)
        self.stop_btn.clicked.connect(self.stop_video_stream)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.yolo_btn.clicked.connect(self.toggle_yolo_detection)
        
        # 视频流信号
        self.video_widget.frame_ready.connect(self.process_video_frame)
    
    def update_ui_time(self):
        """更新时间显示"""
        # 此处已删除，不再显示时间
        pass
    
    def update_connection_status(self, connected):
        """更新连接状态"""
        resources = self.resource_manager.get_resources_summary()
        processor_type = "GPU" if resources["gpu_available"] else "CPU"
        
        if connected:
            self.statusBar().showMessage(f"🌐 MQTT服务器连接成功 - 使用{processor_type}处理视频 - 传感器数据流已建立")
        else:
            self.statusBar().showMessage(f"❌ MQTT连接失败 - 使用{processor_type}处理视频 - 请检查网络设置")
    
    def select_mode(self, mode):
        """选择运动模式"""
        # 取消所有按钮选中状态
        for btn in self.mode_buttons:
            btn.setChecked(False)
        
        if self.current_mode == mode:
            # 如果点击的是当前模式，则取消选择
            self.current_mode = None
            self.mode_status_label.setText("模式: 未选择")
            self.statusBar().showMessage("❌ 已取消模式选择")
        else:
            # 选择新模式
            for btn in self.mode_buttons:
                if btn.text() == mode:
                    btn.setChecked(True)
                    break
            
            self.current_mode = mode
            self.mode_status_label.setText(f"模式: {mode}")
            self.statusBar().showMessage(f"✅ 已选择运动模式: {mode}")
            
            # 如果是复位模式，直接发送命令
            if mode == "复位模式":
                self.send_robot_command({"mode": mode, "direction": "复位"})
    
    def select_direction(self, direction):
        """选择移动方向"""
        # 更新方向按钮状态
        for name, btn in self.direction_buttons.items():
            btn.setChecked(name == direction)
        
        self.current_direction = direction
        
        # 检查是否已选择模式
        if not self.current_mode:
            QMessageBox.warning(self, "警告", 
                "⚠️ 请先选择运动模式！\n\n需要选择蠕动模式或蜿蜒模式后才能控制方向。")
            # 清除方向选择
            for btn in self.direction_buttons.values():
                btn.setChecked(False)
            return
        
        if self.current_mode == "复位模式":
            QMessageBox.information(self, "提示", 
                "ℹ️ 复位模式不支持方向控制\n\n复位模式会自动执行复位动作。")
            return
        
        # 发送控制命令
        command = {
            "mode": self.current_mode,
            "direction": direction,
            "timestamp": time.time(),
            "user": CURRENT_USER
        }
        
        success = self.send_robot_command(command)
        if success:
            self.statusBar().showMessage(f"🎯 发送控制命令: {self.current_mode} - {direction}")
        else:
            self.statusBar().showMessage("❌ 命令发送失败 - 请检查MQTT连接")
    
    def send_robot_command(self, command):
        """发送机器人控制命令"""
        if hasattr(self, 'mqtt_thread') and self.mqtt_thread.is_connected:
            return self.mqtt_thread.publish_command(command)
        return False
    
    def handle_sensor_data(self, data):
        """处理传感器数据"""
        try:
            # 提取传感器数据
            temperature = data.get("temperature", 0)
            pressure = data.get("pressure", 0)
            air_quality = data.get("air_quality", 0)
            humidity = data.get("humidity", 0)
            latitude = data.get("latitude")
            longitude = data.get("longitude")
            
            # 更新数据卡片
            self.temp_card.update_value(f"{temperature:.1f}")
            self.pressure_card.update_value(f"{pressure:.1f}")
            self.humidity_card.update_value(f"{humidity:.1f}")
            
            # 更新GPS状态
            if latitude is not None and longitude is not None:
                self.gps_card.update_value("已连接", "#00FF88")
                self.map_widget.update_position(latitude, longitude)
            else:
                self.gps_card.update_value("无信号", "#FF5252")
            
            # 更新空气质量仪表盘
            self.air_gauge.set_value(air_quality, animated=True)
            
            # 存储数据用于图表显示
            self.chart_data["temperature"].append(temperature)
            self.chart_data["pressure"].append(pressure / 10)  # 缩放显示
            self.chart_data["air_quality"].append(air_quality / 10)  # 缩放显示
            self.chart_data["timestamps"].append(len(self.chart_data["timestamps"]))
            
            # 限制数据点数量
            for key in self.chart_data:
                if len(self.chart_data[key]) > self.max_data_points:
                    self.chart_data[key].pop(0)
            
            # 重新编号时间戳
            self.chart_data["timestamps"] = list(range(len(self.chart_data["temperature"])))
            
            # 更新图表
            self.update_sensor_charts()
            
            # 保存完整数据记录
            record = {
                "timestamp": datetime.now().isoformat(),
                "temperature": temperature,
                "pressure": pressure,
                "air_quality": air_quality,
                "humidity": humidity,
                "latitude": latitude,
                "longitude": longitude
            }
            self.sensor_data.append(record)
            
            # 更新数据计数
            self.data_status_label.setText(f"数据: {len(self.sensor_data)} 条记录")
            
        except Exception as e:
            print(f"传感器数据处理错误: {e}")
            self.statusBar().showMessage(f"❌ 数据处理错误: {str(e)}")
    
    def process_mqtt_frame(self, cv_frame):
        """处理MQTT传来的视频帧"""
        try:
            if cv_frame is None or cv_frame.size == 0:
                print("收到空的MQTT视频帧")
                return
                
            # 保存当前MQTT帧
            self.last_mqtt_frame = cv_frame.copy()
            
            if YOLO_DEBUG:
                print(f"处理MQTT帧: shape={cv_frame.shape}")
            
            # 更新视频显示
            self.video_widget.update_frame(cv_frame)
            
            # 如果正在录制，保存帧
            if self.is_recording and self.video_writer is not None:
                try:
                    # 确保帧大小与视频写入器设置一致
                    h, w = cv_frame.shape[:2]
                    if (w, h) != self.recording_frame_size:
                        resized_frame = cv2.resize(cv_frame, self.recording_frame_size)
                        self.video_writer.write(resized_frame)
                    else:
                        self.video_writer.write(cv_frame)
                except Exception as e:
                    print(f"视频录制错误: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 如果启用了检测，直接在这里处理MQTT帧检测
            if self.mqtt_detection_enabled and self.yolo_detector:
                self.process_mqtt_detection()
            
        except Exception as e:
            print(f"MQTT视频帧处理错误: {e}")
            import traceback
            traceback.print_exc()
    
    def process_mqtt_detection(self):
        """处理MQTT视频帧的YOLO检测"""
        if not self.mqtt_detection_enabled or self.yolo_detector is None:
            return
            
        if self.last_mqtt_frame is None or self.last_mqtt_frame.size == 0:
            print("没有可用的MQTT帧用于检测")
            return
            
        try:
            # 创建帧的副本以防止修改原始帧
            frame_to_detect = self.last_mqtt_frame.copy()
            
            print(f"开始检测MQTT帧: 尺寸={frame_to_detect.shape}")
            
            # 直接调用检测方法，获取处理后的帧和检测结果
            processed_frame, detections = self.yolo_detector.detect(frame_to_detect)
            
            # 更新视频显示
            if processed_frame is not None:
                self.video_widget.update_frame(processed_frame)
                
            # 更新检测表格
            self.update_detection_table(detections)
            
            # 更新检测状态
            if detections:
                self.detection_info.setText(f"🎯 检测状态: 已检测到 {len(detections)} 个物体")
            else:
                self.detection_info.setText("🎯 检测状态: 未检测到物体")
            
        except Exception as e:
            print(f"MQTT检测处理错误: {e}")
            import traceback
            traceback.print_exc()
    
    def update_sensor_charts(self):
        """更新传感器数据图表"""
        try:
            # 清除现有数据
            self.temp_series.clear()
            self.pressure_series.clear()
            self.air_series.clear()
            
            # 添加新数据点
            timestamps = self.chart_data["timestamps"]
            temperatures = self.chart_data["temperature"]
            pressures = self.chart_data["pressure"]
            air_qualities = self.chart_data["air_quality"]
            
            for i, (temp, pressure, air) in enumerate(zip(temperatures, pressures, air_qualities)):
                self.temp_series.append(i, temp)
                self.pressure_series.append(i, pressure)
                self.air_series.append(i, air)
            
            # 更新坐标轴范围
            if timestamps:
                self.axis_x.setRange(0, max(len(timestamps) - 1, self.max_data_points - 1))
                
                # 计算Y轴范围
                all_values = temperatures + pressures + air_qualities
                if all_values:
                    min_val = min(all_values)
                    max_val = max(all_values)
                    margin = max((max_val - min_val) * 0.1, 5)
                    self.axis_y.setRange(min_val - margin, max_val + margin)
                    
        except Exception as e:
            print(f"图表更新错误: {e}")
    
    def process_video_frame(self, cv_frame):
        """处理视频帧"""
        # 仅处理本地视频检测结果，MQTT视频帧的检测单独处理
        if hasattr(self, 'yolo_detector') and self.yolo_detector and hasattr(self.yolo_detector, 'get_detections'):
            # 更新检测结果表格
            detections = self.yolo_detector.get_detections()
            self.update_detection_table(detections)
    
    def process_video_frames(self):
        """处理视频帧队列 - 定时器调用"""
        if hasattr(self, 'video_widget'):
            self.video_widget.process_frame_queue()
    
    def update_detection_table(self, detections):
        """更新检测结果表格"""
        try:
            self.detection_table.setRowCount(0)
            
            if not detections:
                if YOLO_DEBUG:
                    print("检测结果为空，清空表格")
                return
                
            if YOLO_DEBUG:
                print(f"更新检测表格: {len(detections)} 项")
                
            for i, detection in enumerate(detections):
                self.detection_table.insertRow(i)
                
                # 物体名称
                name_item = QTableWidgetItem(detection.get('name', 'Unknown'))
                name_item.setTextAlignment(Qt.AlignCenter)
                
                # 置信度
                conf = detection.get('confidence', 0)
                conf_item = QTableWidgetItem(f"{conf:.2f}")
                conf_item.setTextAlignment(Qt.AlignCenter)
                
                # 位置
                x1, y1, x2, y2 = detection.get('box', [0, 0, 0, 0])
                pos_item = QTableWidgetItem(f"({x1},{y1})-({x2},{y2})")
                pos_item.setTextAlignment(Qt.AlignCenter)
                
                self.detection_table.setItem(i, 0, name_item)
                self.detection_table.setItem(i, 1, conf_item)
                self.detection_table.setItem(i, 2, pos_item)
                
                # 设置颜色 - 根据置信度
                color = QColor("#00FF88") if conf > 0.7 else (
                    QColor("#FFD700") if conf > 0.5 else QColor("#FF6B6B"))
                
                for col in range(3):
                    self.detection_table.item(i, col).setForeground(color)
                    
            if YOLO_DEBUG and len(detections) > 0:
                self.detection_info.setText(f"🎯 检测状态: 已检测到 {len(detections)} 个物体")
                
        except Exception as e:
            print(f"更新检测表格错误: {e}")
            import traceback
            traceback.print_exc()
    
    def start_video_stream(self):
        """启动视频流"""
        if not hasattr(self, 'video_widget'):
            return
            
        if self.video_widget.start_capture():
            self.statusBar().showMessage("📹 视频流已启动")
            
            # 启动视频处理定时器
            if not self.video_timer.isActive():
                self.video_timer.start()
                
            # 更新按钮状态
            self.play_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "错误", "无法启动视频流。请检查摄像头连接。")
    
    def stop_video_stream(self):
        """停止视频流"""
        if hasattr(self, 'video_widget'):
            self.video_widget.stop_capture()
            
        # 停止视频处理定时器
        if self.video_timer.isActive():
            self.video_timer.stop()
            
        self.statusBar().showMessage("⏹️ 视频流已停止")
        
        # 更新按钮状态
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def toggle_recording(self):
        """切换录制状态"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """开始录制视频"""
        if self.is_recording:
            return
            
        try:
            # 获取当前帧，决定视频尺寸
            current_frame = None
            if hasattr(self, 'last_mqtt_frame') and self.last_mqtt_frame is not None:
                current_frame = self.last_mqtt_frame
            elif hasattr(self, 'video_widget') and self.video_widget.current_frame is not None:
                current_frame = self.video_widget.current_frame
            
            if current_frame is None:
                QMessageBox.warning(self, "录制失败", "没有可用的视频帧，请确保视频流已启动或MQTT视频正在接收。")
                return
                
            # 设置录制参数
            self.is_recording = True
            
            # 确定帧大小
            h, w = current_frame.shape[:2]
            self.recording_frame_size = (w, h)
            
            # 创建临时视频文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_file_path = f"temp_recording_{timestamp}.mp4"
            
            # 确定编码器
            # 尝试查找平台上可用的编解码器
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
            except:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
                except:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID编码
            
            # 创建视频写入器
            print(f"创建视频写入器: 分辨率={self.recording_frame_size}, FPS={self.recording_fps}, 路径={self.recording_file_path}")
            self.video_writer = cv2.VideoWriter(
                self.recording_file_path, 
                fourcc, 
                self.recording_fps, 
                self.recording_frame_size
            )
            
            # 检查视频写入器是否成功创建
            if not self.video_writer.isOpened():
                raise Exception(f"无法创建视频写入器，请检查编解码器是否支持: {fourcc}")
            
            # 更新UI
            self.record_btn.setText("⏹️ 停止录制")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 107, 107, 180);
                    border: 2px solid #FF6B6B;
                    border-radius: 10px;
                    font-size: 12pt;
                    font-weight: bold;
                    color: white;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background: rgba(255, 107, 107, 220);
                    box-shadow: 0 0 15px rgba(255, 107, 107, 150);
                }
            """)
            
            self.statusBar().showMessage(f"🔴 开始录制视频... 分辨率: {self.recording_frame_size[0]}x{self.recording_frame_size[1]}")
            
        except Exception as e:
            self.is_recording = False
            if hasattr(self, 'video_writer') and self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            QMessageBox.critical(self, "录制失败", f"无法开始录制视频: {str(e)}")
            print(f"录制启动错误: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_recording(self):
        """停止录制并保存视频"""
        if not self.is_recording:
            return
            
        try:
            # 停止录制
            self.is_recording = False
            
            # 释放视频写入器
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                print(f"视频写入器已释放")
            
            # 检查录制文件是否存在
            if not os.path.exists(self.recording_file_path):
                raise Exception(f"录制失败，文件不存在: {self.recording_file_path}")
                
            file_size = os.path.getsize(self.recording_file_path)
            if file_size == 0:
                raise Exception(f"录制失败，文件大小为0: {self.recording_file_path}")
                
            print(f"录制文件信息: 路径={self.recording_file_path}, 大小={file_size}字节")
                
            # 打开文件保存对话框
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "保存录制视频", 
                f"机器蛇视频_{timestamp}.mp4",
                "视频文件 (*.mp4);;所有文件 (*)"
            )
            
            if file_path:
                # 复制临时文件到用户选择的位置
                import shutil
                shutil.copy2(self.recording_file_path, file_path)
                print(f"视频已复制到: {file_path}")
                
                # 删除临时文件
                os.remove(self.recording_file_path)
                print(f"临时文件已删除: {self.recording_file_path}")
                
                self.statusBar().showMessage(f"✅ 视频已保存到: {file_path}")
                
                # 显示成功消息
                QMessageBox.information(self, "保存成功", 
                    f"视频已成功保存到:\n{file_path}")
            else:
                # 用户取消了保存，删除临时文件
                if os.path.exists(self.recording_file_path):
                    os.remove(self.recording_file_path)
                    print(f"用户取消保存，临时文件已删除: {self.recording_file_path}")
                self.statusBar().showMessage("❌ 视频保存已取消")
                
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存视频时出错: {str(e)}")
            print(f"视频保存错误: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 重置录制状态
            self.recording_file_path = ""
            
            # 更新UI
            self.record_btn.setText("🔴 录制")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(0, 212, 255, 120);
                    border: 2px solid #00D4FF;
                    border-radius: 10px;
                    font-size: 12pt;
                    font-weight: bold;
                    color: white;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background: rgba(0, 212, 255, 180);
                    box-shadow: 0 0 15px rgba(0, 212, 255, 100);
                }
            """)
    
    def toggle_yolo_detection(self):
        """切换YOLO检测"""
        try:
            # 检查YOLO是否可用
            try:
                from ultralytics import YOLO
                yolo_available = True
            except ImportError:
                yolo_available = False
            
            if not yolo_available:
                QMessageBox.warning(self, "功能不可用", 
                    "❌ YOLO检测模块未安装\n\n请安装ultralytics包以启用物体检测功能。")
                return
            
            # 获取处理器类型
            resources = self.resource_manager.get_resources_summary()
            processor_type = "GPU" if resources["gpu_available"] else "CPU"
            
            # 检查是否已初始化YOLO检测器
            if not hasattr(self, 'yolo_detector') or self.yolo_detector is None:
                # 初始化检测器
                try:
                    print("开始初始化YOLO检测器...")
                    self.yolo_detector = YOLODetector()
                    self.mqtt_detection_enabled = True
                    
                    # 启动MQTT检测定时器
                    if not self.mqtt_detection_timer.isActive():
                        self.mqtt_detection_timer.start()
                        print("MQTT检测定时器已启动")
                    
                    self.yolo_btn.setText("⏹️ 停止检测")
                    self.detection_info.setText(f"🎯 检测状态: 使用{processor_type}检测MQTT视频")
                    self.statusBar().showMessage(f"🎯 YOLO物体检测已启动 - 检测MQTT视频流")
                    
                    # 立即执行一次检测，如果有可用帧
                    if hasattr(self, 'last_mqtt_frame') and self.last_mqtt_frame is not None:
                        self.process_mqtt_detection()
                    
                except Exception as e:
                    QMessageBox.critical(self, "检测启动失败", 
                        f"❌ YOLO检测器初始化失败\n\n错误信息: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                # 切换检测状态
                new_state = not self.mqtt_detection_enabled
                self.mqtt_detection_enabled = new_state
                
                if new_state:
                    # 启动MQTT检测定时器
                    if not self.mqtt_detection_timer.isActive():
                        self.mqtt_detection_timer.start()
                        print("MQTT检测定时器已启动")
                    
                    self.yolo_btn.setText("⏹️ 停止检测")
                    self.detection_info.setText(f"🎯 检测状态: 使用{processor_type}检测MQTT视频")
                    self.statusBar().showMessage(f"🎯 YOLO物体检测已启动 - 检测MQTT视频流")
                    
                    # 立即执行一次检测，如果有可用帧
                    if hasattr(self, 'last_mqtt_frame') and self.last_mqtt_frame is not None:
                        self.process_mqtt_detection()
                else:
                    # 停止MQTT检测定时器
                    if self.mqtt_detection_timer.isActive():
                        self.mqtt_detection_timer.stop()
                        print("MQTT检测定时器已停止")
                    
                    self.yolo_btn.setText("🎯 检测")
                    self.detection_info.setText("🎯 检测状态: 已停止")
                    self.statusBar().showMessage("⏹️ YOLO物体检测已停止")
                    # 清空检测表格
                    self.detection_table.setRowCount(0)
        except Exception as e:
            print(f"切换YOLO检测时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def export_sensor_data(self):
        """导出传感器数据"""
        if not self.sensor_data:
            QMessageBox.warning(self, "导出失败", "❌ 没有可导出的数据！\n\n请等待传感器数据收集后再尝试导出。")
            return
        
        # 选择保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"BIRobot_传感器数据_{timestamp}.csv"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出传感器数据", default_filename,
            "CSV文件 (*.csv);;Excel文件 (*.xlsx);;所有文件 (*)"
        )
        
        if not file_path:
            return
        
        try:
            # 创建DataFrame
            df = pd.DataFrame(self.sensor_data)
            
            # 根据文件扩展名选择导出格式
            if file_path.endswith('.xlsx'):
                df.to_excel(file_path, index=False, sheet_name='传感器数据')
            else:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            # 生成统计报告
            stats_data = {
                '数据项': ['温度 (°C)', '气压 (hPa)', '空气质量', '湿度 (%)'],
                '最小值': [
                    df['temperature'].min() if 'temperature' in df else 'N/A',
                    df['pressure'].min() if 'pressure' in df else 'N/A',
                    df['air_quality'].min() if 'air_quality' in df else 'N/A',
                    df['humidity'].min() if 'humidity' in df else 'N/A'
                ],
                '最大值': [
                    df['temperature'].max() if 'temperature' in df else 'N/A',
                    df['pressure'].max() if 'pressure' in df else 'N/A',
                    df['air_quality'].max() if 'air_quality' in df else 'N/A',
                    df['humidity'].max() if 'humidity' in df else 'N/A'
                ],
                '平均值': [
                    f"{df['temperature'].mean():.2f}" if 'temperature' in df else 'N/A',
                    f"{df['pressure'].mean():.2f}" if 'pressure' in df else 'N/A',
                    f"{df['air_quality'].mean():.2f}" if 'air_quality' in df else 'N/A',
                    f"{df['humidity'].mean():.2f}" if 'humidity' in df else 'N/A'
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            stats_path = file_path.replace('.csv', '_统计报告.csv').replace('.xlsx', '_统计报告.xlsx')
            
            if stats_path.endswith('.xlsx'):
                stats_df.to_excel(stats_path, index=False, sheet_name='统计报告')
            else:
                stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
            
            # 显示成功消息
            QMessageBox.information(self, "导出成功", 
                f"📊 数据导出完成！\n\n"
                f"数据文件: {file_path}\n"
                f"统计报告: {stats_path}\n\n"
                f"共导出 {len(self.sensor_data)} 条记录")
            
            self.statusBar().showMessage(f"📊 数据已导出: {len(self.sensor_data)} 条记录")
            
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"❌ 导出过程中发生错误:\n\n{str(e)}")
            print(f"数据导出错误: {e}")
    
    def reset_charts(self):
        """重置图表显示"""
        if self.chart:
            self.chart.zoomReset()
        self.statusBar().showMessage("🔄 图表显示已重置")
    
    def clear_all_data(self):
        """清除所有数据"""
        reply = QMessageBox.question(self, "确认清除", 
            "⚠️ 确定要清除所有收集的数据吗？\n\n"
            "此操作将删除：\n"
            "• 所有传感器历史数据\n"
            "• 图表显示数据\n"
            "• 检测结果记录\n\n"
            "此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 清除数据
            self.sensor_data.clear()
            for key in self.chart_data:
                self.chart_data[key].clear()
            
            # 重置图表
            self.temp_series.clear()
            self.pressure_series.clear()
            self.air_series.clear()
            
            # 重置显示
            self.temp_card.update_value("N/A")
            self.pressure_card.update_value("N/A")
            self.humidity_card.update_value("N/A")
            self.gps_card.update_value("无信号", "#FF5252")
            self.air_gauge.set_value(0, animated=False)
            
            # 重置状态
            self.data_status_label.setText("数据: 0 条记录")
            self.detection_table.setRowCount(0)
            self.detection_info.setText("🎯 检测状态: 待启动")
            
            self.statusBar().showMessage("🗑️ 所有数据已清除")
    
    def resizeEvent(self, event):
        """窗口大小改变事件 - 响应式布局调整"""
        super().resizeEvent(event)
        
        # 根据窗口大小调整字体
        window_width = self.width()
        if window_width < 1600:
            font_scale = 0.8
        elif window_width > 2000:
            font_scale = 1.2
        else:
            font_scale = 1.0
        
        # 可以在这里添加更多响应式调整逻辑
    
    def closeEvent(self, event):
        """关闭事件处理"""
        # 如果正在录制，停止录制
        if self.is_recording:
            self.stop_recording()
            
        # 停止视频处理
        if hasattr(self, 'video_widget'):
            self.video_widget.stop_capture()
            
        # 停止视频处理定时器
        if hasattr(self, 'video_timer') and self.video_timer.isActive():
            self.video_timer.stop()
        
        # 停止MQTT检测定时器
        if hasattr(self, 'mqtt_detection_timer') and self.mqtt_detection_timer.isActive():
            self.mqtt_detection_timer.stop()
            
        # 停止MQTT线程
        if hasattr(self, 'mqtt_thread') and self.mqtt_thread.isRunning():
            self.mqtt_thread.quit()
            self.mqtt_thread.wait(3000)  # 等待3秒
        
        # 停止定时器
        if hasattr(self, 'ui_timer') and self.ui_timer.isActive():
            self.ui_timer.stop()
        
        print("应用程序正在关闭...")
        super().closeEvent(event)


class SplashScreenOptimized(QSplashScreen):
    """优化的启动画面"""
    def __init__(self):
        # 创建启动画面
        splash_pixmap = QPixmap(1200, 700)
        splash_pixmap.fill(QColor(10, 14, 20))
        super().__init__(splash_pixmap)
        
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        
        # 进度相关
        self.progress = 0
        self.messages = [
            "初始化核心系统...",
            "检测硬件资源...",
            "加载传感器模块...",
            "连接网络服务...",
            "初始化地图服务...",
            "配置视频处理...",
            "优化资源分配...",
            "启动用户界面...",
            "系统准备就绪！"
        ]
        self.current_message_idx = 0
        
        # 启动动画定时器
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_progress)
        self.animation_timer.start(300)  # 每300ms更新一次
        
    def update_progress(self):
        """更新进度"""
        self.progress += 8
        if self.progress <= 100:
            # 更新消息
            msg_idx = min(len(self.messages) - 1, self.progress // 12)
            if msg_idx != self.current_message_idx:
                self.current_message_idx = msg_idx
            
            self.showMessage(
                self.messages[self.current_message_idx], 
                Qt.AlignBottom | Qt.AlignHCenter, 
                QColor("#00D4FF")
            )
            self.repaint()
        else:
            self.animation_timer.stop()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 渐变背景
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(10, 14, 20))
        gradient.setColorAt(0.5, QColor(26, 31, 46))
        gradient.setColorAt(1, QColor(15, 20, 25))
        painter.fillRect(self.rect(), gradient)
        
        # 标题
        painter.setPen(QColor("#00D4FF"))
        title_font = QFont("Arial Black", 56, QFont.Bold)
        painter.setFont(title_font)
        title_rect = QRect(0, 180, self.width(), 100)
        painter.drawText(title_rect, Qt.AlignCenter, "BIRobot")
        
        # 副标题
        painter.setPen(QColor("#FFFFFF"))
        subtitle_font = QFont("Microsoft YaHei", 22, QFont.Bold)
        painter.setFont(subtitle_font)
        subtitle_rect = QRect(0, 280, self.width(), 50)
        painter.drawText(subtitle_rect, Qt.AlignCenter, "机器蛇智能控制系统 v2.0")
        
        # 版本信息 - 不显示时间
        painter.setPen(QColor("#00FF88"))
        version_font = QFont("Consolas", 14)
        painter.setFont(version_font)
        version_rect = QRect(0, 330, self.width(), 30)
        painter.drawText(version_rect, Qt.AlignCenter, f"版本: 2.0")
        
        # 显示资源信息
        resource_manager = ResourceManager()
        resources = resource_manager.get_resources_summary()
        processor_type = "GPU" if resources["gpu_available"] else "CPU"
        
        painter.setPen(QColor("#FFD700"))
        resource_font = QFont("Consolas", 12)
        painter.setFont(resource_font)
        resource_rect = QRect(0, 360, self.width(), 30)
        painter.drawText(resource_rect, Qt.AlignCenter, f"视频处理: {processor_type} | CPU核心: {resources['cpu_count']}")
        
        # 进度条
        progress_rect = QRect(250, 450, self.width() - 500, 25)
        
        # 进度条背景
        painter.setPen(QPen(QColor("#00D4FF"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(progress_rect, 12, 12)
        
        # 进度条填充
        if self.progress > 0:
            fill_width = int((progress_rect.width() * self.progress) / 100)
            fill_rect = QRect(progress_rect.x(), progress_rect.y(), fill_width, progress_rect.height())
            
            progress_gradient = QLinearGradient(fill_rect.topLeft(), fill_rect.topRight())
            progress_gradient.setColorAt(0, QColor("#00D4FF"))
            progress_gradient.setColorAt(0.5, QColor("#00FF88"))
            progress_gradient.setColorAt(1, QColor("#FFD700"))
            
            painter.setBrush(progress_gradient)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(fill_rect, 12, 12)
        
        # 进度百分比
        painter.setPen(QColor("#FFFFFF"))
        percent_font = QFont("Consolas", 16, QFont.Bold)
        painter.setFont(percent_font)
        percent_rect = QRect(0, 490, self.width(), 30)
        painter.drawText(percent_rect, Qt.AlignCenter, f"{self.progress}%")


def main():
    """主函数"""
    # 创建应用程序
    app = QApplication(sys.argv)
    app.setApplicationName("BIRobot")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("BIRobot Team")
    app.setStyle("Fusion")
    
    # 设置应用程序图标（如果有的话）
    # app.setWindowIcon(QIcon("icon.png"))
    
    # 显示启动画面
    splash = SplashScreenOptimized()
    splash.show()
    
    # 提前初始化资源管理器
    resource_manager = ResourceManager()
    
    # 处理启动画面事件
    for _ in range(35):  # 约10秒的启动时间
        app.processEvents()
        time.sleep(0.15)
    
    # 创建并显示主窗口
    main_window = MainDashboard()
    splash.finish(main_window)
    
    # 启动应用程序事件循环
    return app.exec()


if __name__ == "__main__":
    # 更新当前时间和用户
    CURRENT_DATE = "2025-06-22 14:10:16"
    CURRENT_USER = "12ljf"
    
    # 运行应用程序
    sys.exit(main())