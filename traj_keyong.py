import sys
import os
import cv2
import numpy as np
import pandas as pd
import math
import tempfile
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QListWidget, QTableWidget, QTableWidgetItem,
                             QGroupBox, QFileDialog, QMessageBox, QComboBox, QSpinBox,
                             QDoubleSpinBox, QProgressBar, QCheckBox, QTextEdit, QSlider,
                             QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QTabWidget, QFrame, QSplitter)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint, QEventLoop, QSize
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QIcon, QFont


# ==================== 配置常量 ====================
# 跳帧处理设置：每SKIP_FRAMES帧处理1帧（例如：6表示每6帧处理1帧）

######################跟踪效果下降原因######################
#跳帧设置调高导致检测效果下降的根本原因是：跟踪算法需要连续帧间的检测信息来维持目标跟踪的连续性。
#当跳帧间隔过大时，目标在跳过的帧中可能发生显著变化，超出了跟踪算法的关联能力范围，从而导致跟踪失败和检测丢失。
#这是一个在处理速度和跟踪精度之间的权衡问题。建议根据你的具体应用场景，选择合适的跳帧值（通常 2-4 帧是比较好的平衡点）。

######################目标检测效果下降原因######################
#跳帧设置过高导致连目标都检测不到的根本原因是：YOLO跟踪算法的内部状态管理机制。
#当跳帧间隔过大时，跟踪器会"忘记"之前检测到的目标，或者认为它们已经"丢失"，从而不再在后续帧中检测这些目标。
#这不是检测算法本身的问题，而是跟踪算法的状态管理问题。
#为了获得最佳检测精度，现在设置为1，即处理每一帧。人工校正时可以在labelImg中跳帧。
SKIP_FRAMES = 1

# 可选导入
try:
    from pyproj import CRS, Transformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False

try:
    from yolo_processor import YOLOProcessor, YOLOProcessorThread
except Exception:
    # 开发时的基本存根
    class YOLOProcessor:
        frame_processed = pyqtSignal(object, object) if hasattr(pyqtSignal, '__call__') else None
        progress_updated = pyqtSignal(int, int, int) if hasattr(pyqtSignal, '__call__') else None
        fps_updated = pyqtSignal(float) if hasattr(pyqtSignal, '__call__') else None
        error_occurred = pyqtSignal(str) if hasattr(pyqtSignal, '__call__') else None
        processing_finished = pyqtSignal()
        detection_info_updated = pyqtSignal(str)
        model_loaded = pyqtSignal(str)

        def __init__(self): pass
        def load_model(self, path): return False
        def set_tracking_enabled(self, b): pass
        def set_target_fps(self, f): pass
        def stop_processing(self): pass
        def pause_processing(self): pass
        def resume_processing(self): pass
        def set_export_options(self, *args, **kwargs): pass

# ----------------- VideoLabel: 用于在 QLabel 显示可缩放/平移图像并映射到像素坐标 -------------
class VideoLabel(QLabel):
    clicked = pyqtSignal(int, int)  # 原始图像像素坐标（int,int）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundRole(self.backgroundRole())
        self.setSizePolicy(self.sizePolicy())
        self.setMinimumSize(200, 150)
        self._original_pixmap = None   # QPixmap 原始帧
        self._zoom = 1.0               # 缩放因子（相对于原始像素）
        self._offset = QPoint(0, 0)    # 平移偏移（像素）
        self._dragging = False
        self._last_mouse_pos = QPoint(0, 0)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #e0e0e0;")

    def set_frame(self, frame_bgr):
        """传入 OpenCV 的 BGR frame（numpy array），内部转换为 QPixmap 保存为原始图像"""
        if frame_bgr is None:
            self._original_pixmap = None
            self.update()
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self._original_pixmap = QPixmap.fromImage(qt_image)
        # 不自动 reset_view，保留用户当前缩放/偏移体验
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#e0e0e0'))
        if not self._original_pixmap:
            return

        # 计算用于绘制的缩放后的 pixmap
        target_w = int(self._original_pixmap.width() * self._zoom)
        target_h = int(self._original_pixmap.height() * self._zoom)
        if target_w <= 0 or target_h <= 0:
            return

        scaled_pix = self._original_pixmap.scaled(
            target_w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # 居中显示并加上偏移量
        x = (self.width() - scaled_pix.width()) // 2 + self._offset.x()
        y = (self.height() - scaled_pix.height()) // 2 + self._offset.y()
        painter.drawPixmap(x, y, scaled_pix)

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else (1.0 / 1.1)
        old_zoom = self._zoom
        new_zoom = max(0.1, min(4.0, old_zoom * factor))

        pos = event.position().toPoint()
        # 把鼠标在控件坐标映射到图像坐标（在旧缩放下）
        img_x_before, img_y_before = self._map_label_to_image_coords(pos)
        self._zoom = new_zoom
        img_x_after, img_y_after = self._map_label_to_image_coords(pos)

        # 调整偏移以尽可能保持鼠标位置不动（经验性调整）
        if img_x_before is not None and img_x_after is not None:
            self._offset -= QPoint(int((img_x_after - img_x_before) * self._zoom),
                                   int((img_y_after - img_y_before) * self._zoom))

        self._zoom = new_zoom
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            img_coords = self._map_label_to_image_coords(event.pos())
            if img_coords[0] is not None:
                self.clicked.emit(int(img_coords[0]), int(img_coords[1]))
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._dragging = True
            self._last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()
            self._offset += delta
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging = False

    def _map_label_to_image_coords(self, pos):
        """把 QLabel 内的坐标 pos(QPoint) 映射回原始图像坐标 (float x, float y)
           如果点击在图像外返回 (None, None)
        """
        if not self._original_pixmap:
            return (None, None)

        scaled_w = int(self._original_pixmap.width() * self._zoom)
        scaled_h = int(self._original_pixmap.height() * self._zoom)
        x0 = (self.width() - scaled_w) // 2 + self._offset.x()
        y0 = (self.height() - scaled_h) // 2 + self._offset.y()

        px = pos.x() - x0
        py = pos.y() - y0
        if 0 <= px < scaled_w and 0 <= py < scaled_h:
            img_x = px / self._zoom
            img_y = py / self._zoom
            return (img_x, img_y)
        else:
            return (None, None)

    def reset_view(self):
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self.update()

    def set_zoom(self, zoom_value):
        self._zoom = max(0.1, min(4.0, zoom_value))
        self.update()

    def get_zoom(self):
        return self._zoom

# 设置原点坐标的对话框
class OriginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置原点坐标")
        self.setModal(True)
        layout = QFormLayout(self)
        self.lon_input = QLineEdit()
        self.lat_input = QLineEdit()
        layout.addRow("经度:", self.lon_input)
        layout.addRow("纬度:", self.lat_input)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_coords(self):
        try:
            lon = float(self.lon_input.text())
            lat = float(self.lat_input.text())
            return lon, lat
        except ValueError:
            return None, None

# ----------------- 主窗口 -----------------
class VehicleTrajectoryAnalyzer(QMainWindow):
    """车辆轨迹与运动分析系统主窗口"""
    
    # 信号定义
    matrix_check_required = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._init_window()
        self._init_variables()
        self.init_ui()
        self.connect_signals()
        self.play_preview_btn.clicked.connect(self._on_preview_play)
        self.stop_preview_btn.clicked.connect(self._on_preview_stop)

        # 预览定时器
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self._preview_next_frame)
        
        # 校正视频播放定时器
        self.correction_timer = QTimer()
        self.correction_timer.timeout.connect(self._correction_next_frame)
        
        # 标志：是否正在播放（用于优化拖动进度条性能）
        self.is_playing_correction = False
        
        self.next_track_id = 100000
    
    def _init_window(self):
        """初始化窗口属性"""
        self.setWindowTitle("车辆轨迹与运动分析系统")
        self.setGeometry(100, 100, 1400, 800)
    
    def _init_variables(self):
        """初始化所有变量"""
        # 视频相关
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.video_fps = None
        
        # 坐标转换相关
        self.homography_matrix = None
        self.origin_lonlat = None
        self.origin_pixel = None
        self.utm_zone = None
        self.is_northern = True
        
        # 车辆跟踪相关
        self.vehicle_tracks = {}
        self.selected_vehicle_id = None
        self.pending_missing_ids = []
        
        # 模型相关
        self.model_folder = None
        
        # YOLO处理器
        self.yolo_processor = YOLOProcessor()
        # 连接处理器信号到UI日志显示（用于显示模型加载的详细状态和错误信息）
        self.yolo_processor.detection_info_updated.connect(self._on_detection_info)
        self.yolo_processor.error_occurred.connect(self._on_processor_error)
        self.yolo_thread = None
        
        # 处理状态
        self.last_frame_id = 0
        self.last_frame_time = 0.0
        self.last_detection_info = None
        self.last_processed_count = 0
        self.expected_processed_frames = 0
        self._paused = False
        
        # 校正后视频相关
        self.correction_folder = None
        self.correction_images = []
        self.correction_labels = []
        self.correction_timer = None
        self.correction_current_frame = 0
        self.correction_playing = False
        self.correction_fps = 25
        
        # 校正视频车辆跟踪相关
        self.correction_vehicle_tracks = {}
        self.correction_selected_vehicle_id = None
        
        # 轨迹显示设置
        self.trajectory_length = 100  # 显示最近100帧的轨迹
        
        # 校正数据坐标计算标志
        self.calculate_correction_coords = False
        
        # 当前模式：'original' 或 'correction'
        self.current_mode = 'original'
        
        # 车道检测相关
        self.lanes = {}  # 车道信息: {lane_id: {name, coord1, coord2, width, rect_corners, visible, vehicles_inside, flow_count, queue_count}}
        self.lane_counter = 0  # 统一车道计数器
        self.lane_detection_enabled = False  # 是否启用车道检测
        self.lane_rect_width = 30  # 默认矩形宽度
        self.vehicle_lane_path = {}  # 车辆路径记录: {vehicle_id: ['车道1', '车道3', ...]}
        self.waiting_for_coord = None  # 等待坐标输入的状态: (lane_id, coord_num)
        
        # 相机参数与畸变校正相关
        self.camera_matrix = None
        self.dist_coeffs = None
        self.undistort_enabled = False

    def init_ui(self):
        # --- 应用浅色主题样式 ---
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                color: #333333;
            }
            QWidget {
                background-color: #f5f5f5;
                color: #333333;
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                font-size: 10pt;
            }
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                left: 10px;
            }
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #b0b0b0;
                border-radius: 4px;
                padding: 5px 10px;
                color: #333333;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
                border-color: #909090;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
            QPushButton:disabled {
                background-color: #f0f0f0;
                color: #999999;
                border-color: #d0d0d0;
            }
            /* 特殊按钮颜色 */
            QPushButton#start_btn { background-color: #4caf50; border-color: #388e3c; color: #ffffff; }
            QPushButton#start_btn:hover { background-color: #66bb6a; }
            QPushButton#stop_btn { background-color: #f44336; border-color: #d32f2f; color: #ffffff; }
            QPushButton#stop_btn:hover { background-color: #ef5350; }
            
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 3px;
                color: #333333;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #2196f3;
            }
            QListWidget, QTableWidget {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                gridline-color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #e8e8e8;
                padding: 4px;
                border: 1px solid #cccccc;
                color: #333333;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 3px;
                text-align: center;
                background-color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #2196f3;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: #f5f5f5;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                border: 1px solid #cccccc;
                border-bottom-color: #cccccc;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                margin-right: 2px;
                color: #666666;
            }
            QTabBar::tab:selected {
                background-color: #f5f5f5;
                border-bottom-color: #f5f5f5;
                color: #333333;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #d8d8d8;
            }
            QSlider::groove:horizontal {
                border: 1px solid #cccccc;
                height: 6px;
                background: #e0e0e0;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2196f3;
                border: 1px solid #1976d2;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ================= 左侧面板 (视频与控制) =================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # 1. 视频显示区域
        video_container = QWidget()
        video_container.setStyleSheet("background-color: #e0e0e0; border: 1px solid #cccccc;")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = VideoLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        from PyQt6.QtWidgets import QSizePolicy
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self.video_label)
        
        left_layout.addWidget(video_container, 1) # Stretch factor 1

        # 2. 进度条与缩放控制
        progress_zoom_layout = QHBoxLayout()
        
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(0, 0)
        self.zoom_slider.setValue(0)
        self.zoom_slider.setEnabled(False)
        progress_zoom_layout.addWidget(self.zoom_slider)
        
        # 缩放按钮组
        self.zoom_out_btn = QPushButton("选择校正文件夹") # 功能保持不变，仅位置调整
        self.zoom_in_btn = QPushButton("播放校正视频")
        self.zoom_in_btn.setEnabled(False)
        
        self.calculate_coords_btn = QPushButton("计算坐标")
        self.calculate_coords_btn.setEnabled(False)
        self.calculate_coords_btn.setToolTip("根据单应矩阵计算校正数据的真实地理坐标")
        
        # 新增：轨迹帧数标签
        trajectory_label = QLabel("轨迹显示帧数:")
        self.trajectory_length_edit = QLineEdit()
        self.trajectory_length_edit.setPlaceholderText("帧数")
        self.trajectory_length_edit.setText("100")
        self.trajectory_length_edit.setMaximumWidth(60)
        self.trajectory_length_edit.setEnabled(False)
        self.trajectory_length_edit.returnPressed.connect(self.on_trajectory_length_changed)
        
        # 将这些按钮放入工具栏
        
        left_layout.addLayout(progress_zoom_layout)

        # 3. 综合控制面板 (分组显示)
        controls_frame = QFrame()
        controls_frame.setStyleSheet("QFrame { background-color: #e8e8e8; border-radius: 6px; }")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        
        # 组1: 资源加载
        load_group = QVBoxLayout()
        self.load_video_btn = QPushButton("加载视频")
        self.load_model_btn = QPushButton("选择模型文件夹")
        self.load_model_btn.setEnabled(False)
        
        model_select_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEnabled(False)
        self.model_combo.setMinimumWidth(120)
        self.refresh_models_btn = QPushButton("刷新")
        self.refresh_models_btn.setMaximumWidth(50)
        self.refresh_models_btn.setEnabled(False)
        model_select_layout.addWidget(self.model_combo)
        model_select_layout.addWidget(self.refresh_models_btn)
        
        self.load_selected_model_btn = QPushButton("加载模型")
        self.load_selected_model_btn.setEnabled(False)
        
        load_group.addWidget(self.load_video_btn)
        load_group.addWidget(self.load_model_btn)
        load_group.addLayout(model_select_layout)
        load_group.addWidget(self.load_selected_model_btn)
        controls_layout.addLayout(load_group)
        
        # 分隔线
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.VLine)
        line1.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(line1)
        
        # 组2: 处理控制
        process_group = QVBoxLayout()
        self.start_btn = QPushButton("开始处理")
        self.start_btn.setObjectName("start_btn") # For custom styling
        self.start_btn.setEnabled(False)
        
        pause_stop_layout = QHBoxLayout()
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setEnabled(False)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        pause_stop_layout.addWidget(self.pause_btn)
        pause_stop_layout.addWidget(self.stop_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        
        process_group.addWidget(self.start_btn)
        process_group.addLayout(pause_stop_layout)
        process_group.addWidget(self.progress_bar)
        controls_layout.addLayout(process_group)
        
        # 分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.VLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(line2)
        
        # 组3: 预览与回放工具
        tools_group = QVBoxLayout()
        preview_layout = QHBoxLayout()
        self.play_preview_btn = QPushButton("播放预览")
        self.play_preview_btn.setEnabled(False)
        self.stop_preview_btn = QPushButton("停止预览")
        self.stop_preview_btn.setEnabled(False)
        preview_layout.addWidget(self.play_preview_btn)
        preview_layout.addWidget(self.stop_preview_btn)
        
        correction_tools_layout = QHBoxLayout()
        correction_tools_layout.addWidget(self.zoom_out_btn) # 选择校正文件夹
        correction_tools_layout.addWidget(self.zoom_in_btn)  # 播放校正视频
        
        calc_layout = QHBoxLayout()
        calc_layout.addWidget(self.calculate_coords_btn)
        calc_layout.addWidget(trajectory_label) # 添加标签
        calc_layout.addWidget(self.trajectory_length_edit)
        
        tools_group.addLayout(preview_layout)
        tools_group.addLayout(correction_tools_layout)
        tools_group.addLayout(calc_layout)
        controls_layout.addLayout(tools_group)

        left_layout.addWidget(controls_frame)

        # 4. 状态日志 (底部)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.status_text.setPlaceholderText("系统日志...")
        left_layout.addWidget(self.status_text)

        # ================= 右侧面板 (标签页) =================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        
        # --- Tab 1: 车辆信息 ---
        tab_vehicle = QWidget()
        tab_vehicle_layout = QVBoxLayout(tab_vehicle)
        
        vehicle_group = QGroupBox("车辆列表")
        vehicle_layout = QVBoxLayout(vehicle_group)
        self.vehicle_list = QListWidget()
        vehicle_layout.addWidget(self.vehicle_list)
        tab_vehicle_layout.addWidget(vehicle_group, 1) # Stretch
        
        info_group = QGroupBox("详细属性")
        info_layout = QVBoxLayout(info_group)
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(6) # 恢复为6列
        self.info_table.setHorizontalHeaderLabels(["属性", "值", "属性", "值", "属性", "值"])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setAlternatingRowColors(True)
        info_layout.addWidget(self.info_table)
        tab_vehicle_layout.addWidget(info_group, 1) # Stretch
        
        self.tabs.addTab(tab_vehicle, "车辆信息")
        
        # --- Tab 2: 参数设置 ---
        tab_settings = QWidget()
        tab_settings_layout = QVBoxLayout(tab_settings)
        
        homography_group = QGroupBox("单应矩阵设置")
        homography_layout = QVBoxLayout(homography_group)

        self.homography_input = QTextEdit()
        self.homography_input.setPlaceholderText("粘贴单应矩阵内容（9个数值，空格或换行分隔）")
        self.homography_input.setMaximumHeight(80)
        homography_layout.addWidget(self.homography_input)

        load_h_btn_layout = QHBoxLayout()
        self.load_h_file_btn = QPushButton("从文件加载")
        self.apply_h_btn = QPushButton("应用矩阵")
        load_h_btn_layout.addWidget(self.load_h_file_btn)
        load_h_btn_layout.addWidget(self.apply_h_btn)
        homography_layout.addLayout(load_h_btn_layout)

        origin_layout = QFormLayout()
        self.origin_lon_input = QDoubleSpinBox()
        self.origin_lon_input.setRange(-180, 180)
        self.origin_lon_input.setDecimals(8)
        self.origin_lat_input = QDoubleSpinBox()
        self.origin_lat_input.setRange(-90, 90)
        self.origin_lat_input.setDecimals(8)
        origin_layout.addRow("原点经度:", self.origin_lon_input)
        origin_layout.addRow("原点纬度:", self.origin_lat_input)
        homography_layout.addLayout(origin_layout)

        utm_layout = QHBoxLayout()
        utm_layout.addWidget(QLabel("UTM区域:"))
        self.utm_zone_input = QSpinBox()
        self.utm_zone_input.setRange(1, 60)
        utm_layout.addWidget(self.utm_zone_input)
        utm_layout.addWidget(QLabel("半球:"))
        self.hemisphere_combo = QComboBox()
        self.hemisphere_combo.addItems(["北半球", "南半球"])
        utm_layout.addWidget(self.hemisphere_combo)
        homography_layout.addLayout(utm_layout)
        
        tab_settings_layout.addWidget(homography_group)
        
        # 相机参数设置组（畸变校正）
        camera_group = QGroupBox("相机参数（畸变校正）")
        camera_layout = QVBoxLayout(camera_group)
        
        camera_file_layout = QHBoxLayout()
        self.camera_file_le = QLineEdit()
        self.camera_file_le.setPlaceholderText("相机参数文件 (JSON/YAML)")
        self.camera_file_le.setReadOnly(True)
        self.load_camera_btn = QPushButton("选择文件")
        camera_file_layout.addWidget(self.camera_file_le)
        camera_file_layout.addWidget(self.load_camera_btn)
        camera_layout.addLayout(camera_file_layout)
        
        self.camera_status_label = QLabel("未加载相机参数")
        self.camera_status_label.setStyleSheet("color: #666666;")
        camera_layout.addWidget(self.camera_status_label)
        
        self.undistort_cb = QCheckBox("启用畸变校正（需先加载相机参数和勾选此项）")
        self.undistort_cb.setChecked(False)
        self.undistort_cb.setEnabled(False)
        self.undistort_cb.stateChanged.connect(self.on_undistort_toggle)
        camera_layout.addWidget(self.undistort_cb)
        
        tab_settings_layout.addWidget(camera_group)
        
        save_group = QGroupBox("保存设置")
        save_layout = QVBoxLayout(save_group)
        self.save_frame_annotations_cb = QCheckBox("保存帧与标注到本地")
        self.save_frame_annotations_cb.setChecked(True)
        save_layout.addWidget(self.save_frame_annotations_cb)
        
        path_layout = QHBoxLayout()
        self.save_folder_le = QLineEdit()
        self.save_folder_le.setPlaceholderText("标注保存目录")
        self.save_folder_le.setText("results")
        self.choose_save_folder_btn = QPushButton("...")
        self.choose_save_folder_btn.setMaximumWidth(40)
        path_layout.addWidget(self.save_folder_le)
        path_layout.addWidget(self.choose_save_folder_btn)
        save_layout.addLayout(path_layout)
        
        tab_settings_layout.addWidget(save_group)
        tab_settings_layout.addStretch()
        
        self.tabs.addTab(tab_settings, "参数设置")
        
        # --- Tab 3: 车道配置 ---
        # 直接使用现有的 create_lane_config_panel 方法，但需要调整其父级
        # 注意：create_lane_config_panel 返回一个 QGroupBox
        self.lane_config_panel = self.create_lane_config_panel()
        # 为了放入 tab，我们需要一个容器 widget
        tab_lane = QWidget()
        tab_lane_layout = QVBoxLayout(tab_lane)
        tab_lane_layout.addWidget(self.lane_config_panel)
        self.tabs.addTab(tab_lane, "车道配置")
        
        # --- Tab 4: 数据管理 ---
        tab_data = QWidget()
        tab_data_layout = QVBoxLayout(tab_data)
        
        data_group = QGroupBox("数据操作")
        data_layout = QVBoxLayout(data_group)
        
        self.export_btn = QPushButton("导出数据 (CSV)")
        self.export_btn.setEnabled(False)
        self.clear_btn = QPushButton("清除所有数据")
        
        data_layout.addWidget(self.export_btn)
        data_layout.addWidget(self.clear_btn)
        tab_data_layout.addWidget(data_group)
        
        tools_group_box = QGroupBox("辅助工具")
        tools_layout = QVBoxLayout(tools_group_box)
        
        self.request_manual_btn = QPushButton("手动修正 (LabelImg)")
        self.request_manual_btn.setEnabled(True)
        
        self.open_geo_mapper_btn = QPushButton("GCP工具 (计算单应矩阵)")
        self.open_geo_mapper_btn.setEnabled(True)
        
        tools_layout.addWidget(self.request_manual_btn)
        tools_layout.addWidget(self.open_geo_mapper_btn)
        tab_data_layout.addWidget(tools_group_box)
        
        tab_data_layout.addStretch()
        self.tabs.addTab(tab_data, "数据管理")

        right_layout.addWidget(self.tabs)

        # 添加到主布局
        # 使用 Splitter 允许调整左右比例
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)

    def create_lane_config_panel(self):
        """创建车道配置面板"""
        panel = QGroupBox("车道检测配置")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(5)
        
        # 车道列表容器（可滚动）
        from PyQt6.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.lanes_container = QWidget()
        self.lanes_layout = QVBoxLayout(self.lanes_container)
        self.lanes_layout.setSpacing(8)
        self.lanes_layout.setContentsMargins(5, 5, 5, 5)
        self.lanes_layout.addStretch()
        
        scroll_area.setWidget(self.lanes_container)
        panel_layout.addWidget(scroll_area)
        
        # 底部按钮
        bottom_layout = QHBoxLayout()
        self.add_lane_btn = QPushButton("+ 添加车道")
        self.add_lane_btn.clicked.connect(self.create_lane_widget)
        bottom_layout.addWidget(self.add_lane_btn)
        
        self.lane_detection_cb = QCheckBox("启用检测")
        self.lane_detection_cb.setChecked(False)
        self.lane_detection_cb.stateChanged.connect(self.on_lane_detection_toggle)
        bottom_layout.addWidget(self.lane_detection_cb)
        
        panel_layout.addLayout(bottom_layout)
        
        return panel

    def connect_signals(self):
        """连接所有信号和槽"""
        # 视频和模型相关
        self.load_video_btn.clicked.connect(self.load_video)
        self.load_model_btn.clicked.connect(self.browse_model_folder)
        self.refresh_models_btn.clicked.connect(self.refresh_model_list)
        self.load_selected_model_btn.clicked.connect(self.load_selected_model)
        
        # 处理控制相关
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.pause_btn.clicked.connect(self.toggle_pause)
        
        # 坐标转换相关
        self.load_h_file_btn.clicked.connect(self.load_homography_from_file)
        self.apply_h_btn.clicked.connect(self.apply_homography)
        
        # 相机参数相关
        self.load_camera_btn.clicked.connect(self.load_camera_params)
        
        # 数据管理相关
        self.export_btn.clicked.connect(self.export_data)
        self.clear_btn.clicked.connect(self.clear_data)
        
        # 视图控制相关
        self.zoom_in_btn.clicked.connect(self.play_correction_video)
        self.zoom_out_btn.clicked.connect(self.select_correction_folder)
        self.calculate_coords_btn.clicked.connect(self.on_calculate_correction_coords)
        self.zoom_slider.valueChanged.connect(self.on_correction_slider_changed)
        
        # 车辆选择相关
        self.vehicle_list.itemSelectionChanged.connect(self.on_vehicle_selected)
        
        # 视频交互相关
        self.video_label.clicked.connect(self.on_video_label_clicked)
        
        # 矩阵检查信号
        self.matrix_check_required.connect(self.check_matrix_and_continue)
        
        # 可选功能连接
        self._connect_optional_signals()
    
    def _connect_optional_signals(self):
        """连接可选功能的信号"""
        try:
            self.choose_save_folder_btn.clicked.connect(self.on_choose_save_folder)
            self.request_manual_btn.clicked.connect(self.on_request_manual_correction)
            self.open_geo_mapper_btn.clicked.connect(self.on_open_geo_mapper)
            self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        except Exception:
            pass

    def _on_model_combo_changed(self, idx):
        try:
            txt = self.model_combo.currentText()
            ok = bool(txt) and not txt.startswith("（该文件夹中未发现")
            self.load_selected_model_btn.setEnabled(ok)
        except Exception:
            pass

    def _on_detection_info(self, info_text):
        """处理YOLO处理器发出的检测信息，显示在日志区域"""
        try:
            self.status_text.append(info_text)
        except Exception:
            pass

    def _on_processor_error(self, error_text):
        """处理YOLO处理器发出的错误信息，显示在日志区域"""
        try:
            self.status_text.append(f"❌ {error_text}")
        except Exception:
            pass

    # ---------- 相机参数与畸变校正相关 ----------
    def load_camera_params(self):
        """加载相机参数文件 (JSON 或 YAML)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择相机参数文件", "",
            "Camera Files (*.json *.yaml *.yml);;All Files (*)"
        )
        if not file_path:
            return
        
        try:
            import json
            import yaml
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.lower().endswith('.json'):
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)
            
            # 提取相机矩阵
            camera_matrix = None
            if 'camera_matrix' in data:
                camera_matrix = np.array(data['camera_matrix'], dtype=np.float64)
            elif 'K' in data:
                camera_matrix = np.array(data['K'], dtype=np.float64)
            
            # 提取畸变系数
            dist_coeffs = None
            if 'dist_coeffs' in data:
                dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float64)
            elif 'dist' in data:
                dist_coeffs = np.array(data['dist'], dtype=np.float64)
            
            if camera_matrix is None or dist_coeffs is None:
                QMessageBox.warning(self, "错误", "相机参数文件中缺少 camera_matrix 或 dist_coeffs")
                return
            
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            
            # 更新UI
            self.camera_file_le.setText(os.path.basename(file_path))
            self.camera_status_label.setText(f"✅ 已加载: K={camera_matrix.shape}, dist={dist_coeffs.shape}")
            self.camera_status_label.setStyleSheet("color: #2e7d32;")
            self.undistort_cb.setEnabled(True)
            
            self.status_text.append(f"相机参数已加载: {file_path}")
            
        except Exception as e:
            self.status_text.append(f"加载相机参数失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"加载相机参数失败: {str(e)}")
    
    def on_undistort_toggle(self, state):
        """畸变校正开关状态变化"""
        self.undistort_enabled = (state == Qt.CheckState.Checked.value)
        if self.undistort_enabled:
            self.status_text.append("畸变校正已启用，坐标转换时将先对像素坐标进行校正")
        else:
            self.status_text.append("畸变校正已禁用")

    # ---------- 校正视频相关 ----------
    def select_correction_folder(self):
        """选择包含images和labels文件夹的校正目录"""
        # 检查是否有正在运行的视频处理，如果有则自动停止
        if hasattr(self, 'yolo_thread') and self.yolo_thread and self.yolo_thread.isRunning():
            self.status_text.append("检测到正在运行的视频处理，自动停止处理...")
            try:
                if hasattr(self, 'yolo_processor') and self.yolo_processor:
                    self.yolo_processor.stop_processing()
            except Exception:
                pass
            try:
                self.yolo_thread.quit()
                self.yolo_thread.wait()
            except Exception:
                pass
            # 重置暂停状态
            self._paused = False
            self.pause_btn.setText("暂停")
            self.status_text.append("视频处理已停止")
        
        folder = QFileDialog.getExistingDirectory(self, "选择校正文件夹", "")
        if not folder:
            return
        
        images_dir = os.path.join(folder, "images")
        labels_dir = os.path.join(folder, "labels")
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            QMessageBox.warning(self, "错误", "所选文件夹必须包含images和labels子文件夹")
            return
        
        self.correction_folder = folder
        
        # 清空上一个校正视频的数据
        self.correction_vehicle_tracks.clear()
        
        # 重置坐标计算标志（新加载的校正文件夹默认不计算坐标）
        self.calculate_correction_coords = False
        self.calculate_coords_btn.setEnabled(True)  # 启用计算坐标按钮
        
        self.load_correction_files()
        self.status_text.append(f"已加载校正文件夹: {folder}")
        self.status_text.append("提示: 默认不计算真实坐标，如需计算请设置单应矩阵并点击'计算坐标'按钮")
        
        # 切换到校正模式
        self.current_mode = 'correction'
        self.update_vehicle_list()  # 刷新车辆列表，只显示校正视频的车辆
        
        # 自动显示第一帧（用于设置车道坐标）
        if self.correction_images:
            self.correction_current_frame = 0
            self.display_correction_frame(0)
            self.status_text.append("已显示第一帧，可以开始设置车道坐标")
        
        # 按下zoom_out_btn后的按钮状态切换
        self.load_model_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.refresh_models_btn.setEnabled(False)
        self.load_selected_model_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.play_preview_btn.setEnabled(False)
        self.stop_preview_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        
        self.zoom_in_btn.setEnabled(True)
        self.calculate_coords_btn.setEnabled(True)  # 启用计算坐标按钮
        self.trajectory_length_edit.setEnabled(True)
        self.zoom_slider.setEnabled(True)
    
    def load_correction_files(self):
        """加载校正文件夹中的图片和标签文件"""
        if not self.correction_folder:
            return
        
        images_dir = os.path.join(self.correction_folder, "images")
        labels_dir = os.path.join(self.correction_folder, "labels")
        
        # 获取所有图片文件
        image_files = []
        for f in os.listdir(images_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(f)
        
        # 按文件名排序
        image_files.sort()
        
        self.correction_images = []
        self.correction_labels = []
        
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            # 对应的标签文件
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            self.correction_images.append(img_path)
            self.correction_labels.append(label_path if os.path.exists(label_path) else None)
        
        # 更新进度条范围
        if self.correction_images:
            self.zoom_slider.setRange(0, len(self.correction_images) - 1)
            self.zoom_slider.setValue(0)
            self.status_text.append(f"加载了 {len(self.correction_images)} 个校正帧")
        else:
            self.status_text.append("校正文件夹中没有找到图片文件")
    
    def play_correction_video(self):
        """播放校正后的视频"""
        if not self.correction_images:
            self.status_text.append("请先选择校正文件夹")
            return
        
        if self.correction_playing:
            # 停止播放
            self.correction_timer.stop()
            self.correction_playing = False
            self.is_playing_correction = False
            self.zoom_in_btn.setText("播放校正视频")
            self.status_text.append("停止播放校正视频")
        else:
            # 开始播放
            self.correction_playing = True
            self.is_playing_correction = True
            self.zoom_in_btn.setText("停止播放")
            
            # 设置定时器间隔
            interval = max(10, int(1000.0 / self.correction_fps))
            self.correction_timer.setInterval(interval)
            self.correction_timer.start()
            
            self.status_text.append("开始播放校正视频")
    
    def _correction_next_frame(self):
        """校正视频播放的下一帧"""
        if not self.correction_images or self.correction_current_frame >= len(self.correction_images):
            # 播放结束
            self.correction_timer.stop()
            self.correction_playing = False
            self.is_playing_correction = False
            self.zoom_in_btn.setText("播放校正视频")
            self.correction_current_frame = 0
            self.zoom_slider.setValue(0)
            self.status_text.append("校正视频播放结束")
            return
        
        # 显示当前帧
        self.display_correction_frame(self.correction_current_frame)
        self.correction_current_frame += 1
        
        # 更新进度条
        if len(self.correction_images) > 0:
            # 直接使用当前帧索引作为进度条值，而不是百分比计算
            self.zoom_slider.setValue(min(self.correction_current_frame, len(self.correction_images) - 1))
    
    def display_correction_frame(self, frame_index):
        """显示校正后的帧（包含检测框）"""
        if frame_index >= len(self.correction_images):
            return
        
        try:
            # 读取图片
            img_path = self.correction_images[frame_index]
            frame = cv2.imread(img_path)
            if frame is None:
                self.status_text.append(f"无法读取图片: {img_path}")
                return
            
            # 读取标签并绘制检测框
            label_path = self.correction_labels[frame_index]
            if label_path and os.path.exists(label_path):
                frame = self.draw_correction_labels(frame, label_path)
            
            # 显示图片
            self.video_label.set_frame(frame)
            self.current_frame = frame
            
        except Exception as e:
            self.status_text.append(f"显示校正帧失败: {e}")
    
    def draw_correction_labels(self, frame, label_path):
        """在帧上绘制校正后的标签并更新车辆跟踪数据"""
        try:
            h, w = frame.shape[:2]
            frame_id = self.correction_current_frame
            frame_time = frame_id / self.correction_fps if self.correction_fps > 0 else frame_id
            
            # 存储当前帧的检测信息
            current_detection_info = {
                'frame_id': frame_id,
                'frame_time': frame_time,
                'objects': []
            }
            
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                try:
                    class_id = int(float(parts[0]))
                    confidence = float(parts[-2]) if len(parts) >= 6 else 1.0
                    track_id = int(float(parts[-1])) if len(parts) >= 7 else -1
                    
                    # 创建对象信息
                    obj_info = {
                        'track_id': track_id if track_id >= 0 else None,
                        'class_id': class_id,
                        'class_name': f'Class_{class_id}',
                        'confidence': confidence,
                        'bbox_type': 'obb' if len(parts) >= 9 else 'rect',
                        'bbox': []
                    }
                    
                    if len(parts) >= 9:  # OBB格式
                        # 解析8个点的坐标
                        coords = [float(x) for x in parts[1:9]]
                        points = np.array(coords).reshape(-1, 2)
                        
                        # 归一化坐标转像素坐标
                        points[:, 0] *= w
                        points[:, 1] *= h
                        points = points.astype(int)
                        
                        # 计算中心点
                        center_x = np.mean(points[:, 0])
                        center_y = np.mean(points[:, 1])
                        center_px = (float(center_x), float(center_y))
                        
                        # 存储bbox信息（像素坐标）
                        obj_info['bbox'] = points.flatten().tolist()
                        obj_info['center_px'] = center_px
                        obj_info['corners_px'] = [(float(p[0]), float(p[1])) for p in points]
                        
                        # 绘制旋转边界框
                        color = self.get_correction_color_for_class(class_id)
                        cv2.polylines(frame, [points], True, color, 2)
                        
                        # 绘制标签
                        x1, y1 = points.min(axis=0)
                        label = f"ID:{track_id} Class:{class_id} {confidence:.2f}" if track_id >= 0 else f"Class:{class_id} {confidence:.2f}"
                        self.draw_label(frame, x1, y1, label, color)
                    
                    else:  # 普通边界框格式
                        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                        
                        # 转换为像素坐标
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        
                        # 计算中心点
                        center_px = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        
                        # 存储bbox信息
                        obj_info['bbox'] = [x1, y1, x2, y2]
                        obj_info['center_px'] = center_px
                        obj_info['corners_px'] = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        
                        # 绘制矩形
                        color = self.get_correction_color_for_class(class_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # 绘制标签
                        label = f"ID:{track_id} Class:{class_id} {confidence:.2f}" if track_id >= 0 else f"Class:{class_id} {confidence:.2f}"
                        self.draw_label(frame, x1, y1, label, color)
                    
                    current_detection_info['objects'].append(obj_info)
                
                except (ValueError, IndexError) as e:
                    continue
            
            # 只在播放时更新车辆跟踪数据和车道检测（优化拖动进度条性能）
            if self.is_playing_correction:
                # 更新校正视频的车辆跟踪数据
                self.update_correction_vehicle_tracks(current_detection_info)
                
                # 车道检测（在更新跟踪数据后）
                for obj_info in current_detection_info['objects']:
                    vehicle_id = obj_info.get('track_id')
                    if vehicle_id is not None and vehicle_id >= 0:
                        corners = obj_info.get('corners_px', [])
                        if corners:
                            center_px = obj_info.get('center_px')
                            if center_px is None:
                                pts_array = np.array(corners)
                                center_px = (np.mean(pts_array[:, 0]), np.mean(pts_array[:, 1]))
                            
                            self.update_lane_statistics(vehicle_id, center_px)
            
            # 总是绘制车辆轨迹（使用已有的跟踪数据）
            self.draw_correction_trajectories(frame)
            
            # 总是绘制车道（在所有其他内容之后）
            frame = self.draw_lanes_on_frame(frame)
            
            # 添加调试信息（可选）
            if frame_id % 30 == 0:  # 每30帧显示一次调试信息
                track_count = len(self.correction_vehicle_tracks)
                if track_count > 0:
                    self.status_text.append(f"帧 {frame_id}: 显示 {track_count} 辆车的轨迹")
            
        except Exception as e:
            self.status_text.append(f"绘制校正标签失败: {e}")
        
        return frame
    
    def get_correction_color_for_class(self, class_id):
        """为不同类别生成不同颜色（校正视频用）"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (0, 128, 0)
        ]
        return colors[class_id % len(colors)]
    
    def draw_correction_trajectories(self, frame):
        """绘制校正视频中车辆的轨迹（最近N帧）- 优化版本"""
        try:
            # 如果轨迹长度为0，不绘制任何轨迹
            if self.trajectory_length <= 0:
                return
                
            if not self.correction_vehicle_tracks:
                return
            
            # 获取当前帧ID
            current_frame = self.correction_current_frame
            
            # 定义车辆消失阈值（帧数）：如果车辆最后出现的帧距离当前帧超过此值，则不显示轨迹
            disappear_threshold = 10  # 可以根据需要调整
            
            for vehicle_id, track in self.correction_vehicle_tracks.items():
                if not track.get('bbox_centers_px'):
                    continue
                
                # 获取车辆最后出现的帧
                last_frame = track.get('last_frame', None)
                if last_frame is None or (current_frame - last_frame) > disappear_threshold:
                    # 车辆已经消失太久，不显示轨迹
                    continue
                
                # 根据车辆ID生成颜色（确保不同车辆有不同颜色）
                color = self.get_correction_color_for_class(vehicle_id % 10)
                
                # 获取所有中心点和对应帧ID
                centers = track['bbox_centers_px']
                frame_ids = track.get('frame_ids', [])
                
                # 构建有效的轨迹点列表（带帧ID信息）
                valid_trajectory = []  # 每个元素是 (center, frame_id)
                for i, center in enumerate(centers):
                    if i < len(frame_ids) and frame_ids[i] is not None:
                        fid = frame_ids[i]
                        # 只包含当前帧及之前的点
                        if fid <= current_frame:
                            valid_trajectory.append((center, fid))
                    else:
                        # 如果没有frame_ids信息，假设是按顺序的
                        if i <= current_frame:
                            valid_trajectory.append((center, i))
                
                # 只保留最近N帧的轨迹
                if len(valid_trajectory) > self.trajectory_length:
                    valid_trajectory = valid_trajectory[-self.trajectory_length:]
                
                if len(valid_trajectory) == 0:
                    continue
                
                # 绘制轨迹线（只连接连续或接近的帧）
                max_frame_gap = 3  # 允许的最大帧间隔，超过此值不连线
                for i in range(1, len(valid_trajectory)):
                    prev_center, prev_fid = valid_trajectory[i-1]
                    curr_center, curr_fid = valid_trajectory[i]
                    
                    # 检查帧ID的连续性，避免跳帧导致的错误连线
                    frame_gap = curr_fid - prev_fid
                    if frame_gap <= max_frame_gap and frame_gap > 0:
                        # 计算线条粗细：越新的线越粗
                        alpha_ratio = i / len(valid_trajectory)
                        line_thickness = max(1, int(2 * alpha_ratio) + 1)
                        
                        # 绘制轨迹线
                        cv2.line(frame, 
                                (int(prev_center[0]), int(prev_center[1])),
                                (int(curr_center[0]), int(curr_center[1])), 
                                color, line_thickness)
                
                # 绘制轨迹点（小圆点）- 调整大小使其更合理
                for i, (center, fid) in enumerate(valid_trajectory):
                    # 计算点的大小：从1到3渐变
                    progress = i / max(1, len(valid_trajectory) - 1)
                    point_size = max(1, int(1 + progress * 2))  # 范围：1-3
                    cv2.circle(frame, (int(center[0]), int(center[1])), point_size, color, -1)
                
                # 绘制当前中心点（稍大但不过分）
                if valid_trajectory and last_frame == current_frame:
                    # 只有车辆在当前帧中才绘制当前中心点
                    current_center, _ = valid_trajectory[-1]
                    # 绘制外圈（白色边框）- 缩小尺寸
                    cv2.circle(frame, (int(current_center[0]), int(current_center[1])), 5, (255, 255, 255), 2)
                    # 绘制内圈（车辆颜色）
                    cv2.circle(frame, (int(current_center[0]), int(current_center[1])), 3, color, -1)
                    
        except Exception as e:
            self.status_text.append(f"绘制轨迹失败: {e}")
    
    def show_trajectory_stats(self):
        """显示轨迹数据统计信息"""
        try:
            if not self.correction_vehicle_tracks:
                self.status_text.append("当前没有轨迹数据")
                return
            
            total_vehicles = len(self.correction_vehicle_tracks)
            current_frame = self.correction_current_frame
            
            stats = []
            for vehicle_id, track in self.correction_vehicle_tracks.items():
                total_points = len(track.get('bbox_centers_px', []))
                frame_ids = track.get('frame_ids', [])
                
                # 计算当前帧及之前的轨迹点数
                valid_points = 0
                for i, frame_id in enumerate(frame_ids):
                    if frame_id is not None and frame_id <= current_frame:
                        valid_points += 1
                
                stats.append(f"车辆{vehicle_id}: 总点数{total_points}, 当前有效点数{valid_points}")
            
            self.status_text.append(f"轨迹统计 (当前帧{current_frame}, 显示长度{self.trajectory_length}):")
            for stat in stats[:5]:  # 只显示前5个车辆的信息
                self.status_text.append(f"  {stat}")
            if len(stats) > 5:
                self.status_text.append(f"  ... 还有{len(stats)-5}个车辆")
                
        except Exception as e:
            self.status_text.append(f"显示轨迹统计失败: {e}")
    
    def draw_label(self, frame, x1, y1, label, color):
        """绘制标签"""
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def on_correction_slider_changed(self, val):
        """校正视频进度条变化"""
        if not self.correction_images:
            return
        
        frame_index = min(val, len(self.correction_images) - 1)
        self.correction_current_frame = frame_index
        self.display_correction_frame(frame_index)
    
    def on_trajectory_length_changed(self):
        """轨迹长度设置变化"""
        try:
            value = int(self.trajectory_length_edit.text())
            if 0 <= value <= 500:
                self.trajectory_length = value
                if value == 0:
                    self.status_text.append("轨迹显示已关闭")
                else:
                    self.status_text.append(f"轨迹长度设置为: {value} 帧")
                
                # 立即刷新当前显示的帧，让新的轨迹长度设置生效
                if hasattr(self, 'correction_images') and self.correction_images:
                    self.display_correction_frame(self.correction_current_frame)
                    
                # 显示轨迹数据统计信息
                self.show_trajectory_stats()
            else:
                self.status_text.append("轨迹长度必须在0-500帧之间")
                self.trajectory_length_edit.setText(str(self.trajectory_length))
        except ValueError:
            self.status_text.append("请输入有效的数字")
            self.trajectory_length_edit.setText(str(self.trajectory_length))
    
    def on_calculate_correction_coords(self):
        """计算校正数据的真实地理坐标"""
        # 检查是否已设置单应矩阵和原点坐标
        missing_items = []
        if self.homography_matrix is None:
            missing_items.append("单应矩阵")
        if self.origin_lonlat is None:
            missing_items.append("原点坐标")
        
        if missing_items:
            missing_str = "、".join(missing_items)
            reply = QMessageBox.question(
                self,
                "坐标设置未完成",
                f"未设置{missing_str}，将无法计算真实地理坐标。\n\n需要设置：\n• 单应矩阵（9个数值）\n• 原点经纬度坐标\n• UTM区域和半球\n• 点击'应用矩阵'按钮\n\n选择：\n• 是：继续（但无法计算真实坐标）\n• 否：取消，先设置坐标系统",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                self.status_text.append(f"已取消：请先设置{missing_str}")
                return
            else:
                self.status_text.append(f"警告: 未设置{missing_str}，将无法计算真实地理坐标")
        
        # 设置标志位，启用坐标计算
        self.calculate_correction_coords = True
        self.status_text.append("已启用校正数据真实坐标计算")
        
        # 显示当前设置（如果有的话）
        if self.homography_matrix is not None:
            self.status_text.append(f"使用单应矩阵: {self.homography_matrix[0,0]:.6f}...")
        else:
            self.status_text.append("警告: 单应矩阵未设置")
        
        if self.origin_lonlat is not None:
            self.status_text.append(f"原点坐标: {self.origin_lonlat}")
        else:
            self.status_text.append("警告: 原点坐标未设置")
        
        # 清空之前的校正车辆跟踪数据，重新计算
        self.correction_vehicle_tracks.clear()
        
        # 如果当前有显示的帧，重新显示以触发坐标计算
        if hasattr(self, 'correction_images') and self.correction_images:
            self.display_correction_frame(self.correction_current_frame)
            self.status_text.append("开始计算真实坐标，请播放或拖动进度条查看结果")
        
        # 更新车辆列表
        self.update_vehicle_list()
    
    def update_correction_vehicle_tracks(self, detection_info):
        """更新校正视频的车辆跟踪数据"""
        frame_id = detection_info.get('frame_id', None)
        frame_time = detection_info.get('frame_time', None)
        
        current_ids = set()
        for obj in detection_info.get('objects', []):
            vehicle_id = obj.get('track_id', None)
            if vehicle_id is None:
                continue
            current_ids.add(vehicle_id)
            
            # 获取中心点坐标
            center_px = obj.get('center_px', (0, 0))
            corners_px = obj.get('corners_px', [])
            
            # 新建或追加 track
            if vehicle_id not in self.correction_vehicle_tracks:
                self.correction_vehicle_tracks[vehicle_id] = {
                    'class_name': obj.get('class_name', 'unknown'),
                    'positions': [],
                    'frame_times': [],
                    'ground_positions': [],  # 地面坐标
                    'lonlat_positions': [],  # 经纬度坐标
                    'velocities': [],
                    'directions': [],
                    'bbox_corners_px': [],
                    'bbox_centers_px': [],
                    'bbox_corners_lonlat': [],  # 角点经纬度
                    'bbox_centers_lonlat': [],  # 中心点经纬度
                    'frame_ids': [],  # 添加帧ID列表
                    'lane_events': [],  # 车道事件
                    'last_frame': None
                }
            
            track = self.correction_vehicle_tracks[vehicle_id]
            track['positions'].append(center_px)
            track['frame_times'].append(float(frame_time) if frame_time is not None else 0.0)
            track['bbox_corners_px'].append(corners_px)
            track['bbox_centers_px'].append(center_px)
            track['frame_ids'].append(frame_id)  # 存储对应的帧ID
            track['last_frame'] = frame_id
            
            # 如果启用了坐标计算，计算真实地理坐标
            if self.calculate_correction_coords and self.homography_matrix is not None and self.origin_lonlat is not None:
                try:
                    # 计算角点的地理坐标
                    corners_lonlat = []
                    for (px, py) in corners_px:
                        gx, gy = self.image_to_ground((px, py), self.homography_matrix)
                        lon, lat = self.utm_to_lonlat(gx, gy)
                        corners_lonlat.append((lon, lat))
                    
                    # 计算中心点的地理坐标
                    gx, gy = self.image_to_ground(center_px, self.homography_matrix)
                    clon, clat = self.utm_to_lonlat(gx, gy)
                    center_lonlat = (clon, clat)
                    
                    # 保存地面坐标和经纬度
                    track['ground_positions'].append((float(gx), float(gy)))
                    track['lonlat_positions'].append(center_lonlat)
                    track['bbox_corners_lonlat'].append(corners_lonlat)
                    track['bbox_centers_lonlat'].append(center_lonlat)
                    
                except Exception as e:
                    self.status_text.append(f"坐标转换错误(车辆{vehicle_id}): {e}")
                    track['ground_positions'].append((None, None))
                    track['lonlat_positions'].append((None, None))
                    track['bbox_corners_lonlat'].append([(None, None)] * len(corners_px))
                    track['bbox_centers_lonlat'].append((None, None))
            else:
                # 未启用坐标计算，填充None
                track['ground_positions'].append((None, None))
                track['lonlat_positions'].append((None, None))
                track['bbox_corners_lonlat'].append([(None, None)] * len(corners_px))
                track['bbox_centers_lonlat'].append((None, None))
            
            # 限制轨迹数据长度，避免内存过度使用
            max_track_length = max(500, self.trajectory_length * 2)  # 保留比当前设置多一倍的数据
            if len(track['positions']) > max_track_length:
                # 保留最近的数据
                for key in ['positions', 'frame_times', 'ground_positions', 'lonlat_positions',
                           'bbox_corners_px', 'bbox_centers_px', 'bbox_corners_lonlat', 
                           'bbox_centers_lonlat', 'frame_ids']:
                    if key in track and len(track[key]) > max_track_length:
                        track[key] = track[key][-max_track_length:]
            
            # 计算速度/方向
            if len(track['positions']) > 1:
                self.calculate_correction_kinematics(vehicle_id)
        
        # 更新车辆列表
        self.update_vehicle_list()
    
    def calculate_correction_kinematics(self, vehicle_id):
        """计算校正视频中车辆的运动学参数"""
        track = self.correction_vehicle_tracks.get(vehicle_id)
        if not track:
            return
        
        n = len(track['positions'])
        if n < 2:
            return
        
        x1, y1 = track['positions'][-2]
        x2, y2 = track['positions'][-1]
        t1 = float(track['frame_times'][-2])
        t2 = float(track['frame_times'][-1])
        dt = t2 - t1
        
        if dt <= 0:
            return
        
        dx = x2 - x1
        dy = y2 - y1
        dist_px = np.hypot(dx, dy)
        speed_px = dist_px / dt
        
        # 如果有地面坐标，计算真实速度
        speed_ms = None
        if len(track.get('ground_positions', [])) >= 2:
            gx1, gy1 = track['ground_positions'][-2]
            gx2, gy2 = track['ground_positions'][-1]
            # 检查坐标是否为None
            if gx1 is not None and gy1 is not None and gx2 is not None and gy2 is not None:
                dist_m = np.hypot(gx2 - gx1, gy2 - gy1)
                try:
                    speed_ms = dist_m / dt
                except Exception:
                    speed_ms = None
        
        direction = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        
        track['velocities'].append({
            'speed_px': float(speed_px),
            'speed_ms': float(speed_ms) if speed_ms is not None else None,
            'timestamp': t2
        })
        
        track['directions'].append({
            'direction': float(direction),
            'timestamp': t2
        })
    
    
    def update_correction_vehicle_info(self, vehicle_id):
        """更新校正视频的车辆信息显示"""
        if vehicle_id not in self.correction_vehicle_tracks:
            return
        
        track = self.correction_vehicle_tracks[vehicle_id]
        self.info_table.setRowCount(10)
        self.info_table.clearContents()
        
        # 基本信息
        self.info_table.setItem(0, 0, QTableWidgetItem("车辆ID"))
        self.info_table.setItem(0, 1, QTableWidgetItem(str(vehicle_id)))
        
        self.info_table.setItem(0, 2, QTableWidgetItem("类型"))
        self.info_table.setItem(0, 3, QTableWidgetItem(track['class_name']))
        
        self.info_table.setItem(0, 4, QTableWidgetItem("轨迹点数"))
        self.info_table.setItem(0, 5, QTableWidgetItem(str(len(track['positions']))))
        
        # 当前位置信息（像素坐标）
        if track['positions']:
            x, y = track['positions'][-1]
            self.info_table.setItem(1, 0, QTableWidgetItem("像素坐标(X)"))
            self.info_table.setItem(1, 1, QTableWidgetItem(f"{x:.2f}"))
            
            self.info_table.setItem(1, 2, QTableWidgetItem("像素坐标(Y)"))
            self.info_table.setItem(1, 3, QTableWidgetItem(f"{y:.2f}"))
        
        # 经纬度信息
        if track.get('lonlat_positions') and track['lonlat_positions']:
            lon, lat = track['lonlat_positions'][-1]
            if lon is not None and lat is not None:
                self.info_table.setItem(2, 0, QTableWidgetItem("经度"))
                self.info_table.setItem(2, 1, QTableWidgetItem(f"{lon:.8f}"))
                self.info_table.setItem(2, 2, QTableWidgetItem("纬度"))
                self.info_table.setItem(2, 3, QTableWidgetItem(f"{lat:.8f}"))
            else:
                self.info_table.setItem(2, 0, QTableWidgetItem("经度"))
                self.info_table.setItem(2, 1, QTableWidgetItem("未计算"))
                self.info_table.setItem(2, 2, QTableWidgetItem("纬度"))
                self.info_table.setItem(2, 3, QTableWidgetItem("未计算"))
        
        # 速度信息
        if track['velocities'] and track['velocities'][-1]:
            speed_info = track['velocities'][-1]
            self.info_table.setItem(3, 0, QTableWidgetItem("速度(像素/秒)"))
            self.info_table.setItem(3, 1, QTableWidgetItem(f"{speed_info.get('speed_px', 0):.2f}"))
            
            # 如果有真实速度，显示
            if speed_info.get('speed_ms') is not None:
                self.info_table.setItem(3, 2, QTableWidgetItem("速度(米/秒)"))
                self.info_table.setItem(3, 3, QTableWidgetItem(f"{speed_info['speed_ms']:.2f}"))
                self.info_table.setItem(3, 4, QTableWidgetItem("速度(km/h)"))
                self.info_table.setItem(3, 5, QTableWidgetItem(f"{speed_info['speed_ms']*3.6:.2f}"))
        
        # 方向信息
        if track['directions'] and track['directions'][-1]:
            direction_info = track['directions'][-1]
            self.info_table.setItem(4, 0, QTableWidgetItem("方向"))
            self.info_table.setItem(4, 1, QTableWidgetItem(f"{direction_info.get('direction', 0):.1f}°"))
        
        # 角点信息
        if track['bbox_corners_px']:
            corners = track['bbox_corners_px'][-1]
            for i, (x, y) in enumerate(corners):
                if i < 4:  # 只显示前4个角点
                    self.info_table.setItem(5+i, 0, QTableWidgetItem(f"角点{i+1}(X)"))
                    self.info_table.setItem(5+i, 1, QTableWidgetItem(f"{x:.2f}"))
                    self.info_table.setItem(5+i, 2, QTableWidgetItem(f"角点{i+1}(Y)"))
                    self.info_table.setItem(5+i, 3, QTableWidgetItem(f"{y:.2f}"))
    
    
    def fit_to_window(self):
        self.video_label.reset_view()
        self.zoom_slider.setValue(0)

    # ---------- 视频加载 ----------
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )

        if not file_path:
            return

        # 关闭已有预览/资源
        try:
            if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
                self.preview_timer.stop()
        except Exception:
            pass
        try:
            if getattr(self, 'cap', None) is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        except Exception:
            pass

        # 打开视频并保留 cap 以便预览或后续处理使用
        cap = cv2.VideoCapture(file_path)
        if not cap or not cap.isOpened():
            self.status_text.append(f"无法打开视频文件: {file_path}")
            return

        self.video_path = file_path
        self.cap = cap
        # 读取 fps（若可获取）
        try:
            self.video_fps = cap.get(cv2.CAP_PROP_FPS) or None
        except Exception:
            self.video_fps = None

        # 读取第一帧并显示为静态预览
        ret, frame = cap.read()
        if not ret or frame is None:
            self.status_text.append("读取视频第一帧失败，可能是编码不兼容。")
            # 释放 cap（避免资源占用）
            try:
                cap.release()
                self.cap = None
            except Exception:
                pass
            return

        self.current_frame = frame
        # 如果你想把 frame_id 记录为 0:
        self.last_frame_id = 0
        self.last_frame_time = 0.0
        try:
            self.video_label.set_frame(frame)
        except Exception as e:
            self.status_text.append(f"显示帧失败: {e}")

        # 切换到原始视频模式
        self.current_mode = 'original'
        self.calculate_correction_coords = False  # 重置校正坐标计算标志
        self.update_vehicle_list()  # 刷新车辆列表

        # 重置暂停状态 - 重新加载视频时应该重置所有处理状态
        self._paused = False
        self.pause_btn.setText("暂停")
        
        # 重置进度条和线程状态
        self.progress_bar.setValue(0)
        self.last_processed_count = 0
        self.expected_processed_frames = 0
        
        # 停止任何正在运行的YOLO线程
        if hasattr(self, 'yolo_thread') and self.yolo_thread and self.yolo_thread.isRunning():
            try:
                self.yolo_processor.stop_processing()
                self.yolo_thread.quit()
                self.yolo_thread.wait()
            except Exception:
                pass
            self.yolo_thread = None
        
        # 按下load_video_btn后的按钮状态切换
        self.zoom_in_btn.setEnabled(False)
        self.trajectory_length_edit.setEnabled(False)
        self.zoom_slider.setEnabled(False)
        
        self.load_model_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.refresh_models_btn.setEnabled(True)
        self.load_selected_model_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.play_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)

        self.status_text.append(f'已选择视频: {file_path} (fps={self.video_fps})：已显示第一帧。如需连续预览请点击"播放预览"。')
        
        # 自动加载默认模型
        self.auto_load_default_model()

    def auto_load_default_model(self):
        """自动加载默认模型"""
        try:
            # 设置默认模型文件夹为weights
            default_weights_folder = "weights"
            if os.path.exists(default_weights_folder):
                self.model_folder = default_weights_folder
                self.status_text.append(f"自动设置模型文件夹: {default_weights_folder}")
                
                # 刷新模型列表
                self.refresh_model_list()
                
                # 尝试加载默认模型
                default_models = ["yolo11x-obb.pt", "yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt"]
                for model_name in default_models:
                    if model_name in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
                        self.model_combo.setCurrentText(model_name)
                        self.status_text.append(f"自动选择模型: {model_name}")
                        # 自动加载选中的模型
                        self.load_selected_model()
                        break
                else:
                    self.status_text.append("未找到默认模型文件，请手动选择模型")
            else:
                self.status_text.append("默认模型文件夹weights不存在，请手动选择模型文件夹")
        except Exception as e:
            self.status_text.append(f"自动加载默认模型失败: {str(e)}")

    # ---------- 模型加载 ----------
    def browse_model_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择模型文件夹", "")
        if folder:
            self.model_folder = folder
            self.refresh_models_btn.setEnabled(True)
            self.status_text.append(f"模型文件夹: {folder}")
            # 我们在选择文件夹后自动刷新一次，省得用户还要手动点"刷新"
            try:
                self.refresh_model_list()
            except Exception as e:
                # 如果自动刷新出现问题，仍然允许用户手动刷新
                self.status_text.append(f"自动刷新模型列表失败: {e}")

    def refresh_model_list(self):
        self.model_combo.clear()
        # disable load button by default
        try:
            self.load_selected_model_btn.setEnabled(False)
        except Exception:
            pass

        if not self.model_folder:
            self.status_text.append("未选择模型文件夹")
            return
        try:
            files = os.listdir(self.model_folder)
            candidates = [f for f in files if f.lower().endswith('.onnx') or f.lower().endswith('.pt') or f.lower().endswith('.pth')]
            if not candidates:
                self.model_combo.addItem("（该文件夹中未发现模型）")
                self.model_combo.setEnabled(False)
                # ensure load button disabled
                try:
                    self.load_selected_model_btn.setEnabled(False)
                except Exception:
                    pass
                self.status_text.append(f"在 {self.model_folder} 中未发现模型文件。")
            else:
                for c in candidates:
                    self.model_combo.addItem(c)
                self.model_combo.setEnabled(True)
                # **修正点：发现模型时启用"加载选中模型"按钮**
                try:
                    self.load_selected_model_btn.setEnabled(True)
                except Exception:
                    pass
                self.status_text.append(f"发现 {len(candidates)} 个模型文件")
        except Exception as e:
            self.status_text.append(f"读取模型目录失败: {e}")

    def load_selected_model(self):
        if not self.model_folder:
            self.status_text.append("未选择模型文件夹")
            return

        selected = self.model_combo.currentText()
        if not selected or selected.startswith("（该文件夹中未发现"):
            self.status_text.append("未选择有效模型文件")
            return

        model_path = os.path.join(self.model_folder, selected)
        if not os.path.exists(model_path):
            self.status_text.append(f"模型文件不存在: {model_path}")
            return

        try:
            success = self.yolo_processor.load_model(model_path)
            if success:
                self.status_text.append(f"模型加载成功: {model_path}")
                if self.video_path:
                    self.start_btn.setEnabled(True)
            else:
                self.status_text.append(f"模型加载失败: {model_path}")
                self.status_text.append("提示: 请查看上方日志获取详细错误信息")
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self.status_text.append(f"❌ 加载模型时异常: {model_path}")
            self.status_text.append(f"错误类型: {type(e).__name__}")
            self.status_text.append(f"错误信息: {str(e)}")
            self.status_text.append(f"详细堆栈:\n{error_detail}")

    def start_processing(self):
        if not self.video_path:
            self.status_text.append("错误: 请先加载视频")
            return

        # 检查单应矩阵和原点坐标
        if self.homography_matrix is None or self.origin_lonlat is None:
            # 发送信号检查矩阵和原点
            self.matrix_check_required.emit()
            return
        
        # 初始化进度值
        self.last_processed_count = 0
        self.expected_processed_frames = 0

        try:
            self.yolo_processor.set_tracking_enabled(True)
            self.yolo_processor.set_target_fps(25)
        except Exception:
            pass

        self.yolo_thread = YOLOProcessorThread(self.yolo_processor)
        
        # 连接线程信号到主窗口槽函数
        self.connect_yolo_thread_signals(self.yolo_thread)
        
        try:
            self.yolo_thread.set_video_path(self.video_path)
        except Exception:
            pass

        # === 新增：配置保存选项 ===
        try:
            if getattr(self, 'save_frame_annotations_cb', None) and self.save_frame_annotations_cb.isChecked():
                base = self.save_folder_le.text().strip() if self.save_folder_le.text() else None
                if not base or base == "results":
                    # 使用自动递增的results文件夹
                    base = self.get_next_results_folder()
                    self.save_folder_le.setText(base)  # 更新UI显示
                images_dir = os.path.join(base, "images")
                labels_dir = os.path.join(base, "labels")
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)

                self.yolo_processor.set_export_options(
                    save_txt=True, save_conf=True, output_dir=labels_dir,
                    save_images=True, image_dir=images_dir
                )
                self.status_text.append(f"已启用自动保存：{images_dir} / {labels_dir}")
        except Exception as e:
            self.status_text.append(f"设置自动保存失败：{e}")

        # === 再启动线程 ===
        try:
            self.yolo_thread.start()
            self.status_text.append("开始处理视频（线程已启动）.")
            
            # 开始处理时的按钮状态切换
            self.start_btn.setEnabled(False)
            self.play_preview_btn.setEnabled(False)
            self.stop_preview_btn.setEnabled(False)
            self.load_model_btn.setEnabled(False)
            self.model_combo.setEnabled(False)
            self.refresh_models_btn.setEnabled(False)
            self.load_selected_model_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
        except Exception as e:
            self.status_text.append(f"启动处理失败: {str(e)}")

    def check_matrix_and_continue(self):
        """检查矩阵和原点坐标并询问用户是否继续"""
        # 检查缺少什么
        missing_items = []
        if self.homography_matrix is None:
            missing_items.append("单应矩阵")
        if self.origin_lonlat is None:
            missing_items.append("原点坐标")
        
        missing_str = "、".join(missing_items)
        
        reply = QMessageBox.question(
            self, 
            "坐标设置未完成", 
            f"未设置{missing_str}，将无法计算地理坐标。\n\n需要设置：\n• 单应矩阵（9个数值）\n• 原点经纬度坐标\n• UTM区域和半球\n• 点击'应用矩阵'按钮\n\n选择：\n• 是：继续处理（仅显示YOLO检测结果，不计算真实坐标）\n• 否：取消处理，先设置坐标系统",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            self.status_text.append(f"处理已取消：请先设置{missing_str}")
        else:
            self.status_text.append(f"警告: 未设置{missing_str}，将无法计算地理坐标，仅显示检测结果")
            # 继续处理
            self.continue_processing_without_matrix()

    def continue_processing_without_matrix(self):
        """在没有矩阵的情况下继续处理"""
        try:
            self.yolo_processor.set_tracking_enabled(True)
            self.yolo_processor.set_target_fps(25)
        except Exception:
            pass

        self.yolo_thread = YOLOProcessorThread(self.yolo_processor)
        
        # 连接线程信号到主窗口槽函数
        self.connect_yolo_thread_signals(self.yolo_thread)
        
        try:
            self.yolo_thread.set_video_path(self.video_path)
        except Exception:
            pass

        # === 新增：配置保存选项 ===
        try:
            if getattr(self, 'save_frame_annotations_cb', None) and self.save_frame_annotations_cb.isChecked():
                base = self.save_folder_le.text().strip() if self.save_folder_le.text() else None
                if not base or base == "results":
                    # 使用自动递增的results文件夹
                    base = self.get_next_results_folder()
                    self.save_folder_le.setText(base)  # 更新UI显示
                images_dir = os.path.join(base, "images")
                labels_dir = os.path.join(base, "labels")
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)

                self.yolo_processor.set_export_options(
                    save_txt=True, save_conf=True, output_dir=labels_dir,
                    save_images=True, image_dir=images_dir
                )
                self.status_text.append(f"已启用自动保存：{images_dir} / {labels_dir}")
        except Exception as e:
            self.status_text.append(f"设置自动保存失败：{e}")

        # === 再启动线程 ===
        try:
            self.yolo_thread.start()
            self.status_text.append("开始处理视频（线程已启动）.")
            
            # 开始处理时的按钮状态切换
            self.start_btn.setEnabled(False)
            self.play_preview_btn.setEnabled(False)
            self.stop_preview_btn.setEnabled(False)
            self.load_model_btn.setEnabled(False)
            self.model_combo.setEnabled(False)
            self.refresh_models_btn.setEnabled(False)
            self.load_selected_model_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
        except Exception as e:
            self.status_text.append(f"启动处理失败: {str(e)}")

    def stop_processing(self):
        # 停止处理线程
        if self.yolo_thread and self.yolo_thread.isRunning():
            try:
                self.yolo_thread.stop_processing()
                self.yolo_thread.quit()
                self.yolo_thread.wait()
            except Exception:
                pass

        # 重置暂停状态
        self._paused = False

        # 重置按钮状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停")
        
        # 恢复其他按钮状态
        self.play_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.load_model_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.refresh_models_btn.setEnabled(True)
        self.load_selected_model_btn.setEnabled(True)
        
        self.status_text.append("处理已停止")

    def load_homography_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择单应矩阵文件", "", "NumPy文件 (*.npy);文本文件 (*.txt)"
        )

        if file_path:
            try:
                if file_path.endswith('.npy'):
                    H = np.load(file_path)
                else:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    H = self.parse_homography_text(content)

                if H is not None and H.shape == (3, 3):
                    self.homography_matrix = H
                    h_text = "\n".join([" ".join([f"{v:.12g}" for v in row]) for row in H.tolist()])
                    self.homography_input.setPlainText(h_text)
                    self.status_text.append(f"已加载单应矩阵 from {file_path}")
                else:
                    self.status_text.append("错误: 无效的单应矩阵文件")
            except Exception as e:
                self.status_text.append(f"加载单应矩阵失败: {str(e)}")

    def parse_homography_text(self, text):
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
        if len(numbers) != 9:
            return None
        H = np.array([float(n) for n in numbers]).reshape(3, 3)
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        return H

    def apply_homography(self):
        text = self.homography_input.toPlainText()
        H = self.parse_homography_text(text)

        if H is not None:
            self.homography_matrix = H
            self.status_text.append("单应矩阵已应用")

            self.origin_lonlat = (
                self.origin_lon_input.value(),
                self.origin_lat_input.value()
            )
            self.utm_zone = self.utm_zone_input.value()
            self.is_northern = self.hemisphere_combo.currentText() == "北半球"

            self.status_text.append(f"原点坐标: {self.origin_lonlat}")
            self.status_text.append(f"UTM区域: {self.utm_zone}, {'北半球' if self.is_northern else '南半球'}")
        else:
            self.status_text.append("输入的单应矩阵无效，请检查格式（9 个数字）。")

    def toggle_pause(self):
        """
        暂停 = stop 线程 + 保留当前位置
        继续 = 新建线程 + 从 last_frame_id+1 继续
        """
        paused = getattr(self, "_paused", False)

        if not paused:
            # --- 执行暂停 ---
            self._paused = True
            # 停止处理器线程，但不要清掉 last_frame_id
            try:
                if self.yolo_thread and self.yolo_thread.isRunning():
                    self.yolo_processor.stop_processing()
                    self.yolo_thread.quit()
                    self.yolo_thread.wait()
            except Exception:
                pass

            self.pause_btn.setText("继续")
            # 暂停时保持停止按钮使能，允许用户停止处理
            self.stop_btn.setEnabled(True)
            self.status_text.append(f"已暂停：位置停在帧 {getattr(self, 'last_frame_id', None)}。")
        else:
            # --- 执行继续 ---
            self._paused = False

            try:
                # 重新创建处理器线程
                self.yolo_thread = YOLOProcessorThread(self.yolo_processor)
                
                # 连接线程信号到主窗口槽函数
                self.connect_yolo_thread_signals(self.yolo_thread)

                # 让处理器从 last_frame_id+1 继续
                start_frame = getattr(self, "last_frame_id", None)
                if start_frame is not None:
                    if hasattr(self.yolo_processor, "set_video_start_frame"):
                        self.yolo_processor.set_video_start_frame(int(start_frame) + 1)
                    else:
                        setattr(self.yolo_processor, "_requested_start_frame", int(start_frame) + 1)

                    # 如果 cap 存在，也 seek 一下
                    if getattr(self, "cap", None) is not None:
                        try:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame) + 1)
                        except Exception:
                            pass

                self.yolo_thread.set_video_path(self.video_path)
                self.yolo_thread.start()
                self.status_text.append(f"继续处理，从帧 {int(start_frame)+1 if start_frame is not None else 0} 开始。")
                
                # 恢复进度条状态 - 基于已处理的帧数而不是当前帧ID
                if hasattr(self, 'last_processed_count') and hasattr(self, 'expected_processed_frames'):
                    try:
                        # 恢复进度条的最大值和当前值
                        self.progress_bar.setMaximum(self.expected_processed_frames)
                        self.progress_bar.setValue(self.last_processed_count)
                        self.status_text.append(f"恢复进度条: {self.last_processed_count}/{self.expected_processed_frames}")
                    except Exception as e:
                        self.status_text.append(f"恢复进度条失败: {e}")
                else:
                    self.status_text.append("无法恢复进度条: 缺少进度数据")
                
                # 重新启用按钮状态
                self.start_btn.setEnabled(False)
                self.pause_btn.setEnabled(True)
                self.stop_btn.setEnabled(True)

            except Exception as e:
                self.status_text.append(f"恢复继续失败: {e}")

            self.pause_btn.setText("暂停")



    # ---------- 处理 YOLO/跟踪器 输出 ----------
    def on_frame_processed(self, frame, detection_info):

        # 立即忽略来自处理器的帧（GUI 暂停状态）
        if getattr(self, "_paused", False):
            # 不更新显示、不处理 track（等待用户恢复）
            return
        """
        处理YOLO处理器返回的帧和检测信息
        """
        # 记录并注入系统时间（epoch float + ISO string）
        sys_now = datetime.now()
        system_timestamp = sys_now.timestamp()
        system_time_iso = sys_now.isoformat(timespec='milliseconds')

        # 确保 detection_info 是 dict
        if detection_info is None:
            detection_info = {}

        # Inject system time (如果外部已经提供则保留外部值)
        detection_info.setdefault('system_timestamp', system_timestamp)
        detection_info.setdefault('system_time', system_time_iso)

        # 暂存当前帧（稍后会绘制车道并显示）
        self.current_frame = frame

        # 解析帧时间（视频时间 / 检测器时间） - 保持原有逻辑
        frame_time = None
        for key in ('frame_time', 'timestamp', 'time', 'frame_timestamp'):
            if key in detection_info and detection_info[key] is not None:
                try:
                    frame_time = float(detection_info[key])
                    break
                except Exception:
                    frame_time = None

        # 如果没有时间戳，使用帧号和FPS计算或系统时间作为备份
        if frame_time is None and 'frame_id' in detection_info:
            fid = int(detection_info.get('frame_id', 0))
            fps = getattr(self, "video_fps", None)
            if fps and fps > 0:
                frame_time = fid / float(fps)
            else:
                frame_time = system_timestamp
                self.status_text.append("警告：无法获取视频 FPS，用系统时间作为时间戳")

        detection_info['frame_time'] = frame_time

        # 更新车辆跟踪信息（现在 detection_info 中包含 system_time/system_timestamp）
        self.update_vehicle_tracks(detection_info)
        
        # 车道检测（原始视频）
        for obj in detection_info.get('objects', []):
            vehicle_id = obj.get('track_id') or obj.get('track_uuid')
            if vehicle_id is not None:
                # 获取bbox角点
                if obj.get('bbox_type') == 'obb':
                    pts = np.array(obj.get('bbox', [])).reshape(-1, 2)
                    corners = [(float(pts[i,0]), float(pts[i,1])) for i in range(min(4, pts.shape[0]))]
                else:
                    x1, y1, x2, y2 = obj.get('bbox', [0, 0, 0, 0])
                    corners = [(float(x1), float(y1)), (float(x2), float(y1)),
                              (float(x2), float(y2)), (float(x1), float(y2))]
                
                if corners:
                    # 计算中心点
                    pts_array = np.array(corners)
                    center_x = np.mean(pts_array[:, 0])
                    center_y = np.mean(pts_array[:, 1])
                    center_px = (center_x, center_y)
                    
                    self.update_lane_statistics(vehicle_id, center_px)
        
        # 在帧上绘制车道
        frame = self.draw_lanes_on_frame(frame)
        self.video_label.set_frame(frame)
        
        self.update_vehicle_list()
        self.last_frame_id = detection_info.get('frame_id')
        self.last_frame_time = detection_info.get('frame_time')
        self.last_detection_info = detection_info

        # 进度条更新已由 on_progress_updated 方法处理，这里不再重复更新


    # ---------- 导出数据（CSV/Excel） ----------
    def export_data(self):
        # 根据当前模式选择要导出的数据
        if self.current_mode == 'original':
            tracks_to_export = self.vehicle_tracks
            data_source = "原始视频"
        elif self.current_mode == 'correction':
            tracks_to_export = self.correction_vehicle_tracks
            data_source = "校正视频"
        else:
            tracks_to_export = {}
            data_source = "未知"
        
        if not tracks_to_export:
            QMessageBox.information(self, "提示", f"当前没有{data_source}轨迹数据可导出。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, f"保存{data_source}数据", "", "CSV文件 (*.csv);;Excel文件 (*.xlsx)"
        )

        if not file_path:
            return

        try:
            all_data = []
            for vehicle_id, track in tracks_to_export.items():
                n_positions = len(track['positions'])

                for i in range(n_positions):
                    data_row = {
                        'vehicle_id': vehicle_id,
                        'vehicle_type': track['class_name'],
                        'timestamp': track['frame_times'][i] if i < len(track['frame_times']) else None,
                        'system_time': track['system_time_strs'][i] if i < len(track.get('system_time_strs', [])) else None,
                        'pixel_x': track['positions'][i][0] if i < len(track['positions']) else None,
                        'pixel_y': track['positions'][i][1] if i < len(track['positions']) else None
                    }

                    # 地面坐标
                    if i < len(track['ground_positions']):
                        data_row['ground_x'] = track['ground_positions'][i][0]
                        data_row['ground_y'] = track['ground_positions'][i][1]

                    # 经纬度坐标
                    if i < len(track['lonlat_positions']):
                        data_row['longitude'] = track['lonlat_positions'][i][0]
                        data_row['latitude'] = track['lonlat_positions'][i][1]

                    # 速度信息
                    if i < len(track['velocities']) and track['velocities'][i]:
                        vel = track['velocities'][i]
                        data_row['speed_px_s'] = vel.get('speed_px')
                        data_row['speed_ms'] = vel.get('speed_ms')

                    # 方向信息
                    if i < len(track['directions']) and track['directions'][i]:
                        data_row['direction_deg'] = track['directions'][i].get('direction')

                    # 角点信息
                    if i < len(track['bbox_corners_px']):
                        corners = track['bbox_corners_px'][i]
                        for j, (x, y) in enumerate(corners):
                            if j < 4:
                                data_row[f'corner{j+1}_x'] = x
                                data_row[f'corner{j+1}_y'] = y

                    all_data.append(data_row)

            df = pd.DataFrame(all_data)
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            else:
                df.to_excel(file_path, index=False)

            self.status_text.append(f"数据已导出到: {file_path}")
            QMessageBox.information(self, "导出成功", f"数据已成功导出到 {file_path}")
        except Exception as e:
            self.status_text.append(f"导出数据失败: {str(e)}")
            QMessageBox.critical(self, "导出失败", f"导出数据时发生错误: {str(e)}")


    def _on_preview_play(self):
        if getattr(self, 'cap', None) is None:
            self.status_text.append("请先加载视频再播放预览。")
            return
        # 若能获取真实 fps，则设置定时器间隔为 1000/fps ms
        if getattr(self, 'video_fps', None) and self.video_fps > 1:
            interval = max(10, int(1000.0 / float(self.video_fps)))
            try:
                self.preview_timer.setInterval(interval)
            except Exception:
                pass
        try:
            self.preview_timer.start()
            self.play_preview_btn.setEnabled(False)
            self.stop_preview_btn.setEnabled(True)
            self.status_text.append("开始视频预览")
        except Exception as e:
            self.status_text.append(f"启动预览失败: {e}")

    def _on_preview_stop(self):
        try:
            if getattr(self, 'preview_timer', None):
                self.preview_timer.stop()
            self.play_preview_btn.setEnabled(True)
            self.stop_preview_btn.setEnabled(False)
            self.status_text.append("停止视频预览")
        except Exception as e:
            self.status_text.append(f"停止预览失败: {e}")

    def _preview_next_frame(self):
        """
        QTimer 回调，从 self.cap 读取下一帧并显示。
        到达结尾时自动停止（或回到开头，取决于你想要的行为）。
        """
        cap = getattr(self, 'cap', None)
        if cap is None:
            self.preview_timer.stop()
            self.play_preview_btn.setEnabled(True)
            self.stop_preview_btn.setEnabled(False)
            return

        ret, frame = cap.read()
        if not ret or frame is None:
            # 到达视频末尾 - 停止预览并把 cap 回到起点（或直接停止）
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到开头（如果你希望循环）
                # 如果不想循环改为停止：
                # self.preview_timer.stop()
                # self.play_preview_btn.setEnabled(True)
                # self.stop_preview_btn.setEnabled(False)
                # self.status_text.append("已到达视频末尾，已停止预览。")
                # return
                # 这里选择循环播放以便测试
                ret2, frame = cap.read()
                if not ret2:
                    self.preview_timer.stop()
                    self.play_preview_btn.setEnabled(True)
                    self.stop_preview_btn.setEnabled(False)
                    self.status_text.append("读取视频帧失败，停止预览。")
                    return
            except Exception:
                self.preview_timer.stop()
                self.play_preview_btn.setEnabled(True)
                self.stop_preview_btn.setEnabled(False)
                self.status_text.append("读取视频帧异常，停止预览。")
                return

        # 显示帧
        try:
            self.current_frame = frame
            # 更新 frame id/time（自行按需求设置）
            if hasattr(self, 'last_frame_id') and self.last_frame_id is not None:
                try:
                    self.last_frame_id += 1
                except Exception:
                    self.last_frame_id = getattr(self, 'last_frame_id', 0)
            else:
                self.last_frame_id = 0
            self.last_frame_time = datetime.now().timestamp()
            self.video_label.set_frame(frame)
        except Exception as e:
            self.status_text.append(f"预览显示失败: {e}")


    def clear_data(self):
        self.vehicle_tracks.clear()
        self.update_vehicle_list()
        self.status_text.append("已清除轨迹数据")

    def on_progress_updated(self, cur, total, progress_percent):
        try:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(cur)
            # 保存进度值，用于暂停后恢复
            self.last_processed_count = cur
            self.expected_processed_frames = total
            # 只在进度有变化时显示，避免重复显示
            if not hasattr(self, '_last_progress_displayed') or self._last_progress_displayed != cur:
                if cur > 0:  # 只在有进度时显示，避免刷屏
                    self.status_text.append(f"进度更新: {cur}/{total} ({progress_percent}%)")
                self._last_progress_displayed = cur
        except Exception as e:
            self.status_text.append(f"进度更新失败: {e}")

    def on_fps_updated(self, fps):
        self.status_text.append(f"处理器 FPS: {fps:.2f}")

    def on_error_occurred(self, errmsg):
        self.status_text.append("模型错误: " + str(errmsg))

    def on_processing_finished(self):
        self.status_text.append("处理完成")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_detection_info_updated(self, info):
        self.status_text.append(str(info))

    def on_model_loaded(self, model_path):
        self.status_text.append(f"模型已加载: {model_path}")
    
    def on_video_info_updated(self, total_frames, original_fps, skip_frames, target_fps):
        """处理视频信息更新"""
        self.status_text.append(f"视频信息: {total_frames}帧, {original_fps:.1f}FPS, 目标{target_fps}FPS, 跳帧{skip_frames}")
    
    def connect_yolo_thread_signals(self, thread):
        """连接YOLO线程信号到主窗口槽函数"""
        thread.frame_processed.connect(self.on_frame_processed)
        thread.progress_updated.connect(self.on_progress_updated)
        thread.fps_updated.connect(self.on_fps_updated)
        thread.error_occurred.connect(self.on_error_occurred)
        thread.processing_finished.connect(self.on_processing_finished)
        thread.detection_info_updated.connect(self.on_detection_info_updated)
        thread.model_loaded.connect(self.on_model_loaded)
        thread.video_info_updated.connect(self.on_video_info_updated)

    def on_vehicle_selected(self):
        selected_items = self.vehicle_list.selectedItems()
        if not selected_items:
            self.selected_vehicle_id = None
            self.correction_selected_vehicle_id = None
            return
        
        item_text = selected_items[0].text()
        try:
            # 从格式化的文本中提取车辆ID
            # 新格式: "ID: {vehicle_id} - {class_name} - 轨迹点: {len} - {status}"
            parts = item_text.split(":")
            if len(parts) > 1:
                vehicle_id_str = parts[1].split("-")[0].strip()
                vehicle_id = int(vehicle_id_str)
                
                # 根据当前模式选择对应的车辆
                if self.current_mode == 'original':
                    # 原始视频模式
                    if vehicle_id in self.vehicle_tracks:
                        self.selected_vehicle_id = vehicle_id
                        self.correction_selected_vehicle_id = None
                        self.update_vehicle_info(vehicle_id)
                    else:
                        self.status_text.append(f"车辆ID {vehicle_id} 不存在")
                
                elif self.current_mode == 'correction':
                    # 校正视频模式
                    if vehicle_id in self.correction_vehicle_tracks:
                        self.correction_selected_vehicle_id = vehicle_id
                        self.selected_vehicle_id = None
                        self.update_correction_vehicle_info(vehicle_id)
                    else:
                        self.status_text.append(f"车辆ID {vehicle_id} 不存在")
                        
        except (ValueError, IndexError) as e:
            self.status_text.append(f"无法解析车辆ID: {e}")

    def update_vehicle_info(self, vehicle_id):
        if vehicle_id not in self.vehicle_tracks:
            return
        
        track = self.vehicle_tracks[vehicle_id]
        # 清空表格并设置为2列
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["属性", "值"])
        self.info_table.setRowCount(0) # 先清空行
        
        def add_row(label, value):
            row = self.info_table.rowCount()
            self.info_table.insertRow(row)
            self.info_table.setItem(row, 0, QTableWidgetItem(label))
            self.info_table.setItem(row, 1, QTableWidgetItem(str(value)))

        # 基本信息
        add_row("车辆ID", vehicle_id)
        add_row("类型", track['class_name'])
        add_row("轨迹点数", len(track['positions']))
        
        # 当前位置信息
        if track['positions']:
            x, y = track['positions'][-1]
            add_row("当前位置(X)", f"{x:.2f}")
            add_row("当前位置(Y)", f"{y:.2f}")
        
        # 经纬度信息
        if track['lonlat_positions']:
            lon, lat = track['lonlat_positions'][-1]
            if lon is not None and lat is not None:
                add_row("经度", f"{lon:.8f}")
                add_row("纬度", f"{lat:.8f}")
            else:
                add_row("经度", "N/A")
                add_row("纬度", "N/A")
        
        # 速度信息
        if track['velocities'] and track['velocities'][-1]:
            speed_info = track['velocities'][-1]
            add_row("速度(像素/秒)", f"{speed_info.get('speed_px_s', 0):.2f}")
            
            if speed_info.get('speed_ms') is not None:
                add_row("速度(米/秒)", f"{speed_info['speed_ms']:.2f}")
                add_row("速度(km/h)", f"{speed_info['speed_ms']*3.6:.2f}")
        
        # 方向信息
        if track['directions'] and track['directions'][-1]:
            direction_info = track['directions'][-1]
            add_row("方向", f"{direction_info.get('direction', 0):.1f}°")
        
        # 角点信息
        if track['bbox_corners_px']:
            corners = track['bbox_corners_px'][-1]
            for i, (x, y) in enumerate(corners):
                if i < 4:
                    add_row(f"角点{i+1}(X)", f"{x:.2f}")
                    add_row(f"角点{i+1}(Y)", f"{y:.2f}")



    def get_next_lane_number(self):
        """获取下一个可用的车道编号（复用已删除的）"""
        existing_numbers = set()
        for lane_id in self.lanes:
            try:
                num = int(lane_id.split('_')[1])
                existing_numbers.add(num)
            except (IndexError, ValueError):
                pass
        
        n = 1
        while n in existing_numbers:
            n += 1
        return n
    
    def create_lane_widget(self):
        """创建单个车道的配置Widget（支持多点绘制）"""
        # 使用复用逻辑获取编号
        lane_num = self.get_next_lane_number()
        lane_id = f"lane_{lane_num}"
        lane_name = f"车道{lane_num}"
        
        # 创建车道数据（支持多点）
        self.lanes[lane_id] = {
            'name': lane_name,
            'points': [],              # 用户选择的控制点列表
            'polygon_points': None,    # 插值后的多边形点（用于检测和绘制）
            'visible': False,          # 完成绘制后才可见
            'vehicles_inside': set(),  # 当前在车道内的车辆
            'counted_vehicles': set(), # 已计入累计的车辆（防止重复计数）
            'pass_count': 0,           # 累计通过车辆数
            'pass_timestamps': [],     # 通过时间戳列表（用于计算veh/h）
            'flow_count': 0,           # 当前在车道内的车辆数
            'queue_count': 0,          # 排队车辆数
            'flow_rate': 0,            # 流量率 veh/h
        }
        
        # 创建Widget
        lane_widget = QWidget()
        lane_widget.setStyleSheet("border: 1px solid #ccc; border-radius: 4px;")
        lane_layout = QVBoxLayout(lane_widget)
        lane_layout.setContentsMargins(8, 8, 8, 8)
        lane_layout.setSpacing(6)
        
        # 第一行：名称 + 操作按钮
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        
        name_label = QLabel(lane_name)
        name_label.setStyleSheet("font-weight: bold; font-size: 13px; border: none;")
        row1.addWidget(name_label)
        
        # 添加点按钮
        add_point_btn = QPushButton("添加点")
        add_point_btn.setToolTip("点击后在视频中添加一个控制点")
        add_point_btn.clicked.connect(lambda: self.on_add_lane_point(lane_id, add_point_btn))
        row1.addWidget(add_point_btn)
        
        # 点数显示
        points_label = QLabel("(0点)")
        points_label.setStyleSheet("color: #666; border: none;")
        row1.addWidget(points_label)
        
        # 完成绘制按钮
        finish_btn = QPushButton("完成")
        finish_btn.setToolTip("完成多边形绘制（至少3个点）")
        finish_btn.clicked.connect(lambda: self.on_finish_lane_drawing(lane_id))
        row1.addWidget(finish_btn)
        
        row1.addStretch()
        
        # 删除按钮
        delete_btn = QPushButton("×")
        delete_btn.setToolTip("删除车道")
        delete_btn.setFixedSize(28, 28)
        delete_btn.setStyleSheet("color: #ff5555; font-weight: bold; font-size: 16px; border: none;")
        delete_btn.clicked.connect(lambda: self.on_delete_lane(lane_id, lane_widget))
        row1.addWidget(delete_btn)
        
        lane_layout.addLayout(row1)
        
        # 第二行：流量统计
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(20)
        
        flow_label = QLabel("当前: 0")
        flow_label.setStyleSheet("color: #2196f3; font-size: 12px; font-weight: bold; border: none;")
        flow_label.setToolTip("当前车道内车辆数")
        row2.addWidget(flow_label)
        
        queue_label = QLabel("排队: 0")
        queue_label.setStyleSheet("color: #ff9800; font-size: 12px; font-weight: bold; border: none;")
        queue_label.setToolTip("静止不动的车辆数")
        row2.addWidget(queue_label)
        
        pass_label = QLabel("累计: 0")
        pass_label.setStyleSheet("color: #4caf50; font-size: 12px; font-weight: bold; border: none;")
        pass_label.setToolTip("累计通过车辆数")
        row2.addWidget(pass_label)
        
        rate_label = QLabel("0 veh/h")
        rate_label.setStyleSheet("color: #9c27b0; font-size: 12px; font-weight: bold; border: none;")
        rate_label.setToolTip("近5分钟换算的小时流量")
        row2.addWidget(rate_label)
        
        row2.addStretch()
        lane_layout.addLayout(row2)
        
        # 存储引用
        self.lanes[lane_id]['widget'] = lane_widget
        self.lanes[lane_id]['add_point_btn'] = add_point_btn
        self.lanes[lane_id]['points_label'] = points_label
        self.lanes[lane_id]['finish_btn'] = finish_btn
        self.lanes[lane_id]['flow_label'] = flow_label
        self.lanes[lane_id]['queue_label'] = queue_label
        self.lanes[lane_id]['pass_label'] = pass_label
        self.lanes[lane_id]['rate_label'] = rate_label
        
        # 添加到布局
        count = self.lanes_layout.count()
        self.lanes_layout.insertWidget(count - 1, lane_widget)
        
        self.status_text.append(f"已添加{lane_name}，请点击\"添加点\"在视频中依次选择控制点，完成后点击\"完成\"")

    def update_vehicle_list(self):
        self.vehicle_list.clear()
        
        # 根据当前模式只显示对应的车辆
        if self.current_mode == 'original':
            # 原始视频模式：只显示原始视频的车辆
            for vehicle_id, track in sorted(self.vehicle_tracks.items()):
                item_text = f"ID: {vehicle_id} - {track['class_name']} - 轨迹点: {len(track['positions'])}"
                # 添加车道事件信息
                lane_events = self.get_lane_events_summary(vehicle_id)
                if lane_events:
                    item_text += lane_events
                self.vehicle_list.addItem(item_text)
            
            if self.vehicle_tracks:
                self.export_btn.setEnabled(True)
        
        elif self.current_mode == 'correction':
            # 校正视频模式：只显示校正视频的车辆
            for vehicle_id, track in sorted(self.correction_vehicle_tracks.items()):
                coords_status = "已计算坐标" if self.calculate_correction_coords else "未计算坐标"
                item_text = f"ID: {vehicle_id} - {track['class_name']} - 轨迹点: {len(track['positions'])} - {coords_status}"
                # 添加车道事件信息
                lane_events = self.get_lane_events_summary(vehicle_id)
                if lane_events:
                    item_text += lane_events
                self.vehicle_list.addItem(item_text)
            
            if self.correction_vehicle_tracks:
                self.export_btn.setEnabled(True)

    def on_video_label_clicked(self, x, y):
        self.status_text.append(f"点击像素: ({x:.1f}, {y:.1f})")
        
        # 优先处理车道坐标选择
        if self.waiting_for_coord is not None:
            self.on_video_coord_selected(x, y)
        else:
            # 否则调用设置原点的方法
            self.set_origin_point(int(x), int(y))

    def set_origin_point(self, img_x, img_y):
        dialog = OriginDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            lon, lat = dialog.get_coords()
            if lon is not None and lat is not None:
                self.origin_lonlat = (lon, lat)
                self.origin_pixel = (img_x, img_y)

                # 自动计算 UTM 区号与半球
                self.calculate_utm_zone()

                self.origin_lon_input.setValue(lon)
                self.origin_lat_input.setValue(lat)
                self.status_text.append(f"已设置原点: 像素坐标({img_x}, {img_y}), 经纬度({lon}, {lat})")

    def calculate_utm_zone(self):
        if not getattr(self, "origin_lonlat", None):
            self.status_text.append("无法计算 UTM 区域：尚未设置原点经纬度。")
            return None

        lon, lat = self.origin_lonlat
        zone = int((float(lon) + 180.0) / 6.0) + 1
        zone = max(1, min(60, zone))
        self.utm_zone = zone
        try:
            self.utm_zone_input.setValue(zone)
        except Exception:
            pass

        is_north = float(lat) >= 0.0
        self.is_northern = is_north
        try:
            self.hemisphere_combo.setCurrentText("北半球" if is_north else "南半球")
        except Exception:
            pass

        self.status_text.append(f"已自动计算 UTM 区号: {zone}，半球: {'北半球' if is_north else '南半球'}")
        return zone

        # ---------- 关键函数：update_vehicle_tracks（记录 bbox 四角点 + 中心点） ----------
    def update_vehicle_tracks(self, detection_info):
        """
        detection_info 结构：
        {
            'frame_id': int,
            'frame_time': float_seconds,   # 视频时间（可选）
            'system_timestamp': float,     # 新增：系统 epoch 秒（float）
            'system_time': str,            # 新增：系统 ISO 字符串（毫秒精度）
            'objects': [ { 'track_id': id, 'bbox': [.], 'bbox_type': 'obb'/'rect', 'class_name': str }, . ]
        }
        """
        frame_id = detection_info.get('frame_id', None)
        frame_time = detection_info.get('frame_time', None)

        # 使用传入的 system_timestamp（若有），否则用当前系统时间
        system_ts = detection_info.get('system_timestamp', None)
        if system_ts is None:
            system_ts = datetime.now().timestamp()
        # system iso 字符串
        system_iso = detection_info.get('system_time', datetime.fromtimestamp(system_ts).isoformat(timespec='milliseconds'))

        current_ids = set()
        for obj in detection_info.get('objects', []):
            vehicle_id = obj.get('track_id', None)
            if vehicle_id is None:
                vehicle_id = obj.get('track_uuid', None)
            if vehicle_id is None:
                continue
            current_ids.add(vehicle_id)

            # 解析 bbox -> corners_px & center_px
            if obj.get('bbox_type') == 'obb':
                pts = np.array(obj.get('bbox', [])).reshape(-1, 2)
                corners_px = [(float(pts[i,0]), float(pts[i,1])) for i in range(pts.shape[0])]
                if len(corners_px) > 4:
                    corners_px = corners_px[:4]
                elif len(corners_px) < 4:
                    xs = pts[:,0]; ys = pts[:,1]
                    x_min, x_max = float(np.min(xs)), float(np.max(xs))
                    y_min, y_max = float(np.min(ys)), float(np.max(ys))
                    corners_px = [(x_min,y_min),(x_max,y_min),(x_max,y_max),(x_min,y_max)]
                cx = float(np.mean([p[0] for p in corners_px]))
                cy = float(np.mean([p[1] for p in corners_px]))
                center_px = (cx, cy)
            else:
                x1, y1, x2, y2 = obj.get('bbox', [0,0,0,0])
                corners_px = [(float(x1), float(y1)), (float(x2), float(y1)),
                            (float(x2), float(y2)), (float(x1), float(y2))]
                center_px = ((float(x1)+float(x2))/2.0, (float(y1)+float(y2))/2.0)

            # 计算经纬（若 H 可用）
            corners_lonlat = []
            center_lonlat = (None, None)
            if self.homography_matrix is not None and self.origin_lonlat is not None:
                try:
                    for (px, py) in corners_px:
                        gx, gy = self.image_to_ground((px, py), self.homography_matrix)
                        lon, lat = self.utm_to_lonlat(gx, gy)
                        corners_lonlat.append((lon, lat))
                    gx, gy = self.image_to_ground(center_px, self.homography_matrix)
                    clon, clat = self.utm_to_lonlat(gx, gy)
                    center_lonlat = (clon, clat)
                except Exception as e:
                    self.status_text.append(f"坐标转换错误(角点/中心): {e}")
                    corners_lonlat = [(None, None)] * len(corners_px)
                    center_lonlat = (None, None)
            else:
                corners_lonlat = [(None, None)] * len(corners_px)
                center_lonlat = (None, None)

            # 新建或追加 track（新增 system_times / system_time_strs / bbox_centers_px 保持）
            if vehicle_id not in self.vehicle_tracks:
                self.vehicle_tracks[vehicle_id] = {
                    'class_name': obj.get('class_name', 'unknown'),
                    'positions': [],
                    'frame_times': [],
                    'system_times': [],           # 新增：epoch float
                    'system_time_strs': [],       # 新增：ISO 字符串
                    'ground_positions': [],
                    'lonlat_positions': [],
                    'velocities': [],
                    'directions': [],
                    'bbox_corners_px': [],
                    'bbox_centers_px': [],
                    'bbox_corners_lonlat': [],
                    'bbox_centers_lonlat': [],
                    'lane_events': [],            # 车道事件
                    'last_frame': None
                }

            track = self.vehicle_tracks[vehicle_id]
            track['positions'].append(center_px)
            # 兼容原有 frame_time（视频时间）
            track['frame_times'].append(float(frame_time) if frame_time is not None else None)
            # 存系统时间
            try:
                track['system_times'].append(float(system_ts))
                track['system_time_strs'].append(str(system_iso))
            except Exception:
                track['system_times'].append(float(datetime.now().timestamp()))
                track['system_time_strs'].append(datetime.now().isoformat(timespec='milliseconds'))

            # ground/ lonlat center (compat)
            if self.homography_matrix is not None and self.origin_lonlat is not None:
                try:
                    gx, gy = self.image_to_ground(center_px, self.homography_matrix)
                    track['ground_positions'].append((float(gx), float(gy)))
                    lon, lat = self.utm_to_lonlat(gx, gy)
                    track['lonlat_positions'].append((lon, lat))
                except Exception:
                    track['ground_positions'].append((None, None))
                    track['lonlat_positions'].append((None, None))
            else:
                track['ground_positions'].append((None, None))
                track['lonlat_positions'].append((None, None))

            track['bbox_corners_px'].append(corners_px)
            track['bbox_centers_px'].append(center_px)
            track['bbox_corners_lonlat'].append(corners_lonlat)
            track['bbox_centers_lonlat'].append(center_lonlat)
            track['last_frame'] = frame_id

            # 计算速度/方向（calculate_kinematics 内部会优先使用 system_times）
            if len(track['positions']) > 1:
                self.calculate_kinematics(vehicle_id)

        # 检测上一帧有但本帧缺失的 track -> 要求人工修正（保持原有逻辑）
        missing_ids = []
        for vid, tr in list(self.vehicle_tracks.items()):
            last = tr.get('last_frame', None)
            if last is not None and frame_id is not None:
                if last == frame_id - 1 and vid not in current_ids:
                    missing_ids.append(vid)

        if missing_ids:
            self.pending_missing_ids = missing_ids
            self.status_text.append(f'检测到漏检车辆 id: {missing_ids}。请在需要时点击"手动修正"按钮打开标注工具进行回填（不会自动弹出）。')
            try:
                self.request_manual_btn.setEnabled(True)
            except Exception:
                pass


    def calculate_kinematics(self, vehicle_id):
        track = self.vehicle_tracks.get(vehicle_id)
        if not track:
            return

        n = len(track['positions'])
        if n < 2:
            return

        x1, y1 = track['positions'][-2]
        x2, y2 = track['positions'][-1]
        t1 = float(track['frame_times'][-2])
        t2 = float(track['frame_times'][-1])
        dt = t2 - t1

        if dt <= 0:
            # 避免除零或负时间差
            return

        dx = x2 - x1
        dy = y2 - y1
        dist_px = np.hypot(dx, dy)
        speed_px = dist_px / dt

        speed_ms = None
        if len(track.get('ground_positions', [])) >= 2:
            gx1, gy1 = track['ground_positions'][-2]
            gx2, gy2 = track['ground_positions'][-1]
            # 检查坐标是否为None，避免None值计算错误
            if gx1 is not None and gy1 is not None and gx2 is not None and gy2 is not None:
                dist_m = np.hypot(gx2 - gx1, gy2 - gy1)
                try:
                    speed_ms = dist_m / dt
                except Exception:
                    speed_ms = None

        direction = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

        track['velocities'].append({
            'speed_px': float(speed_px),
            'speed_ms': float(speed_ms) if speed_ms is not None else None,
            'timestamp': t2
        })

        track['directions'].append({
            'direction': float(direction),
            'timestamp': t2
        })

    def haversine_m(self, lon1, lat1, lon2, lat2):
        # approximate distance (meters) between two lon/lat points
        try:
            R = 6371000.0
            phi1 = math.radians(lat1); phi2 = math.radians(lat2)
            dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        except Exception:
            return None

    def image_to_ground(self, pt_xy, H):
        arr = np.array([[[float(pt_xy[0]), float(pt_xy[1])]]], dtype=np.float64)
        
        # 如果启用畸变校正，先对坐标进行校正
        if self.undistort_enabled and self.camera_matrix is not None and self.dist_coeffs is not None:
            arr_f32 = arr.astype(np.float32)
            arr_undist = cv2.undistortPoints(arr_f32, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
            arr = arr_undist.astype(np.float64)
        
        proj = cv2.perspectiveTransform(arr, H).reshape(2)
        return float(proj[0]), float(proj[1])

    def utm_to_lonlat(self, easting, northing):
        try:
            from pyproj import CRS, Transformer
            if self.origin_lonlat is None:
                return None, None

            origin_lon, origin_lat = self.origin_lonlat
            zone = int((origin_lon + 180) / 6) + 1
            is_north = origin_lat >= 0
            epsg = 32600 + zone if is_north else 32700 + zone

            utm_crs = CRS.from_epsg(epsg)
            transformer = Transformer.from_crs(utm_crs, 'EPSG:4326', always_xy=True)

            origin_transformer = Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
            ox, oy = origin_transformer.transform(origin_lon, origin_lat)
            world_x = easting + ox
            world_y = northing + oy

            lon, lat = transformer.transform(world_x, world_y)
            return lon, lat
        except Exception as e:
            self.status_text.append(f"UTM转经纬度错误: {str(e)}")
            return None, None

    def zoom_fit(self):
        self.video_label.reset_view()

    def get_next_results_folder(self):
        """获取下一个可用的results文件夹编号"""
        base_dir = "results"
        if not os.path.exists(base_dir):
            return os.path.join(base_dir, "01")
        
        # 查找已存在的编号
        existing_numbers = []
        for item in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, item)) and item.isdigit():
                try:
                    existing_numbers.append(int(item))
                except ValueError:
                    continue
        
        if not existing_numbers:
            return os.path.join(base_dir, "01")
        
        # 找到下一个编号
        next_number = max(existing_numbers) + 1
        return os.path.join(base_dir, f"{next_number:02d}")
    
    def on_choose_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择保存目录", "")
        if folder:
            self.save_folder_le.setText(folder)
            self.status_text.append(f"保存目录设置为: {folder}")

    def on_request_manual_correction(self):
        """
        用户点击"手动修正"时调用：
        - 直接打开 labelImg，让用户自己选择路径和文件
        """
        try:
            # 直接启动 labelImg，不指定特定文件
            launched = False
            try:
                import labelImg
                mw = labelImg.MainWindow()
                mw.show()
                mw.raise_()
                mw.activateWindow()
                launched = True
                self.status_text.append("已打开 labelImg，请选择要编辑的图片和标签文件")
            except Exception:
                pass

            if not launched:
                # 尝试通过子进程启动
                candidate = None
                base_dir = os.path.abspath(os.path.dirname(__file__))
                cur = base_dir
                for _ in range(6):
                    p = os.path.join(cur, "projects", "labelimg_OBB", "labelImg.py")
                    if os.path.exists(p):
                        candidate = p
                        break
                    cur = os.path.dirname(cur)
                
                if candidate is None:
                    alt = os.path.join(os.path.dirname(__file__), "labelImg.py")
                    if os.path.exists(alt):
                        candidate = alt
                
                if candidate:
                    subprocess.Popen([sys.executable, candidate])
                    self.status_text.append("已启动 labelImg，请选择要编辑的图片和标签文件")
                else:
                    self.status_text.append("未找到 labelImg，请确保 labelImg 已正确安装")
                    
        except Exception as e:
            self.status_text.append(f"打开 labelImg 失败: {e}")

    def on_open_geo_mapper(self):
        """
        用户点击"计算单应矩阵（GCP工具）"时调用：
        - 启动 GeoVideoMapper.py 程序
        """
        try:
            launched = False
            
            # 尝试直接导入运行
            try:
                # 构建 GeoVideoMapper.py 的路径
                base_dir = os.path.abspath(os.path.dirname(__file__))
                geo_mapper_path = os.path.join(base_dir, "..", "toolbox", "danying", "GCP Homography Tool", "GeoVideoMapper.py")
                geo_mapper_path = os.path.normpath(geo_mapper_path)
                
                if os.path.exists(geo_mapper_path):
                    # 使用子进程启动
                    subprocess.Popen([sys.executable, geo_mapper_path])
                    launched = True
                    self.status_text.append(f"已启动单应矩阵计算工具: {geo_mapper_path}")
                else:
                    self.status_text.append(f"未找到 GeoVideoMapper.py: {geo_mapper_path}")
            except Exception as e:
                self.status_text.append(f"启动失败: {e}")
            
            # 如果上面没有成功，尝试其他可能的路径
            if not launched:
                # 尝试相对于当前文件的其他路径
                alt_paths = [
                    os.path.join(base_dir, "toolbox", "danying", "GCP Homography Tool", "GeoVideoMapper.py"),
                    os.path.join(os.path.dirname(base_dir), "toolbox", "danying", "GCP Homography Tool", "GeoVideoMapper.py"),
                ]
                
                for alt_path in alt_paths:
                    alt_path = os.path.normpath(alt_path)
                    if os.path.exists(alt_path):
                        subprocess.Popen([sys.executable, alt_path])
                        self.status_text.append(f"已启动单应矩阵计算工具: {alt_path}")
                        launched = True
                        break
                
                if not launched:
                    self.status_text.append("未找到 GeoVideoMapper.py，请确保文件位于 toolbox/danying/GCP Homography Tool/ 目录下")
                    QMessageBox.warning(self, "找不到工具", 
                                      "未找到单应矩阵计算工具 (GeoVideoMapper.py)\n"
                                      "请确保文件位于: toolbox/danying/GCP Homography Tool/GeoVideoMapper.py")
                    
        except Exception as e:
            self.status_text.append(f"打开单应矩阵计算工具失败: {e}")
            QMessageBox.critical(self, "启动失败", f"无法启动单应矩阵计算工具:\n{str(e)}")


    def _deleted_prompt_manual_correction(self, frame_bgr, frame_id, pending_ids, detection_info):
        """已删除的方法 - 不再需要"""
        pass

    def _deleted_apply_manual_annotations(self, frame_id, new_objs, frame_time=None):
        """已删除的方法 - 不再需要"""
        pass

    # ==================== 车道检测相关方法 ====================
    

    def on_add_lane_point(self, lane_id, button):
        """进入连续添加点模式"""
        if lane_id not in self.lanes:
            return
        
        # 如果已经在添加点模式，退出
        if self.waiting_for_coord is not None:
            old_lane_id, _, old_button = self.waiting_for_coord
            if old_lane_id == lane_id:
                # 同一车道，退出添加模式
                old_button.setStyleSheet("")
                old_button.setText("添加点")
                self.waiting_for_coord = None
                self.status_text.append("已退出添加点模式")
                return
            else:
                self.status_text.append("请先完成当前车道的点选择")
                return
        
        self.waiting_for_coord = (lane_id, 'add_point', button)
        button.setStyleSheet("background-color: #4CAF50; color: white;")
        button.setText("添加中...")
        self.status_text.append(f"进入连续添加点模式，请在视频中依次点击{self.lanes[lane_id]['name']}的控制点，完成后点击\"完成\"按钮")
    
    def on_video_coord_selected(self, x, y):
        """视频点击后的坐标处理（连续添加模式）"""
        if self.waiting_for_coord is None:
            return
        
        lane_id, action, button = self.waiting_for_coord
        
        if lane_id not in self.lanes:
            self.waiting_for_coord = None
            return
        
        lane = self.lanes[lane_id]
        
        if action == 'add_point':
            # 添加新控制点
            lane['points'].append((int(x), int(y)))
            point_count = len(lane['points'])
            
            # 更新点数显示
            if 'points_label' in lane:
                lane['points_label'].setText(f"({point_count}点)")
            
            self.status_text.append(f"第{point_count}点: ({int(x)},{int(y)})")
            
            # 【保持等待状态，继续添加点】
            # 不重置 self.waiting_for_coord
            
            # 实时预览（如果有3个以上的点）
            if point_count >= 3:
                self.calculate_lane_polygon(lane_id)
                lane['visible'] = True
                self.refresh_current_frame()
    
    def on_finish_lane_drawing(self, lane_id):
        """完成车道绘制"""
        if lane_id not in self.lanes:
            return
        
        lane = self.lanes[lane_id]
        point_count = len(lane['points'])
        
        if point_count < 3:
            self.status_text.append(f"{lane['name']}: 至少需要3个点才能完成绘制")
            return
        
        # 退出添加点模式
        if self.waiting_for_coord is not None:
            waiting_lane_id, _, button = self.waiting_for_coord
            if waiting_lane_id == lane_id:
                button.setStyleSheet("")
                self.waiting_for_coord = None
        
        # 计算最终多边形
        self.calculate_lane_polygon(lane_id)
        lane['visible'] = True
        
        # 禁用添加点按钮
        if 'add_point_btn' in lane:
            lane['add_point_btn'].setEnabled(False)
            lane['add_point_btn'].setText("已完成")
            lane['add_point_btn'].setStyleSheet("")
        if 'finish_btn' in lane:
            lane['finish_btn'].setEnabled(False)
        
        self.status_text.append(f"{lane['name']}绘制完成（{point_count}个控制点）")
        self.refresh_current_frame()
    
    def calculate_lane_polygon(self, lane_id):
        """直接使用控制点作为多边形顶点（折线多边形）"""
        lane = self.lanes[lane_id]
        points = lane['points']
        
        if len(points) < 3:
            lane['polygon_points'] = None
            return
        
        # 直接使用用户选择的点作为多边形顶点
        lane['polygon_points'] = points.copy()
    
    def on_delete_lane(self, lane_id, widget):
        """删除车道"""
        if lane_id in self.lanes:
            lane_name = self.lanes[lane_id]['name']
            del self.lanes[lane_id]
            self.lanes_layout.removeWidget(widget)
            widget.deleteLater()
            self.status_text.append(f"已删除{lane_name}")
            
            # 刷新显示
            self.refresh_current_frame()
    
    def on_lane_detection_toggle(self, state):
        """切换车道检测开关"""
        self.lane_detection_enabled = (state == 2)  # Qt.CheckState.Checked = 2
        
        if self.lane_detection_enabled:
            # 检查是否有有效的车道
            valid_lanes = [l for l in self.lanes.values() 
                          if l.get('polygon_points') and len(l['polygon_points']) >= 3]
            if not valid_lanes:
                self.status_text.append("警告: 没有有效的车道配置，请先添加并绘制车道")
            else:
                self.status_text.append("车道检测已启用")
        else:
            self.status_text.append("车道检测已禁用")
    
    def draw_lanes_on_frame(self, frame):
        """在帧上绘制所有车道（支持多边形曲线）"""
        if not self.lanes:
            return frame
        
        for lane_id, lane in self.lanes.items():
            if not lane.get('visible'):
                continue
            
            # 获取多边形点
            polygon_points = lane.get('polygon_points')
            if not polygon_points or len(polygon_points) < 3:
                continue
            
            # 绘制颜色
            color = (255, 128, 0)  # 橙色
            
            # 绘制填充多边形（半透明）
            overlay = frame.copy()
            pts = np.array(polygon_points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (255, 200, 100))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # 绘制边界线
            cv2.polylines(frame, [pts], True, color, 2)
            
            # 绘制控制点（小圆点）
            for i, pt in enumerate(lane.get('points', [])):
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            
            # 绘制车道编号
            lane_number = lane_id.split('_')[-1] if '_' in lane_id else '1'
            if polygon_points:
                x, y = int(polygon_points[0][0]), int(polygon_points[0][1])
                cv2.putText(frame, lane_number, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame
    
    def calculate_iou(self, box1, box2):
        """计算两个多边形的IOU"""
        try:
            # box1和box2都是角点列表 [(x1,y1), (x2,y2), ...]
            poly1 = np.array(box1, dtype=np.float32)
            poly2 = np.array(box2, dtype=np.float32)
            
            # 使用OpenCV计算交集
            _, intersection = cv2.intersectConvexConvex(poly1, poly2)
            
            if intersection is None or len(intersection) < 3:
                return 0.0
            
            # 计算交集面积
            inter_area = cv2.contourArea(intersection)
            
            # 计算各自面积
            area1 = cv2.contourArea(poly1)
            area2 = cv2.contourArea(poly2)
            
            # 计算并集面积
            union_area = area1 + area2 - inter_area
            
            if union_area == 0:
                return 0.0
            
            iou = inter_area / union_area
            return float(iou)
        except Exception:
            return 0.0
    
    def is_point_in_lane(self, center_px, polygon_points):
        """判断车辆中心点是否在车道多边形内"""
        if polygon_points is None or len(polygon_points) < 3:
            return False
        try:
            polygon = np.array(polygon_points, dtype=np.float32)
            result = cv2.pointPolygonTest(polygon, tuple(center_px), False)
            return result >= 0
        except Exception:
            return False
    
    def update_lane_statistics(self, vehicle_id, center_px):
        """更新车道流量统计（使用中心点检测）"""
        if not self.lane_detection_enabled:
            return
        
        for lane_id, lane in self.lanes.items():
            polygon_points = lane.get('polygon_points')
            if not polygon_points or len(polygon_points) < 3:
                continue
            
            was_inside = vehicle_id in lane['vehicles_inside']
            is_inside = self.is_point_in_lane(center_px, polygon_points)
            
            if is_inside:
                lane['vehicles_inside'].add(vehicle_id)
                
                # 首次进入车道时计入累计（防止重复计数）
                counted = lane.setdefault('counted_vehicles', set())
                if vehicle_id not in counted:
                    counted.add(vehicle_id)
                    lane['pass_count'] = lane.get('pass_count', 0) + 1
                    # 记录通过时间戳
                    import time
                    lane.setdefault('pass_timestamps', []).append(time.time())
                
                # 记录车道路径
                if not was_inside:
                    self._record_lane_path(vehicle_id, lane['name'])
            else:
                lane['vehicles_inside'].discard(vehicle_id)
            
            # 当前在车道内的车辆数
            lane['flow_count'] = len(lane['vehicles_inside'])
            
            # 排队 = 当前在车道内且静止的车辆数（约1秒=30帧）
            queue_count = 0
            for vid in lane['vehicles_inside']:
                if self.is_vehicle_stationary(vid):
                    queue_count += 1
            lane['queue_count'] = queue_count
            
            # 计算流量率 veh/h（基于最近5分钟的通过数）
            self._calculate_flow_rate(lane_id)
            
            # 更新UI显示
            self._update_lane_statistics_display(lane_id)
    
    def is_vehicle_stationary(self, vehicle_id, min_frames=30, max_displacement=15):
        """
        判断车辆是否静止（连续N帧位移小于阈值）
        
        Args:
            vehicle_id: 车辆ID
            min_frames: 最少需要的连续帧数（30帧≈1秒@30fps）
            max_displacement: N帧内允许的最大总位移（像素）
        """
        # 根据当前模式选择轨迹数据
        if self.current_mode == 'correction':
            tracks = self.correction_vehicle_tracks
        else:
            tracks = self.vehicle_tracks
        
        track = tracks.get(vehicle_id)
        if not track:
            return False
        
        positions = track.get('positions', [])
        if len(positions) < min_frames:
            return False
        
        # 取最近 min_frames 帧的位置
        recent_positions = positions[-min_frames:]
        
        # 计算总位移
        total_displacement = 0
        for i in range(1, len(recent_positions)):
            x1, y1 = recent_positions[i - 1]
            x2, y2 = recent_positions[i]
            displacement = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_displacement += displacement
        
        # 如果总位移小于阈值，认为是静止
        return total_displacement < max_displacement
    
    def _record_lane_path(self, vehicle_id, lane_name):
        """记录车辆经过的车道序列"""
        if vehicle_id not in self.vehicle_lane_path:
            self.vehicle_lane_path[vehicle_id] = []
        
        path = self.vehicle_lane_path[vehicle_id]
        # 避免连续重复记录同一车道
        if not path or path[-1] != lane_name:
            path.append(lane_name)
    
    def _calculate_flow_rate(self, lane_id, window_minutes=5):
        """
        计算流量率 veh/h（基于最近N分钟的通过数换算）
        
        Args:
            lane_id: 车道ID
            window_minutes: 时间窗口（分钟），默认5分钟
        """
        import time
        
        if lane_id not in self.lanes:
            return
        
        lane = self.lanes[lane_id]
        timestamps = lane.get('pass_timestamps', [])
        
        if not timestamps:
            lane['flow_rate'] = 0
            return
        
        current_time = time.time()
        window_seconds = window_minutes * 60
        cutoff_time = current_time - window_seconds
        
        # 清理旧的时间戳，只保留窗口内的
        recent_timestamps = [t for t in timestamps if t >= cutoff_time]
        lane['pass_timestamps'] = recent_timestamps
        
        # 计算窗口内通过的车辆数
        count_in_window = len(recent_timestamps)
        
        # 换算成 veh/h
        lane['flow_rate'] = int(count_in_window * (60 / window_minutes))
    
    def _update_lane_statistics_display(self, lane_id):
        """更新车道统计UI显示"""
        if lane_id not in self.lanes:
            return
        
        lane = self.lanes[lane_id]
        
        # 更新当前车辆数标签
        if 'flow_label' in lane and lane['flow_label']:
            lane['flow_label'].setText(f"当前: {lane['flow_count']}")
        
        # 更新排队标签
        if 'queue_label' in lane and lane['queue_label']:
            lane['queue_label'].setText(f"排队: {lane.get('queue_count', 0)}")
        
        # 更新累计通过数标签
        if 'pass_label' in lane and lane['pass_label']:
            lane['pass_label'].setText(f"累计: {lane.get('pass_count', 0)}")
        
        # 更新流量率标签
        if 'rate_label' in lane and lane['rate_label']:
            lane['rate_label'].setText(f"{lane.get('flow_rate', 0)} veh/h")
    
    def get_vehicle_lane_path(self, vehicle_id):
        """获取车辆经过的车道路径"""
        if vehicle_id not in self.vehicle_lane_path:
            return ""
        
        path = self.vehicle_lane_path[vehicle_id]
        if not path:
            return ""
        
        return " → ".join(path)
    
    def get_lane_events_summary(self, vehicle_id):
        """获取车辆的车道路径摘要"""
        path = self.get_vehicle_lane_path(vehicle_id)
        if path:
            return f" - 路径: {path}"
        return ""
    
    def refresh_current_frame(self):
        """刷新当前显示的帧（重新绘制车道）"""
        try:
            if self.current_mode == 'correction' and hasattr(self, 'correction_current_frame'):
                self.display_correction_frame(self.correction_current_frame)
            elif self.current_mode == 'original' and hasattr(self, 'current_frame') and self.current_frame is not None:
                # 原始模式下重新显示当前帧
                frame = self.current_frame.copy()
                frame = self.draw_lanes_on_frame(frame)
                self.video_label.set_frame(frame)
        except Exception as e:
            self.status_text.append(f"刷新帧失败: {e}")




# ------------------------- 主程序启动 -------------------------
def main():
    app = QApplication(sys.argv)
    window = VehicleTrajectoryAnalyzer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
