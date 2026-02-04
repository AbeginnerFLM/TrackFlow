#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO处理器
实现YOLOv11目标检测和多目标跟踪功能
"""

import cv2
import numpy as np
from ultralytics import YOLO
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt
import time
import json
from pathlib import Path

# 导入跳帧常量
try:
    from traj_keyong import SKIP_FRAMES
except ImportError:
    # 如果无法导入，使用默认值
    SKIP_FRAMES = 6


class YOLOProcessorThread(QThread):
    """YOLO处理线程"""
    
    # 信号定义
    frame_processed = pyqtSignal(np.ndarray, dict)
    progress_updated = pyqtSignal(int, int, int)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    processing_finished = pyqtSignal()
    detection_info_updated = pyqtSignal(str)
    model_loaded = pyqtSignal(str)
    video_info_updated = pyqtSignal(int, float, int, int)
    
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.video_path = None
        self._connect_signals()
    
    def _connect_signals(self):
        """连接处理器信号到线程信号"""
        self.processor.frame_processed.connect(self.frame_processed, Qt.ConnectionType.QueuedConnection)
        self.processor.progress_updated.connect(self.progress_updated, Qt.ConnectionType.QueuedConnection)
        self.processor.fps_updated.connect(self.fps_updated, Qt.ConnectionType.QueuedConnection)
        self.processor.error_occurred.connect(self.error_occurred, Qt.ConnectionType.QueuedConnection)
        self.processor.processing_finished.connect(self.processing_finished, Qt.ConnectionType.QueuedConnection)
        self.processor.detection_info_updated.connect(self.detection_info_updated, Qt.ConnectionType.QueuedConnection)
        self.processor.model_loaded.connect(self.model_loaded, Qt.ConnectionType.QueuedConnection)
        self.processor.video_info_updated.connect(self.video_info_updated, Qt.ConnectionType.QueuedConnection)
    
    def set_video_path(self, video_path):
        """设置要处理的视频路径"""
        self.video_path = video_path
    
    def run(self):
        """线程运行函数"""
        if self.video_path:
            self.processor.process_video(self.video_path)
    
    def stop_processing(self):
        """停止处理"""
        if self.processor:
            self.processor.is_processing = False


class YOLOProcessor(QObject):
    """YOLO处理器类"""
    
    # 信号定义
    frame_processed = pyqtSignal(np.ndarray, dict)
    progress_updated = pyqtSignal(int, int, int)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    processing_finished = pyqtSignal()
    detection_info_updated = pyqtSignal(str)
    model_loaded = pyqtSignal(str)
    video_info_updated = pyqtSignal(int, float, int, int)
    
    def __init__(self):
        super().__init__()
        self._init_attributes()
    
    def _init_attributes(self):
        """初始化所有属性"""
        # 模型相关
        self.model = None
        self.is_obb_model = False
        
        # 视频处理相关
        self.video_capture = None
        self.is_processing = False
        self.start_frame = 0
        
        # 检测和跟踪设置
        self.detection_enabled = True
        self.tracking_enabled = True
        self.tracker_type = 'bytetrack.yaml'
        self.conf_threshold = 0.1  # 置信度阈值，用于过滤低置信度的误检
        
        # 帧率控制
        self.target_fps = 25
        self.skip_frames = 1
        
        # 性能统计
        self.frame_count = 0
        self.processed_frame_count = 0
        self.expected_processed_frames = 0
        self.start_time = None
        
        # 导出设置
        self.export_options = {'save_txt': False, 'save_conf': False}
        self.output_dir = None
        self.save_images = False
        self.image_dir = None
        
        # 检测结果存储
        self.detection_results = []
        
        # 工作线程
        self.worker_thread = None
    
    # ==================== 配置方法 ====================
    
    def set_target_fps(self, fps):
        """设置目标处理帧率"""
        self.target_fps = max(1, fps)
    
    def set_conf_threshold(self, threshold):
        """设置置信度阈值，用于过滤低置信度的检测结果"""
        self.conf_threshold = max(0.1, min(0.95, threshold))
    
    def set_video_start_frame(self, start_frame):
        """设置视频开始处理的帧号"""
        self.start_frame = start_frame
    
    def set_tracker(self, tracker_name):
        """设置跟踪算法"""
        tracker_map = {
            'ByteTrack': 'bytetrack.yaml',
            'BoT-SORT': 'botsort.yaml'
        }
        self.tracker_type = tracker_map.get(tracker_name, 'bytetrack.yaml')
    
    def set_detection_enabled(self, enabled):
        """设置是否启用检测"""
        self.detection_enabled = enabled
    
    def set_tracking_enabled(self, enabled):
        """设置是否启用跟踪"""
        self.tracking_enabled = enabled
    
    def set_export_options(self, save_txt=False, save_conf=False, output_dir=None, save_images=False, image_dir=None):
        """设置导出选项"""
        self.export_options = {'save_txt': save_txt, 'save_conf': save_conf}
        self.output_dir = output_dir
        self.save_images = save_images
        self.image_dir = image_dir
        
        if save_txt and output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        if save_images and image_dir:
            Path(image_dir).mkdir(parents=True, exist_ok=True)
    
    # ==================== 模型管理 ====================
    
    def load_model(self, model_path=None):
        """加载YOLO模型"""
        try:
            if model_path is None:
                model_path = 'weights/yolo11x-obb.pt'
            
            if not Path(model_path).exists():
                self.detection_info_updated.emit(f"模型文件不存在，请自行下载。")
                return False
            
            self.detection_info_updated.emit(f"正在加载模型: {model_path}")
            self.model = YOLO(model_path)
            
            # 优先通过模型的task属性判断是否为OBB模型，而非仅依赖文件名
            # 这样像 best.pt 这样的自定义OBB模型也能正确识别
            model_task = getattr(self.model, 'task', None)
            if model_task == 'obb':
                self.is_obb_model = True
                self.detection_info_updated.emit(f"✅ 模型加载成功 (OBB模型，通过task属性识别): {model_path}")
            elif 'obb' in model_path.lower():
                self.is_obb_model = True
                self.detection_info_updated.emit(f"✅ 模型加载成功 (OBB模型，通过文件名识别): {model_path}")
            else:
                self.is_obb_model = False
                self.detection_info_updated.emit(f"✅ 模型加载成功 (普通检测模型): {model_path}")
            
            self.model_loaded.emit(model_path)
            return True
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self.detection_info_updated.emit(f"❌ 模型加载失败: {model_path}")
            self.detection_info_updated.emit(f"错误类型: {type(e).__name__}")
            self.detection_info_updated.emit(f"错误信息: {str(e)}")
            self.detection_info_updated.emit(f"详细堆栈:\n{error_detail}")
            self.error_occurred.emit(f"模型加载失败: {str(e)}")
            return False
    
    # ==================== 视频处理 ====================
    
    def process_video(self, video_path):
        """处理视频文件"""
        try:
            if not self._setup_video_processing(video_path):
                return False
            
            self._process_video_frames()
            self.processing_finished.emit()
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"视频处理失败: {str(e)}")
            return False
        finally:
            if self.video_capture:
                self.video_capture.release()
    
    def _setup_video_processing(self, video_path):
        """设置视频处理环境"""
        # 确保模型已加载
        if self.model is None:
            if not self.load_model():
                return False
        
        # 打开视频文件
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            self.error_occurred.emit("无法打开视频文件")
            return False
        
        # 获取视频信息
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        
        # 计算跳帧数量 - 使用配置文件中的常量
        self.skip_frames = SKIP_FRAMES
        
        # 计算实际需要处理的帧数
        self.expected_processed_frames = (total_frames + self.skip_frames - 1) // self.skip_frames
        video_duration = total_frames / original_fps
        
        # 初始化处理状态
        self.is_processing = True
        self.frame_count = 0
        self._init_processing_state()
        self.start_time = time.time()
        self.detection_results.clear()
        
        # 发送视频信息
        self.video_info_updated.emit(total_frames, original_fps, self.skip_frames, self.target_fps)
        
        # 发送初始信息
        self.detection_info_updated.emit(f"视频信息: {total_frames}帧, {original_fps:.1f}FPS, 时长{video_duration:.1f}秒")
        self.detection_info_updated.emit(f"处理设置: 目标{self.target_fps}FPS, 需处理{self.expected_processed_frames}帧, 每{self.skip_frames}帧处理1帧（30fps视频中每{SKIP_FRAMES}帧处理1帧）")
        
        # 如果设置了开始帧，跳转到指定位置
        if self.start_frame > 0:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.frame_count = self.start_frame
            self.detection_info_updated.emit(f"从帧 {self.start_frame} 开始处理")
        
        return True
    
    def _init_processing_state(self):
        """初始化处理状态"""
        if not hasattr(self, 'processed_frame_count') or not hasattr(self, 'start_frame') or self.start_frame == 0:
            self.processed_frame_count = 0
        elif self.start_frame > 0:
            processed_before = (self.start_frame - 1) // self.skip_frames + 1
            self.processed_frame_count = processed_before
            self.detection_info_updated.emit(f"继续处理: 从帧{self.start_frame}开始, 已处理{processed_before}帧")
    
    def _process_video_frames(self):
        """处理视频帧"""
        while self.is_processing:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            # 根据跳帧设置决定是否处理当前帧
            should_process = (self.frame_count % self.skip_frames == 0)
            
            if should_process:
                processed_frame, detection_info = self.process_frame(frame, self.frame_count, should_process)
                self.frame_processed.emit(processed_frame, detection_info)
                
                # 发送检测信息
                count = detection_info.get('count', 0)
                if count > 0:
                    info_text = f"帧 {self.frame_count + 1}: 检测到 {count} 个对象"
                    self.detection_info_updated.emit(info_text)
                
                self.processed_frame_count += 1
            
            # 更新进度
            self._update_progress()
            
            # 更新FPS
            if self.processed_frame_count > 0 and self.processed_frame_count % 10 == 0:
                elapsed_time = time.time() - self.start_time
                current_fps = self.processed_frame_count / elapsed_time
                self.fps_updated.emit(current_fps)
            
            self.frame_count += 1
    
    def _update_progress(self):
        """更新处理进度"""
        if self.expected_processed_frames > 0:
            actual_processed = min(self.processed_frame_count, self.expected_processed_frames)
            progress = int((actual_processed / self.expected_processed_frames) * 100)
            progress = min(progress, 100)
        else:
            actual_processed = self.processed_frame_count
            progress = 0
        
        self.progress_updated.emit(actual_processed, self.expected_processed_frames, progress)
    
    def process_frame(self, frame, frame_index, should_process=True):
        """处理单帧"""
        processed_frame = frame.copy()
        detection_info = {
            'frame_id': frame_index,
            'objects': [],
            'count': 0
        }
        
        try:
            if self.detection_enabled and self.model is not None:
                # 进行目标检测，使用置信度阈值过滤低置信度结果
                if self.tracking_enabled:
                    results = self.model.track(frame, tracker=self.tracker_type, persist=True, conf=self.conf_threshold)
                else:
                    results = self.model(frame, conf=self.conf_threshold)
                
                # 处理检测结果
                if results and len(results) > 0:
                    result = results[0]
                    boxes, confidences, class_ids, track_ids = self._extract_detection_data(result)
                    
                    if boxes is not None and len(boxes) > 0:
                        self._draw_detections(processed_frame, boxes, confidences, class_ids, track_ids, detection_info)
            
            # 保存检测结果
            self.detection_results.append(detection_info)
            
            # 保存标签和图像 - 只保存实际处理的帧
            if should_process:
                if self.export_options['save_txt'] and self.output_dir and detection_info['count'] > 0:
                    self.save_labels_to_txt(frame_index, detection_info, frame.shape)
                
                if self.save_images and self.image_dir:
                    self.save_frame_image(frame_index, frame)
            
        except Exception as e:
            print(f"处理帧 {frame_index} 时出错: {e}")
        
        return processed_frame, detection_info
    
    def _extract_detection_data(self, result):
        """从检测结果中提取数据"""
        if self.is_obb_model and hasattr(result, 'obb') and result.obb is not None:
            # OBB模型处理
            boxes = result.obb.xyxyxyxy.cpu().numpy()
            confidences = result.obb.conf.cpu().numpy()
            class_ids = result.obb.cls.cpu().numpy().astype(int)
            track_ids = None
            if self.tracking_enabled and hasattr(result.obb, 'id') and result.obb.id is not None:
                track_ids = result.obb.id.cpu().numpy().astype(int)
        elif result.boxes is not None:
            # 普通模型处理
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            track_ids = None
            if self.tracking_enabled and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            boxes = None
            confidences = None
            class_ids = None
            track_ids = None
        
        return boxes, confidences, class_ids, track_ids
    
    def _draw_detections(self, processed_frame, boxes, confidences, class_ids, track_ids, detection_info):
        """绘制检测结果"""
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            # 获取类别名称
            class_name = self.model.names[class_id] if class_id < len(self.model.names) else f"Class_{class_id}"
            
            # 获取跟踪ID
            track_id = track_ids[i] if track_ids is not None and i < len(track_ids) else None
            
            # 获取颜色
            color = self.get_color_for_class(class_id)
            
            if self.is_obb_model:
                # OBB模型：绘制旋转边界框
                points = box.reshape(-1, 2).astype(int)
                cv2.polylines(processed_frame, [points], True, color, 2)
                
                # 获取边界框用于标签位置
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                
                # 保存检测信息（OBB格式）
                obj_info = {
                    'bbox': box.tolist(),
                    'bbox_type': 'obb',
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'track_id': int(track_id) if track_id is not None else None
                }
            else:
                # 普通模型：绘制矩形边界框
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                
                # 保存检测信息（普通格式）
                obj_info = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'bbox_type': 'xyxy',
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'track_id': int(track_id) if track_id is not None else None
                }
            
            # 绘制标签
            self._draw_label(processed_frame, x1, y1, class_name, conf, track_id, color)
            detection_info['objects'].append(obj_info)
        
        detection_info['count'] = len(boxes)
    
    def _draw_label(self, frame, x1, y1, class_name, conf, track_id, color):
        """绘制标签"""
        label = f"{class_name}: {conf:.2f}"
        if track_id is not None:
            label = f"ID:{track_id} {label}"
        
        # 绘制标签背景
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # 绘制标签文本
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def get_color_for_class(self, class_id):
        """为不同类别生成不同颜色"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (0, 128, 0)
        ]
        return colors[class_id % len(colors)]
    
    # ==================== 文件保存 ====================
    
    def save_labels_to_txt(self, frame_index, detection_info, frame_shape):
        """保存标签到txt文件（YOLO格式）"""
        try:
            if not self.output_dir:
                return
            
            txt_filename = f"frame_{frame_index:06d}.txt"
            txt_path = Path(self.output_dir) / txt_filename
            height, width = frame_shape[:2]
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                for obj in detection_info['objects']:
                    self._write_object_to_txt(f, obj, width, height)
                    
        except Exception as e:
            self.error_occurred.emit(f"保存标签文件失败: {str(e)}")
    
    def _write_object_to_txt(self, file, obj, width, height):
        """将单个对象写入txt文件"""
        class_id = obj['class_id']
        confidence = obj['confidence']
        bbox = obj['bbox']
        bbox_type = obj['bbox_type']
        track_id = obj.get('track_id', -1)
        
        if bbox_type == 'obb':
            # OBB格式：8个点的坐标 -> 归一化
            points = np.array(bbox).reshape(-1, 2)
            norm_points = points.copy()
            norm_points[:, 0] /= width
            norm_points[:, 1] /= height
            
            line_parts = [str(class_id)]
            for point in norm_points:
                line_parts.extend([f"{point[0]:.6f}", f"{point[1]:.6f}"])
            
            if self.export_options['save_conf']:
                line_parts.append(f"{confidence:.6f}")
                if track_id is not None:
                    line_parts.append(str(track_id))
        else:
            # 普通边界框格式：xyxy -> 中心点+宽高格式
            x1, y1, x2, y2 = bbox
            center_x = ((x1 + x2) / 2.0) / width
            center_y = ((y1 + y2) / 2.0) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            
            line_parts = [
                str(class_id), f"{center_x:.6f}", f"{center_y:.6f}",
                f"{box_width:.6f}", f"{box_height:.6f}"
            ]
            
            if self.export_options['save_conf']:
                line_parts.append(f"{confidence:.6f}")
                if track_id is not None:
                    line_parts.append(str(track_id))
        
        file.write(' '.join(line_parts) + '\n')
    
    def save_frame_image(self, frame_index, frame):
        """保存原始帧图像"""
        try:
            if not self.image_dir:
                return False
            
            img_filename = f"frame_{frame_index:06d}.jpg"
            img_path = Path(self.image_dir) / img_filename
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"保存图像失败: {str(e)}")
            return False
    
    # ==================== 线程管理 ====================
    
    def stop_processing(self):
        """停止处理"""
        self.is_processing = False
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
    
    def start_processing_thread(self, video_path):
        """在线程中启动处理"""
        if self.worker_thread and self.worker_thread.isRunning():
            return False
        
        self.worker_thread = YOLOProcessorThread(self)
        self.worker_thread.set_video_path(video_path)
        self.worker_thread.start()
        return True
    
    # ==================== 数据导出 ====================
    
    def export_results(self, output_path, format='json'):
        """导出检测结果"""
        try:
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.detection_results, f, ensure_ascii=False, indent=2)
            elif format.lower() == 'csv':
                self._export_to_csv(output_path)
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"导出结果失败: {str(e)}")
            return False
    
    def _export_to_csv(self, output_path):
        """导出为CSV格式"""
        import pandas as pd
        
        rows = []
        for frame_data in self.detection_results:
            frame_id = frame_data['frame_id']
            for obj in frame_data['objects']:
                row = {
                    'frame_id': frame_id,
                    'object_id': obj['track_id'],
                    'class_id': obj['class_id'],
                    'class_name': obj['class_name'],
                    'confidence': obj['confidence'],
                    'bbox_x1': obj['bbox'][0],
                    'bbox_y1': obj['bbox'][1],
                    'bbox_x2': obj['bbox'][2],
                    'bbox_y2': obj['bbox'][3]
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    def get_detection_summary(self):
        """获取检测摘要信息"""
        if not self.detection_results:
            return {}
        
        total_detections = sum(frame['count'] for frame in self.detection_results)
        total_frames = len(self.detection_results)
        
        # 统计各类别数量
        class_counts = {}
        for frame_data in self.detection_results:
            for obj in frame_data['objects']:
                class_name = obj['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_frames': total_frames,
            'total_detections': total_detections,
            'average_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
            'class_counts': class_counts
        }