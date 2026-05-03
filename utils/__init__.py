"""
工具模块

提供硬件检测、CPU优化、GPU优化和人脸检测功能

导出:
    - 人脸检测: detect_faces, crop_faces, draw_boxes, batch_detect_folder
    - CPU优化: cpu_auto_optimize
    - GPU优化: setup_gpu_optimizations, format_memory_info
    - 硬件检测: get_hardware_info
"""

from .face_detector import detect_faces, crop_faces, draw_boxes, batch_detect_folder
from .cpu_optimizer import auto_optimize as cpu_auto_optimize
from .gpu_optimizer import setup_gpu_optimizations, format_memory_info
from .hardware_detector import get_hardware_info
