"""
核心模块

提供模型定义、训练、数据加载和预测功能

导出:
    - 模型: create_model
    - 训练: train_model, train_one_epoch_contrastive, SupConLoss
    - 数据加载: load_datasets, load_contrastive_datasets, 
                load_local_region_datasets, load_combined_contrastive_local_datasets, get_transforms
    - 预测: load_model, predict_image, predict_folder
"""

from .model import create_model
from .train import train_model, train_one_epoch_contrastive, SupConLoss
from .dataset_loader import (
    load_datasets,
    load_contrastive_datasets,
    load_local_region_datasets,
    load_combined_contrastive_local_datasets,
    get_transforms,
)
from .predict import load_model, predict_image, predict_folder
