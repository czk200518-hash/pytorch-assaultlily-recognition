"""
预测模块

包含:
    - 模型加载: load_model
    - 单图预测: predict_image
    - 批量预测: predict_folder
    - 特征提取: extract_feature
"""

import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .model import create_model
from .dataset_loader import get_transforms


# ============================================================================
# 模型加载
# ============================================================================

def load_model(model_path: str, device: torch.device = None):
    """加载训练好的模型
    
    参数:
        model_path: 模型文件路径
        device: 计算设备（默认自动选择）
    
    返回:
        model: 加载的模型
        class_names: 类别名称列表
        device: 计算设备
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get('class_names', [])

    if not class_names:
        classes_path = model_path.replace('.pth', '_classes.json')
        if os.path.exists(classes_path):
            with open(classes_path, 'r', encoding='utf-8') as f:
                class_names = json.load(f)

    args = checkpoint.get('args', {})
    model_name = args.get('model_name', 'standard')
    input_size = args.get('input_size', 128)
    num_classes = len(class_names)

    if num_classes == 0:
        raise ValueError('无法确定类别数量，请确保模型文件包含 class_names')

    model = create_model(model_name, num_classes, input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, class_names, device


# ============================================================================
# 预测函数
# ============================================================================

def predict_image(model, image_path, class_names, device, input_size=128):
    """对单张图片进行预测
    
    参数:
        model: 神经网络模型
        image_path: 图片路径
        class_names: 类别名称列表
        device: 计算设备
        input_size: 输入图片尺寸
    
    返回:
        dict: 包含预测结果的字典
            - predicted_class: 预测类别
            - confidence: 置信度
            - top3: 前三个预测结果
    """
    transform = get_transforms(input_size, is_train=False)

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence = max_prob.item()

    top3_prob, top3_idx = torch.topk(probabilities, min(3, len(class_names)))
    top3_results = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top3_idx[0], top3_prob[0])
    ]

    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top3': top3_results,
    }


def predict_folder(model, folder_path, class_names, device, input_size=128):
    """对文件夹内所有图片进行预测
    
    参数:
        model: 神经网络模型
        folder_path: 文件夹路径
        class_names: 类别名称列表
        device: 计算设备
        input_size: 输入图片尺寸
    
    返回:
        dict: 文件名到预测结果的映射
    """
    results = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    folder = Path(folder_path)
    for img_path in folder.iterdir():
        if img_path.suffix.lower() in image_extensions:
            result = predict_image(
                model, str(img_path), class_names, device, input_size
            )
            results[img_path.name] = result

    return results


# ============================================================================
# 特征提取
# ============================================================================

def extract_feature(model, image_path, device, input_size=128):
    """提取图片的特征向量
    
    参数:
        model: 神经网络模型
        image_path: 图片路径
        device: 计算设备
        input_size: 输入图片尺寸
    
    返回:
        numpy.ndarray: 特征向量
    """
    transform = get_transforms(input_size, is_train=False)

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.extract_features(image_tensor)

    return features.cpu().numpy().flatten()
