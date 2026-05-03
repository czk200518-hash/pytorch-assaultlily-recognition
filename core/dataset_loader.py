"""
数据集加载模块

包含:
    - 数据变换: get_transforms, get_contrastive_transforms, get_local_region_transforms
    - 变换类: ContrastiveTransform, LocalRegionTransform
    - 数据集类: ContrastiveDataset, CombinedDataset, CombinedContrastiveLocalDataset, LocalRegionDataset
    - 加载函数: load_datasets, load_test_dataset, load_contrastive_datasets, 
                load_combined_contrastive_local_datasets, load_local_region_datasets
"""

import os
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


# ============================================================================
# 数据变换
# ============================================================================

def get_transforms(input_size: int = 128, is_train: bool = True):
    """获取数据增强/预处理变换
    
    参数:
        input_size: 输入图片尺寸
        is_train: 是否为训练模式（训练模式会应用数据增强）
    
    返回:
        transforms.Compose: 变换管道
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class LocalRegionTransform:
    """局部区域数据增强 - 随机裁剪 + 五裁剪 + 混合训练
    
    增强策略:
        - 45%: 整图 + 标准增强
        - 40%: 随机裁剪 (方案A) - 裁剪50%-90%区域
        - 15%: 五裁剪 (方案B) - 左上/右上/左下/右下/中心
    
    参数:
        input_size: 输出图片尺寸
        random_crop_prob: 随机裁剪概率 (方案A)
        five_crop_prob: 五裁剪概率 (方案B)
    """
    
    def __init__(self, input_size: int = 128, 
                 random_crop_prob: float = 0.4,
                 five_crop_prob: float = 0.15):
        self.input_size = input_size
        self.random_crop_prob = random_crop_prob
        self.five_crop_prob = five_crop_prob
        
        self.to_tensor = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        self.color_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        ])
    
    def _random_crop_region(self, img, scale_range=(0.5, 0.9)):
        """随机裁剪图片的 50%-90% 区域"""
        w, h = img.size
        area = w * h
        
        target_area = random.uniform(*scale_range) * area
        aspect_ratio = random.uniform(0.7, 1.3)
        
        crop_h = int((target_area / aspect_ratio) ** 0.5)
        crop_w = int(crop_h * aspect_ratio)
        
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        
        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)
        
        return img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
    
    def _five_crop_region(self, img):
        """五裁剪: 左上、右上、左下、右下、中心"""
        w, h = img.size
        crop_w = int(w * 0.65)
        crop_h = int(h * 0.65)
        
        positions = [
            (0, 0),                  # 左上
            (w - crop_w, 0),         # 右上
            (0, h - crop_h),         # 左下
            (w - crop_w, h - crop_h),# 右下
            ((w - crop_w) // 2, (h - crop_h) // 2),  # 中心
        ]
        
        pos = random.choice(positions)
        x1, y1 = pos
        return img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
    
    def __call__(self, img):
        rand_val = random.random()
        
        if rand_val < self.random_crop_prob:
            img = self._random_crop_region(img)
        elif rand_val < self.random_crop_prob + self.five_crop_prob:
            img = self._five_crop_region(img)
        
        img = self.color_augment(img)
        img = self.to_tensor(img)
        
        return img


def get_local_region_transforms(input_size: int = 128, 
                                 random_crop_prob: float = 0.4,
                                 five_crop_prob: float = 0.15):
    """获取局部区域数据增强变换
    
    参数:
        input_size: 输出图片尺寸
        random_crop_prob: 随机裁剪概率
        five_crop_prob: 五裁剪概率
    
    返回:
        LocalRegionTransform: 局部区域变换对象
    """
    return LocalRegionTransform(input_size, random_crop_prob, five_crop_prob)


class ContrastiveTransform:
    """对比学习变换：对同一张图片生成两个不同的增强版本
    
    参数:
        input_size: 输出图片尺寸
    """
    
    def __init__(self, input_size):
        self.input_size = input_size
        
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)


def get_contrastive_transforms(input_size: int = 128):
    """获取对比学习数据增强变换 - 返回两个不同的增强版本
    
    参数:
        input_size: 输出图片尺寸
    
    返回:
        ContrastiveTransform: 对比学习变换对象
    """
    return ContrastiveTransform(input_size)


# ============================================================================
# 数据集类
# ============================================================================

class ContrastiveDataset(Dataset):
    """对比学习数据集 - 返回同一图片的两个增强版本
    
    参数:
        data_dir: 数据集目录路径
        input_size: 输入图片尺寸
    """
    
    def __init__(self, data_dir: str, input_size: int = 128):
        self.data_path = Path(data_dir)
        self.transform = get_contrastive_transforms(input_size)
        
        self.dataset = datasets.ImageFolder(root=str(self.data_path))
        self.samples = self.dataset.samples
        self.targets = self.dataset.targets
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        
        view1, view2 = self.transform(img)
        
        return view1, view2, label


class CombinedDataset(Dataset):
    """组合数据集 - 同时支持分类和对比学习
    
    参数:
        samples: 样本路径列表
        targets: 标签列表
        input_size: 输入图片尺寸
    """
    
    def __init__(self, samples, targets, input_size: int = 128):
        self.samples = samples
        self.targets = targets
        self.input_size = input_size
        
        self.classify_transform = get_transforms(input_size, is_train=True)
        self.contrastive_transform = get_contrastive_transforms(input_size)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        
        img_for_classify = self.classify_transform(img)
        view1, view2 = self.contrastive_transform(img)
        
        return img_for_classify, view1, view2, label


class CombinedContrastiveLocalDataset(Dataset):
    """组合数据集 - 同时支持对比学习和局部区域增强
    
    数据增强策略:
        - view1: 整图 + 标准对比增强 (完整角色特征)
        - view2: 局部裁剪 + 标准增强 (局部特征)
        - img_cls: 整图 + 局部增强混合 (分类训练)
    
    这样模型会学习:
        1. 同一角色的整图和局部特征应该相似 (对比学习)
        2. 即使只看到局部也能识别角色 (局部增强)
    
    参数:
        samples: 样本路径列表
        targets: 标签列表
        input_size: 输入图片尺寸
        random_crop_prob: 随机裁剪概率
        five_crop_prob: 五裁剪概率
    """
    
    def __init__(self, samples, targets, input_size: int = 128,
                 random_crop_prob: float = 0.4, five_crop_prob: float = 0.15):
        self.samples = samples
        self.targets = targets
        self.input_size = input_size
        
        self.classify_transform = LocalRegionTransform(
            input_size, random_crop_prob, five_crop_prob
        )
        
        self.view1_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
        ])
        
        self.local_crop = LocalRegionTransform(
            input_size, random_crop_prob=1.0, five_crop_prob=0.0
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        
        img_for_classify = self.classify_transform(img)
        view1 = self.view1_transform(img)
        view2 = self.local_crop(img)
        
        return img_for_classify, view1, view2, label


class LocalRegionDataset(Dataset):
    """支持局部区域增强的数据集
    
    参数:
        samples: 样本路径列表
        targets: 标签列表
        input_size: 输入图片尺寸
        random_crop_prob: 随机裁剪概率
        five_crop_prob: 五裁剪概率
    """
    
    def __init__(self, samples, targets, input_size: int = 128,
                 random_crop_prob: float = 0.4, five_crop_prob: float = 0.15):
        self.samples = samples
        self.targets = targets
        self.transform = LocalRegionTransform(
            input_size, random_crop_prob, five_crop_prob
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label


# ============================================================================
# 数据加载函数
# ============================================================================

def load_datasets(data_dir: str, input_size: int = 128, val_split: float = 0.2,
                  batch_size: int = 32, num_workers: int = 0):
    """加载数据集并划分训练集和验证集

    数据集目录结构要求:
        data_dir/
        ├── 角色A/        ← 子文件夹名即类别标签
        │   ├── 001.jpg
        │   └── 002.jpg
        ├── 角色B/
        │   └── 001.jpg
        └── ...
    
    参数:
        data_dir: 数据集目录路径
        input_size: 输入图片尺寸
        val_split: 验证集比例
        batch_size: 批量大小
        num_workers: 数据加载线程数
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        class_names: 类别名称列表
        num_classes: 类别数量
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(
            f"在 {data_dir} 中没有找到任何子文件夹。\n\n"
            "数据集目录中需要按角色名称建立子文件夹来存放各自的图片，例如:\n"
            "  dataset/\n"
            "    ├── 鸣人/\n"
            "    │   ├── 001.jpg\n"
            "    │   └── 002.jpg\n"
            "    ├── 佐助/\n"
            "    │   └── 001.jpg\n"
            "    └── ...\n\n"
            "如果你已经有一个角色文件夹（如 一柳梨璃/），\n"
            "请选择它的**上级目录**（如 M分类/）作为数据集目录。"
        )

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    dirs_with_images = 0
    for d in subdirs:
        if any(f.suffix.lower() in image_extensions for f in d.iterdir()):
            dirs_with_images += 1

    if dirs_with_images == 0:
        raise FileNotFoundError(
            f"在子文件夹中没有找到任何图片文件。\n"
            f"共找到 {len(subdirs)} 个子文件夹，请确保每个角色子文件夹中都有图片。"
        )

    full_dataset = datasets.ImageFolder(
        root=str(data_path),
        transform=get_transforms(input_size, is_train=True),
    )

    class_names = full_dataset.classes
    num_classes = len(class_names)

    if num_classes < 2:
        raise ValueError(
            f"只找到了 {num_classes} 个分类，至少需要 2 个分类才能训练。\n"
            f"请确保数据集目录中有多个角色子文件夹。"
        )

    if len(full_dataset) < 10:
        print(f"⚠ 警告: 数据集只有 {len(full_dataset)} 张图片，建议至少 50 张以上")

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"数据集太小 (共 {len(full_dataset)} 张, 验证集比例 {val_split})，"
            f"训练集或验证集为空。请添加更多图片或降低验证集比例。"
        )

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    val_dataset.dataset.transform = get_transforms(input_size, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, class_names, num_classes


def load_test_dataset(data_dir: str, input_size: int = 128,
                      batch_size: int = 1, num_workers: int = 0) -> tuple:
    """加载完整的测试数据集（不做随机划分）
    
    参数:
        data_dir: 数据集目录路径
        input_size: 输入图片尺寸
        batch_size: 批量大小
        num_workers: 数据加载线程数
    
    返回:
        loader: 数据加载器
        classes: 类别名称列表
        num_classes: 类别数量
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    dataset = datasets.ImageFolder(
        root=str(data_path),
        transform=get_transforms(input_size, is_train=False),
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return loader, dataset.classes, len(dataset.classes)


def load_contrastive_datasets(data_dir: str, input_size: int = 128, val_split: float = 0.2,
                               batch_size: int = 32, num_workers: int = 0):
    """加载对比学习数据集
    
    参数:
        data_dir: 数据集目录路径
        input_size: 输入图片尺寸
        val_split: 验证集比例
        batch_size: 批量大小
        num_workers: 数据加载线程数
    
    返回:
        train_loader: 训练数据加载器 (返回: img_classify, view1, view2, label)
        val_loader: 验证数据加载器 (返回: img, label)
        class_names: 类别名称列表
        num_classes: 类别数量
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    full_dataset = datasets.ImageFolder(root=str(data_path))
    class_names = full_dataset.classes
    num_classes = len(class_names)

    if num_classes < 2:
        raise ValueError(f"只找到了 {num_classes} 个分类，至少需要 2 个分类才能训练。")

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size], generator=generator
    )

    train_samples = [full_dataset.samples[i] for i in train_indices.indices]
    train_targets = [full_dataset.targets[i] for i in train_indices.indices]
    
    train_combined = CombinedDataset(train_samples, train_targets, input_size)
    train_combined.classes = class_names
    train_combined.class_to_idx = full_dataset.class_to_idx
    
    val_dataset = datasets.ImageFolder(
        root=str(data_path),
        transform=get_transforms(input_size, is_train=False),
    )
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)

    train_loader = DataLoader(
        train_combined, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, class_names, num_classes


def load_combined_contrastive_local_datasets(data_dir: str, input_size: int = 128, 
                                              val_split: float = 0.2,
                                              batch_size: int = 32, num_workers: int = 0,
                                              random_crop_prob: float = 0.4, 
                                              five_crop_prob: float = 0.15):
    """加载组合数据集 - 同时支持对比学习和局部区域增强
    
    数据增强策略:
        - view1: 整图 + 标准对比增强 (完整角色特征)
        - view2: 局部裁剪 + 标准增强 (局部特征)
        - img_cls: 整图 + 局部增强混合 (分类训练)
    
    参数:
        data_dir: 数据集目录路径
        input_size: 输入图片尺寸
        val_split: 验证集比例
        batch_size: 批量大小
        num_workers: 数据加载线程数
        random_crop_prob: 随机裁剪概率
        five_crop_prob: 五裁剪概率
    
    返回:
        train_loader: 训练数据加载器 (返回: img_classify, view1, view2, label)
        val_loader: 验证数据加载器 (返回: img, label)
        class_names: 类别名称列表
        num_classes: 类别数量
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    full_dataset = datasets.ImageFolder(root=str(data_path))
    class_names = full_dataset.classes
    num_classes = len(class_names)

    if num_classes < 2:
        raise ValueError(f"只找到了 {num_classes} 个分类，至少需要 2 个分类才能训练。")

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size], generator=generator
    )

    train_samples = [full_dataset.samples[i] for i in train_indices.indices]
    train_targets = [full_dataset.targets[i] for i in train_indices.indices]
    
    train_dataset = CombinedContrastiveLocalDataset(
        train_samples, train_targets, input_size,
        random_crop_prob, five_crop_prob
    )
    train_dataset.classes = class_names
    train_dataset.class_to_idx = full_dataset.class_to_idx
    
    val_dataset = datasets.ImageFolder(
        root=str(data_path),
        transform=get_transforms(input_size, is_train=False),
    )
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, class_names, num_classes


def load_local_region_datasets(data_dir: str, input_size: int = 128, val_split: float = 0.2,
                                batch_size: int = 32, num_workers: int = 0,
                                random_crop_prob: float = 0.4, five_crop_prob: float = 0.15):
    """加载支持局部区域增强的数据集
    
    参数:
        data_dir: 数据集目录路径
        input_size: 输入图片尺寸
        val_split: 验证集比例
        batch_size: 批量大小
        num_workers: 数据加载线程数
        random_crop_prob: 随机裁剪概率 (方案A)
        five_crop_prob: 五裁剪概率 (方案B)
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        class_names: 类别名称列表
        num_classes: 类别数量
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    full_dataset = datasets.ImageFolder(root=str(data_path))
    class_names = full_dataset.classes
    num_classes = len(class_names)

    if num_classes < 2:
        raise ValueError(f"只找到了 {num_classes} 个分类，至少需要 2 个分类才能训练。")

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size], generator=generator
    )

    train_samples = [full_dataset.samples[i] for i in train_indices.indices]
    train_targets = [full_dataset.targets[i] for i in train_indices.indices]
    
    train_dataset = LocalRegionDataset(
        train_samples, train_targets, input_size,
        random_crop_prob, five_crop_prob
    )
    train_dataset.classes = class_names
    train_dataset.class_to_idx = full_dataset.class_to_idx
    
    val_dataset = datasets.ImageFolder(
        root=str(data_path),
        transform=get_transforms(input_size, is_train=False),
    )
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, class_names, num_classes
