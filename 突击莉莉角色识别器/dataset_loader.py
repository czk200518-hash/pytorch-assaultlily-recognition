import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_transforms(input_size: int = 128, is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def load_datasets(data_dir: str, input_size: int = 128, val_split: float = 0.2,
                  batch_size: int = 32, num_workers: int = 0):
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
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, class_names, num_classes

def load_test_dataset(data_dir: str, input_size: int = 128,
                      batch_size: int = 1, num_workers: int = 0) -> tuple:
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
