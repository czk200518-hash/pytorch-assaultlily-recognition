"""
训练模块

包含:
    - 损失函数: SupConLoss (监督对比学习损失)
    - 训练函数: train_one_epoch, train_one_epoch_contrastive
    - 验证函数: validate
    - 检查点保存: save_checkpoint
    - 主训练流程: train_model
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from .model import create_model
from .dataset_loader import load_datasets, load_contrastive_datasets


# ============================================================================
# 损失函数
# ============================================================================

class SupConLoss(nn.Module):
    """监督对比学习损失函数
    
    同一类别的样本在特征空间中应该靠近，不同类别的样本应该远离
    
    参数:
        temperature: 温度参数，控制分布的平滑度
        base_temperature: 基础温度参数
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features, labels):
        """
        参数:
            features: [batch_size, feature_dim] 特征向量
            labels: [batch_size] 标签
        
        返回:
            对比学习损失
        """
        device = features.device
        batch_size = features.shape[0]
        
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        mask_self = torch.eye(batch_size, device=device)
        mask = mask - mask_self
        
        exp_sim = torch.exp(similarity_matrix / self.temperature) * (1 - mask_self)
        
        pos_mask = mask.sum(1) > 0
        
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        pos_sim = (exp_sim * mask).sum(1)
        neg_sim = exp_sim.sum(1)
        
        losses = -torch.log(pos_sim[pos_mask] / neg_sim[pos_mask] + 1e-8)
        
        return losses.mean()


# ============================================================================
# 训练函数
# ============================================================================

def train_one_epoch_contrastive(
    model, 
    loader, 
    criterion_cls, 
    criterion_con, 
    optimizer, 
    device, 
    epoch, 
    epochs, 
    log_fn=None, 
    use_amp=False, 
    scaler=None, 
    accum_steps=1, 
    stop_check_fn=None, 
    contrastive_weight: float = 0.3
):
    """带对比学习的训练函数
    
    参数:
        model: 神经网络模型
        loader: 数据加载器
        criterion_cls: 分类损失函数
        criterion_con: 对比学习损失函数
        optimizer: 优化器
        device: 计算设备
        epoch: 当前轮次
        epochs: 总轮次
        log_fn: 日志记录函数
        use_amp: 是否使用自动混合精度
        scaler: AMP梯度缩放器
        accum_steps: 梯度累积步数
        stop_check_fn: 停止检查函数
        contrastive_weight: 对比学习损失权重
    
    返回:
        epoch_loss: 轮次平均损失
        epoch_acc: 轮次准确率
        epoch_cls_loss: 分类损失
        epoch_con_loss: 对比学习损失
    """
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_con_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    skipped_batches = 0
    
    total_data_time = 0.0
    total_transfer_time = 0.0
    total_compute_time = 0.0
    batch_times = []
    log_interval = max(1, len(loader) // 10)
    
    prev_batch_end = time.time()

    for batch_idx, (img_cls, view1, view2, labels) in enumerate(loader):
        data_loaded_time = time.time()
        data_time = data_loaded_time - prev_batch_end
        total_data_time += data_time
        
        if stop_check_fn and stop_check_fn():
            return None, None
            
        if img_cls.size(0) == 1:
            skipped_batches += 1
            prev_batch_end = time.time()
            continue
        
        img_cls = img_cls.to(device)
        view1 = view1.to(device)
        view2 = view2.to(device)
        labels = labels.to(device)
        transfer_time = time.time() - data_loaded_time
        total_transfer_time += transfer_time

        if use_amp:
            with torch.amp.autocast(device.type):
                outputs = model(img_cls)
                cls_loss = criterion_cls(outputs, labels)
                
                if hasattr(model, 'use_contrastive') and model.use_contrastive:
                    _, features1 = model(view1, return_features=True)
                    _, features2 = model(view2, return_features=True)
                    
                    features = torch.cat([features1, features2], dim=0)
                    labels_con = torch.cat([labels, labels], dim=0)
                    con_loss = criterion_con(features, labels_con)
                else:
                    con_loss = torch.tensor(0.0, device=device)
                
                loss = cls_loss + contrastive_weight * con_loss
                loss = loss / accum_steps
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            outputs = model(img_cls)
            cls_loss = criterion_cls(outputs, labels)
            
            if hasattr(model, 'use_contrastive') and model.use_contrastive:
                _, features1 = model(view1, return_features=True)
                _, features2 = model(view2, return_features=True)
                
                features = torch.cat([features1, features2], dim=0)
                labels_con = torch.cat([labels, labels], dim=0)
                con_loss = criterion_con(features, labels_con)
            else:
                con_loss = torch.tensor(0.0, device=device)
            
            loss = cls_loss + contrastive_weight * con_loss
            loss = loss / accum_steps
            loss.backward()

        running_loss += loss.item() * accum_steps
        running_cls_loss += cls_loss.item()
        running_con_loss += con_loss.item() if isinstance(con_loss, torch.Tensor) else 0
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        batch_end_time = time.time()
        compute_time = batch_end_time - data_loaded_time - transfer_time
        total_compute_time += compute_time
        batch_total = batch_end_time - prev_batch_end
        batch_times.append(batch_total)
        prev_batch_end = batch_end_time
        
        if log_fn and (batch_idx + 1) % log_interval == 0:
            if device.type == 'cuda':
                mem_used = torch.cuda.memory_allocated(device) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)
                mem_info = f'显存:{mem_used:.2f}/{mem_reserved:.2f}GB'
            else:
                mem_info = ''
            avg_batch = sum(batch_times[-log_interval:]) / len(batch_times[-log_interval:])
            log_fn(f'  [批次 {batch_idx+1}/{len(loader)}] '
                   f'加载:{data_time:.3f}s + 传输:{transfer_time:.3f}s + '
                   f'计算:{compute_time:.3f}s = {batch_total:.2f}s/批 | {mem_info}')

    if (batch_idx + 1) % accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(loader)
    epoch_cls_loss = running_cls_loss / len(loader)
    epoch_con_loss = running_con_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    if log_fn:
        avg_batch_time = sum(batch_times) / len(batch_times)
        total_time = total_data_time + total_transfer_time + total_compute_time
        log_fn(f'  [诊断] 数据加载: {total_data_time:.1f}s, '
               f'数据传输: {total_transfer_time:.1f}s, '
               f'模型计算: {total_compute_time:.1f}s, '
               f'合计: {total_time:.1f}s')
        log_fn(f'  [诊断] 平均批次: {avg_batch_time:.2f}s')
        if device.type == 'cuda':
            mem_used = torch.cuda.max_memory_allocated(device) / (1024**3)
            log_fn(f'  [诊断] 峰值显存: {mem_used:.2f}GB')
    
    return epoch_loss, epoch_acc, epoch_cls_loss, epoch_con_loss


def train_one_epoch(
    model, 
    loader, 
    criterion, 
    optimizer, 
    device, 
    epoch, 
    epochs,
    log_fn=None, 
    use_amp=False, 
    scaler=None, 
    accum_steps=1, 
    stop_check_fn=None
):
    """标准训练函数（不含对比学习）
    
    参数:
        model: 神经网络模型
        loader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        epoch: 当前轮次
        epochs: 总轮次
        log_fn: 日志记录函数
        use_amp: 是否使用自动混合精度
        scaler: AMP梯度缩放器
        accum_steps: 梯度累积步数
        stop_check_fn: 停止检查函数
    
    返回:
        epoch_loss: 轮次平均损失
        epoch_acc: 轮次准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    skipped_batches = 0
    
    total_data_time = 0.0
    total_transfer_time = 0.0
    total_compute_time = 0.0
    batch_times = []
    log_interval = max(1, len(loader) // 10)
    
    prev_batch_end = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        data_loaded_time = time.time()
        data_time = data_loaded_time - prev_batch_end
        total_data_time += data_time
        
        if stop_check_fn and stop_check_fn():
            return None, None
            
        if images.size(0) == 1:
            skipped_batches += 1
            prev_batch_end = time.time()
            continue
        
        images, labels = images.to(device), labels.to(device)
        transfer_time = time.time() - data_loaded_time
        total_transfer_time += transfer_time

        if use_amp:
            with torch.amp.autocast(device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accum_steps
            loss.backward()

        running_loss += loss.item() * accum_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        batch_end_time = time.time()
        compute_time = batch_end_time - data_loaded_time - transfer_time
        total_compute_time += compute_time
        batch_total = batch_end_time - prev_batch_end
        batch_times.append(batch_total)
        prev_batch_end = batch_end_time
        
        if log_fn and (batch_idx + 1) % log_interval == 0:
            if device.type == 'cuda':
                mem_used = torch.cuda.memory_allocated(device) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)
                mem_info = f'显存:{mem_used:.2f}/{mem_reserved:.2f}GB'
            else:
                mem_info = ''
            avg_batch = sum(batch_times[-log_interval:]) / len(batch_times[-log_interval:])
            log_fn(f'  [批次 {batch_idx+1}/{len(loader)}] '
                   f'加载:{data_time:.3f}s + 传输:{transfer_time:.3f}s + '
                   f'计算:{compute_time:.3f}s = {batch_total:.2f}s/批 | {mem_info}')

    if (batch_idx + 1) % accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    if log_fn:
        avg_batch_time = sum(batch_times) / len(batch_times)
        total_time = total_data_time + total_transfer_time + total_compute_time
        log_fn(f'  [诊断] 数据加载: {total_data_time:.1f}s, '
               f'数据传输: {total_transfer_time:.1f}s, '
               f'模型计算: {total_compute_time:.1f}s, '
               f'合计: {total_time:.1f}s')
        log_fn(f'  [诊断] 平均批次: {avg_batch_time:.2f}s')
        if device.type == 'cuda':
            mem_used = torch.cuda.max_memory_allocated(device) / (1024**3)
            log_fn(f'  [诊断] 峰值显存: {mem_used:.2f}GB')
    
    return epoch_loss, epoch_acc


# ============================================================================
# 验证函数
# ============================================================================

@torch.no_grad()
def validate(model, loader, criterion, device, stop_check_fn=None):
    """验证函数
    
    参数:
        model: 神经网络模型
        loader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        stop_check_fn: 停止检查函数
    
    返回:
        val_loss: 验证损失
        val_acc: 验证准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        if stop_check_fn and stop_check_fn():
            return None, None
            
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


# ============================================================================
# 检查点管理
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, class_names,
                    args, save_path, is_best=False):
    """保存训练检查点
    
    参数:
        model: 神经网络模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前轮次
        best_acc: 最佳准确率
        class_names: 类别名称列表
        args: 训练参数字典
        save_path: 保存路径
        is_best: 是否为最佳模型
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'class_names': class_names,
        'args': args,
    }
    torch.save(checkpoint, save_path)
    
    if is_best:
        orig = Path(save_path)
        model_name = args.get('model_name', 'model')
        total_img = args.get('total_images', '?')
        stem = orig.stem
        parts = stem.split('_')
        date_part = parts[-1] if len(parts) >= 4 else stem
        best_filename = f'{model_name}_{total_img}img_{best_acc:.0f}pct_{date_part}_best.pth'
        best_path = orig.parent / best_filename
        
        for old_best in orig.parent.glob(f'{model_name}_{total_img}img_*pct_*_best.pth'):
            if old_best != best_path:
                old_best.unlink()
        
        torch.save(checkpoint, str(best_path))


# ============================================================================
# 主训练流程
# ============================================================================

def train_model(
    data_dir: str, 
    model_dir: str, 
    model_name: str = 'standard',
    input_size: int = 128, 
    batch_size: int = 32, 
    epochs: int = 50,
    lr: float = 0.001, 
    val_split: float = 0.2,
    num_workers: int = 0, 
    resume: str = None
):
    """训练动漫人脸识别模型
    
    参数:
        data_dir: 数据集目录路径
        model_dir: 模型保存目录路径
        model_name: 模型类型名称
        input_size: 输入图片尺寸
        batch_size: 批量大小
        epochs: 训练轮次
        lr: 学习率
        val_split: 验证集比例
        num_workers: 数据加载线程数
        resume: 恢复训练的检查点路径
    
    返回:
        model: 训练完成的模型
        best_acc: 最佳验证准确率
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_str = 'CUDA (GPU)' if device.type == 'cuda' else 'CPU'
    print(f'设备: {device_str}')
    print(f'数据目录: {data_dir}')
    print(f'模型类型: {model_name}')

    train_loader, val_loader, class_names, num_classes = load_datasets(
        data_dir, input_size=input_size, val_split=val_split,
        batch_size=batch_size, num_workers=num_workers,
    )

    print(f'分类数量: {num_classes}')
    print(f'分类列表: {class_names}')
    print(f'训练集图片数: {len(train_loader.dataset)}')
    print(f'验证集图片数: {len(val_loader.dataset)}')

    if device.type == 'cpu':
        from cpu_optimizer import auto_optimize
        opt_result = auto_optimize()
        print(f'CPU优化: {opt_result["label"]}')

    model = create_model(model_name, num_classes, input_size)
    model = model.to(device)
    if device.type == 'cpu' and opt_result.get('model_transform'):
        model = opt_result['model_transform'](model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    start_epoch = 1
    best_acc = 0.0

    if resume and Path(resume).exists():
        print(f'从检查点恢复: {resume}')
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f'恢复到 Epoch {checkpoint["epoch"]}, 最佳准确率: {best_acc:.2f}%')

    os.makedirs(model_dir, exist_ok=True)
    total_train = len(train_loader.dataset)
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(
        model_dir, f'{model_name}_{total_train}img_{date_str}.pth'
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    train_args = {
        'model_name': model_name,
        'input_size': input_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'val_split': val_split,
        'total_images': total_train,
        'num_classes': num_classes,
    }

    print(f'\n{"="*60}')
    print(f'开始训练: 共 {epochs} 轮, 初始学习率: {lr}')
    print(f'{"="*60}\n')

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - epoch_start

        print(f'Epoch [{epoch}/{epochs}] '
              f'训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}% | '
              f'验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}% | '
              f'学习率: {current_lr:.6f} | 耗时: {elapsed:.1f}秒')

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            print(f'  >>> 新最佳准确率: {best_acc:.2f}% <<<')

        save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
                        class_names, train_args,
                        checkpoint_path, is_best=is_best)

    history_path = checkpoint_path.replace('.pth', '_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    class_names_path = checkpoint_path.replace('.pth', '_classes.json')
    with open(class_names_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print(f'\n{"="*60}')
    print(f'训练完成！最佳验证准确率: {best_acc:.2f}%')
    print(f'模型已保存至: {checkpoint_path}')
    print(f'{"="*60}')

    return model, best_acc
