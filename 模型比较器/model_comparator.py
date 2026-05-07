import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QTextEdit, QGroupBox, QFormLayout, QMessageBox,
    QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QComboBox, QCheckBox,
    QTabWidget, QWidget as QtWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_resnet(model_name, num_classes):
    if model_name == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    elif model_name == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    elif model_name == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    elif model_name == 'resnet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    elif model_name == 'resnet152':
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")


def detect_resnet_config(state_dict):
    has_conv3 = any('conv3' in k for k in state_dict.keys())
    
    layer_counts = []
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        blocks = set()
        for k in state_dict.keys():
            if k.startswith(f'{layer_name}.'):
                parts = k.split('.')
                if len(parts) >= 2:
                    try:
                        blocks.add(int(parts[1]))
                    except ValueError:
                        pass
        layer_counts.append(len(blocks))
    
    return has_conv3, layer_counts


def create_resnet_from_state_dict(state_dict, num_classes):
    has_conv3, layer_counts = detect_resnet_config(state_dict)
    block = Bottleneck if has_conv3 else BasicBlock
    
    layer_configs = {
        (2, 2, 2, 2): ('resnet18', BasicBlock),
        (3, 4, 6, 3): ('resnet34', BasicBlock if not has_conv3 else Bottleneck),
        (3, 4, 23, 3): ('resnet101', Bottleneck),
        (3, 8, 36, 3): ('resnet152', Bottleneck),
    }
    
    config_name = layer_configs.get(tuple(layer_counts), ('custom', block))
    return ResNet(block, layer_counts, num_classes)


def load_model(model_path: str, device: torch.device = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
        raise ValueError('无法确定类别数量')

    state_dict = checkpoint['model_state_dict']
    
    if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        model = create_resnet_from_state_dict(state_dict, num_classes)
    else:
        sys.path.insert(0, str(Path(__file__).parent.parent / '突击莉莉角色识别'))
        from model import create_model
        model = create_model(model_name, num_classes, input_size)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, class_names, device, input_size, model_name


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def load_image_fast(img_path, input_size):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((input_size, input_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img = np.transpose(img, (2, 0, 1))
        return img
    except Exception:
        return None


class CompareThread(QThread):
    progress_signal = pyqtSignal(int, int, str)
    result_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, model1_info, model2_info, dataset_path, batch_size=128):
        super().__init__()
        self.model1_info = model1_info
        self.model2_info = model2_info
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.use_fp16 = torch.cuda.is_available()

    def load_batch_images(self, image_paths, input_size, num_workers=8):
        images = [None] * len(image_paths)
        
        def load_single(idx_path):
            idx, path = idx_path
            return idx, load_image_fast(path, input_size)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(load_single, enumerate(image_paths)))
        
        valid_indices = []
        valid_tensors = []
        for idx, img in results:
            if img is not None:
                valid_indices.append(idx)
                valid_tensors.append(img)
        
        if not valid_tensors:
            return None, []
        
        batch = np.stack(valid_tensors)
        batch_tensor = torch.from_numpy(batch)
        return batch_tensor, valid_indices

    def predict_batch(self, model, batch_tensor, class_names, device):
        batch_tensor = batch_tensor.to(device, non_blocking=True)
        if self.use_fp16:
            batch_tensor = batch_tensor.half()
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            if outputs.dtype == torch.float16:
                outputs = outputs.float()
            probabilities = F.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)
        
        predictions = [class_names[predicted[i].item()] for i in range(len(predicted))]
        confidences = max_probs.cpu().numpy().tolist()
        
        return predictions, confidences

    def run(self):
        try:
            model1, class_names1, device1, input_size1, model_name1 = self.model1_info
            model2, class_names2, device2, input_size2, model_name2 = self.model2_info

            if self.use_fp16:
                model1 = model1.half()
                model2 = model2.half()

            dataset_path = Path(self.dataset_path)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

            all_images = []
            for class_dir in dataset_path.iterdir():
                if class_dir.is_dir():
                    for img_path in class_dir.iterdir():
                        if img_path.suffix.lower() in image_extensions:
                            all_images.append((str(img_path), class_dir.name))

            total = len(all_images)
            if total == 0:
                self.log_signal.emit('[错误] 数据集中没有找到图片')
                return

            self.log_signal.emit(f'[开始] 共 {total} 张图片 | 批量: {self.batch_size} | FP16: {self.use_fp16}')

            results = {
                'model1': {
                    'name': model_name1,
                    'correct': 0,
                    'total': 0,
                    'per_class': defaultdict(lambda: {'correct': 0, 'total': 0}),
                    'errors': [],
                },
                'model2': {
                    'name': model_name2,
                    'correct': 0,
                    'total': 0,
                    'per_class': defaultdict(lambda: {'correct': 0, 'total': 0}),
                    'errors': [],
                },
                'details': [],
            }

            processed = 0
            start_time = datetime.now()
            
            for batch_start in range(0, total, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total)
                batch_items = all_images[batch_start:batch_end]
                batch_paths = [item[0] for item in batch_items]
                batch_labels = [item[1] for item in batch_items]
                
                batch1, valid_idx1 = self.load_batch_images(batch_paths, input_size1)
                batch2, valid_idx2 = self.load_batch_images(batch_paths, input_size2)
                
                if batch1 is None or batch2 is None:
                    continue
                
                preds1, confs1 = self.predict_batch(model1, batch1, class_names1, device1)
                preds2, confs2 = self.predict_batch(model2, batch2, class_names2, device2)
                
                pred_map1 = {valid_idx1[i]: (preds1[i], confs1[i]) for i in range(len(valid_idx1))}
                pred_map2 = {valid_idx2[i]: (preds2[i], confs2[i]) for i in range(len(valid_idx2))}
                
                for local_idx, (img_path, true_label) in enumerate(batch_items):
                    processed += 1
                    
                    if processed % 500 == 0 or processed == total:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        speed = processed / elapsed if elapsed > 0 else 0
                        self.progress_signal.emit(processed, total, f'{speed:.0f} 张/秒')
                    
                    pred1, conf1 = pred_map1.get(local_idx, (None, 0))
                    pred2, conf2 = pred_map2.get(local_idx, (None, 0))
                    
                    if pred1 is None or pred2 is None:
                        continue
                    
                    is_correct1 = (pred1 == true_label)
                    is_correct2 = (pred2 == true_label)

                    results['model1']['total'] += 1
                    results['model2']['total'] += 1
                    results['model1']['per_class'][true_label]['total'] += 1
                    results['model2']['per_class'][true_label]['total'] += 1

                    if is_correct1:
                        results['model1']['correct'] += 1
                        results['model1']['per_class'][true_label]['correct'] += 1
                    else:
                        results['model1']['errors'].append({
                            'image': Path(img_path).name,
                            'true': true_label,
                            'pred': pred1,
                            'conf': conf1,
                        })

                    if is_correct2:
                        results['model2']['correct'] += 1
                        results['model2']['per_class'][true_label]['correct'] += 1
                    else:
                        results['model2']['errors'].append({
                            'image': Path(img_path).name,
                            'true': true_label,
                            'pred': pred2,
                            'conf': conf2,
                        })

                    results['details'].append({
                        'image': Path(img_path).name,
                        'true': true_label,
                        'pred1': pred1,
                        'conf1': conf1,
                        'correct1': is_correct1,
                        'pred2': pred2,
                        'conf2': conf2,
                        'correct2': is_correct2,
                    })

            results['class_names'] = list(set(results['model1']['per_class'].keys()) | set(results['model2']['per_class'].keys()))
            self.result_signal.emit(results)

        except Exception as e:
            self.log_signal.emit(f'[错误] {traceback.format_exc()}')
        finally:
            self.finished_signal.emit()


class ModelComparator(QWidget):
    def __init__(self):
        super().__init__()
        self.model1 = None
        self.model2 = None
        self.model1_info = None
        self.model2_info = None
        self.compare_thread = None
        self.results = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle('模型比较器')
        self.setMinimumSize(1100, 800)
        self.resize(1200, 900)

        layout = QVBoxLayout(self)

        self._setup_model_section(layout)
        self._setup_dataset_section(layout)
        self._setup_progress_section(layout)
        self._setup_results_section(layout)
        self._setup_log_section(layout)

    def _setup_model_section(self, layout):
        models_group = QGroupBox('模型加载')
        models_layout = QVBoxLayout()

        model1_layout = QHBoxLayout()
        model1_layout.addWidget(QLabel('模型A:'))
        self.model1_path_edit = QLineEdit()
        self.model1_path_edit.setPlaceholderText('选择第一个模型文件 (.pth)')
        model1_browse = QPushButton('浏览...')
        model1_browse.clicked.connect(lambda: self.browse_model(1))
        self.model1_load_btn = QPushButton('加载')
        self.model1_load_btn.setStyleSheet('QPushButton { background-color: #2196F3; color: white; border-radius: 4px; padding: 4px 12px; }')
        self.model1_load_btn.clicked.connect(lambda: self.load_model_file(1))
        self.model1_info_label = QLabel('未加载')
        self.model1_info_label.setStyleSheet('color: #888;')
        model1_layout.addWidget(self.model1_path_edit, 1)
        model1_layout.addWidget(model1_browse)
        model1_layout.addWidget(self.model1_load_btn)
        model1_layout.addWidget(self.model1_info_label)

        model2_layout = QHBoxLayout()
        model2_layout.addWidget(QLabel('模型B:'))
        self.model2_path_edit = QLineEdit()
        self.model2_path_edit.setPlaceholderText('选择第二个模型文件 (.pth)')
        model2_browse = QPushButton('浏览...')
        model2_browse.clicked.connect(lambda: self.browse_model(2))
        self.model2_load_btn = QPushButton('加载')
        self.model2_load_btn.setStyleSheet('QPushButton { background-color: #4CAF50; color: white; border-radius: 4px; padding: 4px 12px; }')
        self.model2_load_btn.clicked.connect(lambda: self.load_model_file(2))
        self.model2_info_label = QLabel('未加载')
        self.model2_info_label.setStyleSheet('color: #888;')
        model2_layout.addWidget(self.model2_path_edit, 1)
        model2_layout.addWidget(model2_browse)
        model2_layout.addWidget(self.model2_load_btn)
        model2_layout.addWidget(self.model2_info_label)

        models_layout.addLayout(model1_layout)
        models_layout.addLayout(model2_layout)
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)

    def _setup_dataset_section(self, layout):
        dataset_group = QGroupBox('测试数据集')
        dataset_layout = QHBoxLayout()
        
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText('选择测试数据集目录 (包含分类子文件夹)')
        
        dataset_browse = QPushButton('浏览...')
        dataset_browse.clicked.connect(self.browse_dataset)
        
        self.start_btn = QPushButton('▶ 开始对比测试')
        self.start_btn.setMinimumHeight(36)
        self.start_btn.setStyleSheet(
            'QPushButton { background-color: #FF9800; color: white; font-size: 14px; font-weight: bold; border-radius: 4px; }'
            'QPushButton:hover { background-color: #F57C00; }'
            'QPushButton:disabled { background-color: #cccccc; }'
        )
        self.start_btn.clicked.connect(self.start_comparison)
        
        dataset_layout.addWidget(self.dataset_path_edit, 1)
        dataset_layout.addWidget(dataset_browse)
        dataset_layout.addWidget(self.start_btn)
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

    def _setup_progress_section(self, layout):
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_label = QLabel('就绪')
        progress_layout.addWidget(self.progress_bar, 1)
        progress_layout.addWidget(self.progress_label)
        layout.addLayout(progress_layout)

    def _setup_results_section(self, layout):
        self.result_tabs = QTabWidget()

        summary_widget = QtWidget()
        summary_layout = QVBoxLayout(summary_widget)
        
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(5)
        self.summary_table.setHorizontalHeaderLabels(['指标', '模型A', '模型B', '差异', '胜者'])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        summary_layout.addWidget(self.summary_table)
        
        self.per_class_table = QTableWidget()
        self.per_class_table.setColumnCount(6)
        self.per_class_table.setHorizontalHeaderLabels(['类别', 'A正确数', 'A总数', 'B正确数', 'B总数', '差异'])
        self.per_class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        summary_layout.addWidget(QLabel('各类别详情:'))
        summary_layout.addWidget(self.per_class_table)
        
        self.result_tabs.addTab(summary_widget, '总体对比')

        details_widget = QtWidget()
        details_layout = QVBoxLayout(details_widget)
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(8)
        self.details_table.setHorizontalHeaderLabels(['图片', '真实标签', 'A预测', 'A置信度', 'A正确', 'B预测', 'B置信度', 'B正确'])
        self.details_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        details_layout.addWidget(self.details_table)
        self.result_tabs.addTab(details_widget, '详细结果')

        errors_widget = QtWidget()
        errors_layout = QVBoxLayout(errors_widget)
        
        errors_split = QSplitter(Qt.Horizontal)
        
        errors1_group = QGroupBox('模型A错误')
        errors1_layout = QVBoxLayout(errors1_group)
        self.errors1_table = QTableWidget()
        self.errors1_table.setColumnCount(4)
        self.errors1_table.setHorizontalHeaderLabels(['图片', '真实标签', '预测', '置信度'])
        self.errors1_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        errors1_layout.addWidget(self.errors1_table)
        errors_split.addWidget(errors1_group)

        errors2_group = QGroupBox('模型B错误')
        errors2_layout = QVBoxLayout(errors2_group)
        self.errors2_table = QTableWidget()
        self.errors2_table.setColumnCount(4)
        self.errors2_table.setHorizontalHeaderLabels(['图片', '真实标签', '预测', '置信度'])
        self.errors2_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        errors2_layout.addWidget(self.errors2_table)
        errors_split.addWidget(errors2_group)

        errors_layout.addWidget(errors_split)
        self.result_tabs.addTab(errors_widget, '错误分析')

        layout.addWidget(self.result_tabs, 1)

    def _setup_log_section(self, layout):
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setFont(QFont('Consolas', 9))
        self.log_text.setStyleSheet('QTextEdit { background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #555; }')
        layout.addWidget(self.log_text)

    def browse_model(self, model_num):
        path, _ = QFileDialog.getOpenFileName(
            self, f'选择模型{model_num}文件', '',
            '模型文件 (*.pth);;所有文件 (*)'
        )
        if path:
            if model_num == 1:
                self.model1_path_edit.setText(path)
            else:
                self.model2_path_edit.setText(path)

    def load_model_file(self, model_num):
        if model_num == 1:
            path = self.model1_path_edit.text().strip()
            info_label = self.model1_info_label
        else:
            path = self.model2_path_edit.text().strip()
            info_label = self.model2_info_label

        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, '提示', '请选择有效的 .pth 模型文件')
            return

        try:
            model, class_names, device, input_size, model_name = load_model(path)
            
            if model_num == 1:
                self.model1 = model
                self.model1_info = (model, class_names, device, input_size, model_name)
            else:
                self.model2 = model
                self.model2_info = (model, class_names, device, input_size, model_name)

            total_params = sum(p.numel() for p in model.parameters())
            info_label.setText(f'{model_name} | {input_size}px | {len(class_names)}类 | {total_params/1e6:.1f}M参数')
            info_label.setStyleSheet('color: #4CAF50;')
            self.append_log(f'[加载] 模型{"A" if model_num==1 else "B"}: {os.path.basename(path)} 加载成功')

        except Exception as e:
            QMessageBox.warning(self, '错误', f'加载模型失败: {str(e)}')
            self.append_log(f'[错误] {traceback.format_exc()}')

    def browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, '选择测试数据集目录')
        if path:
            self.dataset_path_edit.setText(path)

    def start_comparison(self):
        if self.model1 is None or self.model2 is None:
            QMessageBox.warning(self, '提示', '请先加载两个模型')
            return

        dataset_path = self.dataset_path_edit.text().strip()
        if not dataset_path or not os.path.isdir(dataset_path):
            QMessageBox.warning(self, '提示', '请选择有效的数据集目录')
            return

        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.append_log('[开始] 启动对比测试...')

        self.compare_thread = CompareThread(
            self.model1_info, self.model2_info, dataset_path, batch_size=256
        )
        self.compare_thread.progress_signal.connect(self.update_progress)
        self.compare_thread.result_signal.connect(self.show_results)
        self.compare_thread.log_signal.connect(self.append_log)
        self.compare_thread.finished_signal.connect(lambda: self.start_btn.setEnabled(True))
        self.compare_thread.start()

    def update_progress(self, current, total, msg):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(f'{current}/{total} - {msg}')

    def show_results(self, results):
        self.results = results
        
        self.summary_table.setRowCount(3)
        
        m1 = results['model1']
        m2 = results['model2']
        
        acc1 = m1['correct'] / m1['total'] * 100 if m1['total'] > 0 else 0
        acc2 = m2['correct'] / m2['total'] * 100 if m2['total'] > 0 else 0
        diff = acc1 - acc2
        
        self.summary_table.setItem(0, 0, QTableWidgetItem('准确率'))
        self.summary_table.setItem(0, 1, QTableWidgetItem(f'{acc1:.2f}%'))
        self.summary_table.setItem(0, 2, QTableWidgetItem(f'{acc2:.2f}%'))
        self.summary_table.setItem(0, 3, QTableWidgetItem(f'{diff:+.2f}%'))
        winner = 'A' if diff > 0 else ('B' if diff < 0 else '平局')
        self.summary_table.setItem(0, 4, QTableWidgetItem(winner))

        self.summary_table.setItem(1, 0, QTableWidgetItem('正确数'))
        self.summary_table.setItem(1, 1, QTableWidgetItem(str(m1['correct'])))
        self.summary_table.setItem(1, 2, QTableWidgetItem(str(m2['correct'])))
        self.summary_table.setItem(1, 3, QTableWidgetItem(f'{m1["correct"]-m2["correct"]:+d}'))
        self.summary_table.setItem(1, 4, QTableWidgetItem('A' if m1['correct'] > m2['correct'] else ('B' if m1['correct'] < m2['correct'] else '平局')))

        self.summary_table.setItem(2, 0, QTableWidgetItem('总测试数'))
        self.summary_table.setItem(2, 1, QTableWidgetItem(str(m1['total'])))
        self.summary_table.setItem(2, 2, QTableWidgetItem(str(m2['total'])))
        self.summary_table.setItem(2, 3, QTableWidgetItem('-'))
        self.summary_table.setItem(2, 4, QTableWidgetItem('-'))

        for row in range(3):
            for col in range(5):
                item = self.summary_table.item(row, col)
                if item:
                    item.setTextAlignment(Qt.AlignCenter)

        class_names = sorted(results['class_names'])
        self.per_class_table.setRowCount(len(class_names))
        
        for i, cls in enumerate(class_names):
            c1 = m1['per_class'].get(cls, {'correct': 0, 'total': 0})
            c2 = m2['per_class'].get(cls, {'correct': 0, 'total': 0})
            
            self.per_class_table.setItem(i, 0, QTableWidgetItem(cls))
            self.per_class_table.setItem(i, 1, QTableWidgetItem(str(c1['correct'])))
            self.per_class_table.setItem(i, 2, QTableWidgetItem(str(c1['total'])))
            self.per_class_table.setItem(i, 3, QTableWidgetItem(str(c2['correct'])))
            self.per_class_table.setItem(i, 4, QTableWidgetItem(str(c2['total'])))
            
            acc1_cls = c1['correct'] / c1['total'] * 100 if c1['total'] > 0 else 0
            acc2_cls = c2['correct'] / c2['total'] * 100 if c2['total'] > 0 else 0
            diff_cls = acc1_cls - acc2_cls
            self.per_class_table.setItem(i, 5, QTableWidgetItem(f'{diff_cls:+.1f}%'))

        self.details_table.setRowCount(len(results['details']))
        for i, detail in enumerate(results['details']):
            self.details_table.setItem(i, 0, QTableWidgetItem(detail['image']))
            self.details_table.setItem(i, 1, QTableWidgetItem(detail['true']))
            self.details_table.setItem(i, 2, QTableWidgetItem(detail['pred1']))
            self.details_table.setItem(i, 3, QTableWidgetItem(f'{detail["conf1"]:.2%}'))
            
            item1 = QTableWidgetItem('✓' if detail['correct1'] else '✗')
            item1.setTextAlignment(Qt.AlignCenter)
            item1.setForeground(QColor('#4CAF50') if detail['correct1'] else QColor('#F44336'))
            self.details_table.setItem(i, 4, item1)
            
            self.details_table.setItem(i, 5, QTableWidgetItem(detail['pred2']))
            self.details_table.setItem(i, 6, QTableWidgetItem(f'{detail["conf2"]:.2%}'))
            
            item2 = QTableWidgetItem('✓' if detail['correct2'] else '✗')
            item2.setTextAlignment(Qt.AlignCenter)
            item2.setForeground(QColor('#4CAF50') if detail['correct2'] else QColor('#F44336'))
            self.details_table.setItem(i, 7, item2)

        self.errors1_table.setRowCount(len(m1['errors']))
        for i, err in enumerate(m1['errors']):
            self.errors1_table.setItem(i, 0, QTableWidgetItem(err['image']))
            self.errors1_table.setItem(i, 1, QTableWidgetItem(err['true']))
            self.errors1_table.setItem(i, 2, QTableWidgetItem(err['pred']))
            self.errors1_table.setItem(i, 3, QTableWidgetItem(f'{err["conf"]:.2%}'))

        self.errors2_table.setRowCount(len(m2['errors']))
        for i, err in enumerate(m2['errors']):
            self.errors2_table.setItem(i, 0, QTableWidgetItem(err['image']))
            self.errors2_table.setItem(i, 1, QTableWidgetItem(err['true']))
            self.errors2_table.setItem(i, 2, QTableWidgetItem(err['pred']))
            self.errors2_table.setItem(i, 3, QTableWidgetItem(f'{err["conf"]:.2%}'))

        self.append_log(f'[完成] 模型A准确率: {acc1:.2f}%, 模型B准确率: {acc2:.2f}%')

    def append_log(self, msg):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f'[{timestamp}] {msg}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ModelComparator()
    window.show()
    sys.exit(app.exec_())
