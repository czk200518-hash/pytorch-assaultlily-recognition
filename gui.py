import sys
import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QTextEdit, QProgressBar, QGroupBox, QFormLayout, QMessageBox,
    QSplitter, QScrollArea, QFrame, QGridLayout, QSizePolicy, QCheckBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer, QTranslator, QLibraryInfo, QLocale
from PyQt5.QtGui import QFont, QDesktopServices, QPixmap, QImage


BASE_DIR = Path(__file__).parent
VENV_PYTHON = str(BASE_DIR.parent.parent / '.venv' / 'Scripts' / 'python.exe')
TITLE = '动漫人脸识别工具'


def _init_matplotlib():
    """延迟初始化 matplotlib"""
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    return FigureCanvas, Figure


class HardwareDetectThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(object, object)

    def __init__(self, data_dir: str, num_classes: int = 86):
        super().__init__()
        self.data_dir = data_dir
        self.num_classes = num_classes

    def run(self):
        try:
            from utils.hardware_detector import run_full_benchmark
            
            hw_info, optimal_config = run_full_benchmark(
                self.data_dir,
                self.num_classes,
                log_fn=self.log_signal.emit,
                progress_fn=self.progress_signal.emit,
            )
            
            self.finished_signal.emit(hw_info, optimal_config)
            
        except Exception as e:
            self.log_signal.emit(f'[错误] 硬件检测失败: {traceback.format_exc()}')
            self.finished_signal.emit(None, None)


class TrainingThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(bool, str)
    time_signal = pyqtSignal(str)
    patience_signal = pyqtSignal(int, float, float)
    plot_data_signal = pyqtSignal(int, float, float, float, float, float, float)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop_requested = False
        self._continue_requested = False

    def request_stop(self):
        self._stop_requested = True

    def request_continue(self):
        self._continue_requested = True

    def run(self):
        try:
            import time
            import torch as torch_module
            from core.train import train_one_epoch, validate, save_checkpoint
            from core.model import create_model

            device_config = self.config.get('device', 'auto')
            if device_config == 'auto':
                device = torch_module.device('cuda' if torch_module.cuda.is_available() else 'cpu')
            elif device_config == 'cpu':
                device = torch_module.device('cpu')
            else:
                device = torch_module.device(device_config)
            
            device_str = 'CUDA (GPU)' if device.type == 'cuda' else 'CPU'
            if device.type == 'cuda':
                device_str = f'CUDA:{device.index or 0} ({torch_module.cuda.get_device_name(device)})'
            self.log_signal.emit(f'[设备] {device_str}')

            use_amp = self.config.get('use_amp', False)
            if use_amp:
                self.log_signal.emit('[AMP] 混合精度训练已启用')
            scaler = torch_module.amp.GradScaler(device.type, enabled=use_amp) if use_amp and device.type == 'cuda' else None

            data_dir = self.config['data_dir']
            model_dir = self.config['model_dir']
            model_name = self.config['model_name']
            input_size = self.config['input_size']
            batch_size = self.config['batch_size']
            epochs = self.config['epochs']
            lr = self.config['lr']
            val_split = self.config['val_split']
            num_workers = self.config.get('num_workers', 0)
            val_freq = self.config.get('val_freq', 1)
            resume_path = self.config.get('resume', None)
            accum_steps = self.config.get('accum_steps', 1)
            skip_gpu_adjust = self.config.get('skip_gpu_adjust', False)
            original_batch_size = batch_size

            if device.type == 'cuda':
                from utils.gpu_optimizer import setup_gpu_optimizations, auto_adjust_for_gpu, format_memory_info
                gpu_opt = setup_gpu_optimizations()
                if gpu_opt['success']:
                    self.log_signal.emit(f'[GPU优化] {gpu_opt["message"]}')
                    if gpu_opt.get('low_vram'):
                        self.log_signal.emit('[GPU优化] 检测到低显存GPU，已启用优化模式')
                
                if not skip_gpu_adjust:
                    batch_size, input_size, accum_steps, adjust_msg = auto_adjust_for_gpu(
                        batch_size, input_size, model_name, 10, log_fn=self.log_signal.emit
                    )
                else:
                    self.log_signal.emit('[GPU优化] 跳过自动调整 (使用硬件检测配置)')
                
                if accum_steps > 1:
                    self.log_signal.emit(f'[GPU优化] 梯度累积: {accum_steps}步 (等效批量: {batch_size * accum_steps})')
                
                self.log_signal.emit(f'[GPU信息]\n{format_memory_info()}')

            use_contrastive = self.config.get('use_contrastive', False)
            use_local_region = self.config.get('use_local_region', False)
            
            self.log_signal.emit('[数据] 正在加载数据集...')
            if use_contrastive and use_local_region:
                from core.dataset_loader import load_combined_contrastive_local_datasets
                train_loader, val_loader, class_names, num_classes = load_combined_contrastive_local_datasets(
                    data_dir, input_size=input_size, val_split=val_split,
                    batch_size=batch_size, num_workers=num_workers,
                )
                self.log_signal.emit('[组合模式] 已启用对比学习 + 局部区域增强')
                self.log_signal.emit('[组合模式] view1: 整图增强, view2: 局部裁剪')
                self.log_signal.emit('[组合模式] 分类: 随机裁剪 40%, 五裁剪 15%, 整图 45%')
            elif use_contrastive:
                from core.dataset_loader import load_contrastive_datasets
                train_loader, val_loader, class_names, num_classes = load_contrastive_datasets(
                    data_dir, input_size=input_size, val_split=val_split,
                    batch_size=batch_size, num_workers=num_workers,
                )
                self.log_signal.emit('[对比学习] 已启用对比学习模式')
            elif use_local_region:
                from core.dataset_loader import load_local_region_datasets
                train_loader, val_loader, class_names, num_classes = load_local_region_datasets(
                    data_dir, input_size=input_size, val_split=val_split,
                    batch_size=batch_size, num_workers=num_workers,
                )
                self.log_signal.emit('[局部增强] 已启用局部区域增强模式')
                self.log_signal.emit('[局部增强] 随机裁剪: 40%, 五裁剪: 15%, 整图 45%')
            else:
                train_loader, val_loader, class_names, num_classes = load_datasets(
                    data_dir, input_size=input_size, val_split=val_split,
                    batch_size=batch_size, num_workers=num_workers,
                )
            
            actual_batch_size = train_loader.batch_size
            if actual_batch_size != original_batch_size:
                self.log_signal.emit(f'[配置] 批量大小: {original_batch_size} → {actual_batch_size} (实际)')
            else:
                self.log_signal.emit(f'[配置] 批量大小: {actual_batch_size}')
            self.log_signal.emit(f'[配置] 等效批量: {actual_batch_size * accum_steps} (含梯度累积)')
            
            self.log_signal.emit(f'[数据] 分类数: {num_classes}, 训练集: {len(train_loader.dataset)} 张, 验证集: {len(val_loader.dataset)} 张')
            self.log_signal.emit(f'[数据] 分类列表: {class_names}')

            total_train = len(train_loader.dataset)

            if device.type == 'cpu':
                from cpu_optimizer import auto_optimize
                opt_result = auto_optimize(log_fn=self.log_signal.emit)
                self.log_signal.emit(f'[优化] {opt_result["label"]}')
            else:
                opt_result = {'model_transform': None}

            model = create_model(model_name, num_classes, input_size, use_contrastive=use_contrastive).to(device)
            if opt_result.get('model_transform'):
                model = opt_result['model_transform'](model)
            
            if use_contrastive:
                self.log_signal.emit('[优化] 使用对比学习训练模式')
            else:
                self.log_signal.emit('[优化] 使用标准训练模式')
            
            criterion = torch_module.nn.CrossEntropyLoss()
            if use_contrastive:
                from core.train import SupConLoss
                criterion_con = SupConLoss(temperature=0.07)
            optimizer = torch_module.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch_module.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5
            )

            start_epoch = 1
            best_acc = 0.0
            loaded_from_resume = False

            if resume_path and os.path.isfile(resume_path):
                self.log_signal.emit(f'[继续] 正在加载模型: {os.path.basename(resume_path)}')
                checkpoint = torch_module.load(resume_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_acc = checkpoint.get('best_acc', 0.0)
                loaded_from_resume = True
                self.log_signal.emit(f'[继续] 已从第 {checkpoint.get("epoch", 0)} 轮继续, 历史最佳: {best_acc:.2f}%')

            os.makedirs(model_dir, exist_ok=True)
            model_type_dir = os.path.join(model_dir, model_name)
            os.makedirs(model_type_dir, exist_ok=True)
            date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_name = f'{model_name}_{total_train}img_{date_str}'
            session_dir = os.path.join(model_type_dir, session_name)
            os.makedirs(session_dir, exist_ok=True)
            checkpoint_path = os.path.join(session_dir, f'{session_name}.pth')

            train_args = {
                'model_name': model_name, 'input_size': input_size,
                'batch_size': batch_size, 'epochs': epochs, 'lr': lr,
                'val_split': val_split, 'total_images': total_train,
                'num_classes': num_classes, 'use_amp': use_amp,
                'accum_steps': accum_steps,
                'use_contrastive': use_contrastive,
                'use_local_region': use_local_region,
                'augment_random_crop': 0.4 if use_local_region else 0.0,
                'augment_five_crop': 0.15 if use_local_region else 0.0,
                'num_workers': num_workers,
                'device': device_config,
            }

            epochs_no_improve = 0
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

            patience = self.config.get('patience', -1)

            if use_contrastive and use_local_region:
                self.log_signal.emit('[训练模式] 对比学习 + 局部区域增强 (组合模式)')
            elif use_contrastive:
                self.log_signal.emit('[训练模式] 对比学习')
            elif use_local_region:
                self.log_signal.emit('[训练模式] 局部区域增强')
            else:
                self.log_signal.emit('[训练模式] 标准训练')

            if loaded_from_resume:
                self.log_signal.emit(f'[训练] 继续训练, 从第 {start_epoch} 轮开始, 共 {epochs} 轮')
            else:
                self.log_signal.emit(f'[训练] 开始训练, 共 {epochs} 轮, 初始学习率: {lr}')
            self.log_signal.emit('─' * 60)

            train_start_time = time.time()
            epoch_times = []
            total_samples_processed = 0
            total_random_crops = 0
            total_five_crops = 0
            total_whole_images = 0
            total_flips = 0
            total_rotations = 0
            total_color_jitters = 0
            total_erases = 0

            for epoch in range(start_epoch, epochs + 1):
                if self._stop_requested:
                    self.log_signal.emit('[中断] 用户请求停止训练')
                    break

                epoch_start = time.time()
                self.progress_signal.emit(epoch, epochs)

                if use_contrastive:
                    from core.train import train_one_epoch_contrastive
                    train_result = train_one_epoch_contrastive(
                        model, train_loader, criterion, criterion_con, optimizer, device, epoch, epochs,
                        log_fn=self.log_signal.emit, use_amp=use_amp, scaler=scaler,
                        accum_steps=accum_steps, stop_check_fn=lambda: self._stop_requested,
                        contrastive_weight=0.3,
                    )
                    if train_result[0] is None:
                        self.log_signal.emit('[中断] 用户请求停止训练')
                        break
                    train_loss, train_acc, cls_loss, con_loss = train_result
                else:
                    train_loss, train_acc = train_one_epoch(
                        model, train_loader, criterion, optimizer, device, epoch, epochs,
                        log_fn=self.log_signal.emit, use_amp=use_amp, scaler=scaler,
                        accum_steps=accum_steps, stop_check_fn=lambda: self._stop_requested,
                    )
                    if train_loss is None:
                        self.log_signal.emit('[中断] 用户请求停止训练')
                        break
                
                samples_this_epoch = len(train_loader.dataset)
                total_samples_processed += samples_this_epoch
                
                if use_local_region:
                    random_crop_prob = 0.4
                    five_crop_prob = 0.15
                    whole_prob = 1.0 - random_crop_prob - five_crop_prob
                    
                    total_random_crops += int(samples_this_epoch * random_crop_prob)
                    total_five_crops += int(samples_this_epoch * five_crop_prob)
                    total_whole_images += samples_this_epoch - int(samples_this_epoch * random_crop_prob) - int(samples_this_epoch * five_crop_prob)
                else:
                    total_whole_images += samples_this_epoch
                
                total_flips += int(samples_this_epoch * 0.5)
                total_rotations += samples_this_epoch
                total_color_jitters += samples_this_epoch
                erase_prob = 0.5 if use_local_region else 0.3
                total_erases += int(samples_this_epoch * erase_prob)
                
                if epoch % val_freq == 0 or epoch == epochs:
                    val_loss, val_acc = validate(model, val_loader, criterion, device,
                                                 stop_check_fn=lambda: self._stop_requested)
                    if val_loss is None:
                        self.log_signal.emit('[中断] 用户请求停止训练')
                        break
                    
                    scheduler.step(val_acc)
                    is_best = val_acc > best_acc
                    if is_best:
                        best_acc = val_acc
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                else:
                    val_loss, val_acc = history['val_loss'][-1] if history['val_loss'] else 0, history['val_acc'][-1] if history['val_acc'] else 0
                    is_best = False
                    
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                current_lr = optimizer.param_groups[0]['lr']

                if patience > 0 and epochs_no_improve >= patience:
                    self.patience_signal.emit(epoch, best_acc, val_acc)
                    while not self._stop_requested and not self._continue_requested:
                        self.msleep(200)
                    if self._stop_requested:
                        break
                    if self._continue_requested:
                        self._continue_requested = False
                        epochs_no_improve = 0
                        self.log_signal.emit('[继续] 用户选择继续训练, 耐心计数已重置')

                epoch_elapsed = time.time() - epoch_start
                epoch_times.append(epoch_elapsed)
                total_elapsed = time.time() - train_start_time

                eta_str = self._compute_eta(epoch, epochs, epoch_times, total_elapsed)
                self.time_signal.emit(eta_str)
                self.plot_data_signal.emit(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, total_elapsed)

                log_line = (
                    f'第{epoch}/{epochs} 轮|'
                    f'训练损失:{train_loss:.4f}|训练准确率:{train_acc:.2f}%|'
                    f'验证损失:{val_loss:.4f}|验证准确率:{val_acc:.2f}%|'
                    f'学习率:{current_lr:.6f}|'
                    f'耗时:{epoch_elapsed:.1f}秒' +
                    ('★最佳' if is_best else '')
                )
                self.log_signal.emit(log_line)
                
                stats_line = f'  [累计] 样本:{total_samples_processed} | 翻转:~{total_flips} | 旋转:{total_rotations} | 抖动:{total_color_jitters} | 擦除:~{total_erases}'
                self.log_signal.emit(stats_line)
                if use_local_region:
                    crop_line = f'  [裁剪] 随机裁剪:{total_random_crops} | 五裁剪:{total_five_crops} | 整图:{total_whole_images}'
                    self.log_signal.emit(crop_line)

                save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
                                class_names, train_args, checkpoint_path, is_best=is_best)

            total_time = time.time() - train_start_time
            self.log_signal.emit('─' * 60)
            self.log_signal.emit(f'[完成] 训练结束, 总耗时: {total_time/60:.1f} 分钟')
            self.log_signal.emit(f'[完成] 最佳验证准确率: {best_acc:.2f}%')
            
            trained_epochs = len(history['train_loss'])
            self._log_training_statistics(train_loader, trained_epochs, use_contrastive, use_local_region)

            for f in Path(session_dir).iterdir():
                if f.suffix == '.pth' and '_best' not in f.stem and f.name != os.path.basename(checkpoint_path):
                    f.unlink()
                    self.log_signal.emit(f'[清理] 已删除中间模型: {f.name}')

            for f in Path(session_dir).iterdir():
                if f.suffix == '.json':
                    f.unlink()

            remaining = [f.name for f in Path(session_dir).iterdir() if f.suffix == '.pth']
            self.log_signal.emit(f'[完成] 模型已保存至: {session_dir}')
            self.log_signal.emit(f'[完成] 保留文件: {", ".join(remaining)}')

            self.finished_signal.emit(True, checkpoint_path)

        except Exception as e:
            self.log_signal.emit(f'[错误] {traceback.format_exc()}')
            self.finished_signal.emit(False, str(e))

    @staticmethod
    def _compute_eta(current_epoch, total_epochs, epoch_times, total_elapsed):
        if len(epoch_times) >= 3:
            avg_time = sum(epoch_times[-3:]) / len(epoch_times[-3:])
        elif len(epoch_times) > 0:
            avg_time = sum(epoch_times) / len(epoch_times)
        else:
            avg_time = 0

        remaining_epochs = total_epochs - current_epoch
        eta_seconds = avg_time * remaining_epochs

        elapsed_str = _format_duration(total_elapsed)
        if avg_time > 0:
            eta_str = _format_duration(eta_seconds)
            return f'已用: {elapsed_str}   预计剩余: {eta_str}'
        else:
            return f'已用: {elapsed_str}   数据不足, 3轮后补正预测'

    def _log_training_statistics(self, train_loader, epochs, use_contrastive, use_local_region):
        """记录训练统计信息总结"""
        try:
            total_samples = len(train_loader.dataset)
            total_batches = len(train_loader)
            total_processed = total_samples * epochs
            
            self.log_signal.emit('')
            self.log_signal.emit('[统计] 训练总结:')
            self.log_signal.emit(f'  • 样本总数: {total_samples} 张 | 批次数: {total_batches} | 轮数: {epochs}')
            self.log_signal.emit(f'  • 累计处理: {total_processed} 张次')
            self.log_signal.emit('')
            
        except Exception as e:
            self.log_signal.emit(f'[统计] 统计信息计算失败: {e}')


def _format_duration(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f'{h:d}:{m:02d}:{s:02d}'


class TrainingPlotCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        FigureCanvas, Figure = _init_matplotlib()
        self.FigureCanvas = FigureCanvas
        self.Figure = Figure
        self.history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'time': [],
        }
        self.total_epochs = 100
        self.current_plot = 'acc'
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        switch_layout = QHBoxLayout()
        switch_layout.addWidget(QLabel('图表类型:'))

        self.plot_combo = QComboBox()
        self.plot_combo.addItems([
            '轮数-准确率',
            '轮数-损失',
            '轮数-学习率',
            '时间-验证准确率',
            '时间-训练准确率',
            '时间-损失',
        ])
        self.plot_combo.currentIndexChanged.connect(self.on_plot_changed)
        switch_layout.addWidget(self.plot_combo)
        switch_layout.addStretch()

        layout.addLayout(switch_layout)

        self.figure = self.Figure(figsize=(8, 3.5), dpi=100)
        self.figure.patch.set_facecolor('#2b2b2b')
        self.canvas = self.FigureCanvas(self.figure)
        self.ax = None
        self.canvas.setMinimumHeight(280)
        layout.addWidget(self.canvas)

        self.update_plot()

    def on_plot_changed(self, index):
        plot_types = ['acc', 'loss', 'lr', 'time_val_acc', 'time_train_acc', 'time_loss']
        self.current_plot = plot_types[index]
        self.update_plot()

    def set_total_epochs(self, total_epochs):
        self.total_epochs = total_epochs

    def add_data(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, elapsed_time):
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
        self.history['time'].append(elapsed_time)
        self.update_plot()

    def clear(self):
        for key in self.history:
            self.history[key] = []
        self.update_plot()

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')

        if len(self.history['epochs']) == 0:
            ax.text(0.5, 0.5, '等待训练数据...', ha='center', va='center',
                    fontsize=12, color='#888888', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#555555')
            self.canvas.draw()
            return

        epochs = self.history['epochs']
        times = self.history['time']

        if self.current_plot == 'acc':
            ax.plot(epochs, self.history['train_acc'], 'b-', label='训练准确率', linewidth=1.5)
            ax.plot(epochs, self.history['val_acc'], 'r-', label='验证准确率', linewidth=1.5)
            ax.set_xlabel('轮数', color='#cccccc')
            ax.set_ylabel('准确率 (%)', color='#cccccc')
            ax.set_title('轮数-准确率曲线', color='#cccccc', fontsize=11)
            ax.set_xlim(0, self.total_epochs)
            ax.set_xticks(range(0, self.total_epochs + 1, max(1, self.total_epochs // 10)))
        elif self.current_plot == 'loss':
            ax.plot(epochs, self.history['train_loss'], 'b-', label='训练损失', linewidth=1.5)
            ax.plot(epochs, self.history['val_loss'], 'r-', label='验证损失', linewidth=1.5)
            ax.set_xlabel('轮数', color='#cccccc')
            ax.set_ylabel('损失', color='#cccccc')
            ax.set_title('轮数-损失曲线', color='#cccccc', fontsize=11)
            ax.set_xlim(0, self.total_epochs)
            ax.set_xticks(range(0, self.total_epochs + 1, max(1, self.total_epochs // 10)))
        elif self.current_plot == 'lr':
            ax.plot(epochs, self.history['lr'], 'g-', label='学习率', linewidth=1.5)
            ax.set_xlabel('轮数', color='#cccccc')
            ax.set_ylabel('学习率', color='#cccccc')
            ax.set_title('轮数-学习率曲线', color='#cccccc', fontsize=11)
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
            ax.set_xlim(0, self.total_epochs)
            ax.set_xticks(range(0, self.total_epochs + 1, max(1, self.total_epochs // 10)))
        elif self.current_plot == 'time_val_acc':
            ax.plot(times, self.history['val_acc'], 'r-', label='验证准确率', linewidth=1.5)
            ax.set_xlabel('时间 (秒)', color='#cccccc')
            ax.set_ylabel('验证准确率 (%)', color='#cccccc')
            ax.set_title('时间-验证准确率曲线', color='#cccccc', fontsize=11)
            ax.set_xlim(0, None)
        elif self.current_plot == 'time_train_acc':
            ax.plot(times, self.history['train_acc'], 'b-', label='训练准确率', linewidth=1.5)
            ax.set_xlabel('时间 (秒)', color='#cccccc')
            ax.set_ylabel('训练准确率 (%)', color='#cccccc')
            ax.set_title('时间-训练准确率曲线', color='#cccccc', fontsize=11)
            ax.set_xlim(0, None)
        elif self.current_plot == 'time_loss':
            ax.plot(times, self.history['train_loss'], 'b-', label='训练损失', linewidth=1.5)
            ax.plot(times, self.history['val_loss'], 'r-', label='验证损失', linewidth=1.5)
            ax.set_xlabel('时间 (秒)', color='#cccccc')
            ax.set_ylabel('损失', color='#cccccc')
            ax.set_title('时间-损失曲线', color='#cccccc', fontsize=11)
            ax.set_xlim(0, None)

        ax.tick_params(colors='#cccccc')
        ax.legend(loc='best', facecolor='#3b3b3b', edgecolor='#555555', labelcolor='#cccccc')
        for spine in ax.spines.values():
            spine.set_color('#555555')
        ax.grid(True, alpha=0.3, color='#555555')

        self.figure.tight_layout()
        self.canvas.draw()


class TrainTab(QWidget):
    CONFIG_FILE = BASE_DIR / 'config.json'
    
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self._log_lines = []
        self.setup_ui()
        self._load_config()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        top_splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_panel.setMinimumWidth(437)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        config_group = QGroupBox('训练配置')
        form = QFormLayout()

        data_layout = QHBoxLayout()
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setPlaceholderText('选择包含角色子文件夹的数据集目录')
        data_browse = QPushButton('浏览...')
        data_browse.clicked.connect(self.browse_data_dir)
        self.recommend_btn = QPushButton('💡 智能推荐参数')
        self.recommend_btn.setStyleSheet(
            'QPushButton { background-color: #FF9800; color: white; border-radius: 3px; padding: 4px 10px; }'
            'QPushButton:hover { background-color: #F57C00; }'
        )
        self.recommend_btn.clicked.connect(self.show_recommendations)
        data_layout.addWidget(self.data_dir_edit)
        data_layout.addWidget(data_browse)
        data_layout.addWidget(self.recommend_btn)
        form.addRow('数据集目录:', data_layout)

        model_layout = QHBoxLayout()
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setText(str(BASE_DIR / 'models'))
        model_browse = QPushButton('浏览...')
        model_browse.clicked.connect(self.browse_model_dir)
        model_layout.addWidget(self.model_dir_edit)
        model_layout.addWidget(model_browse)
        form.addRow('模型保存目录:', model_layout)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            'tiny', 'small', 'standard', 'large', 'mobilenet',
            'standard_se', 'standard_cbam',
            'resnet18', 'resnet34', 'resnet50',
            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2'
        ])
        self.model_combo.setToolTip(
            'tiny:  超轻量 ~500K, 极速, <50张总图\n'
            'small: 轻量 ~3M, <100张/类, 防过拟合\n'
            'standard: 标准 ~10M, 通用之选 (默认)\n'
            'large: 大型 ~20M, >2000张总图, 最高精度\n'
            'mobilenet: MobileNetV2风格 ~2M, CPU高效\n'
            '────────── 注意力模型 ──────────\n'
            'standard_se: 标准模型 + SE注意力\n'
            'standard_cbam: 标准模型 + CBAM注意力 (推荐)\n'
            '────────── 预训练模型 (迁移学习) ──────────\n'
            'resnet18: ResNet18 ~11M, 迁移学习推荐\n'
            'resnet34: ResNet34 ~21M, 更强特征\n'
            'resnet50: ResNet50 ~25M, 最强特征提取\n'
            'efficientnet_b0: EfficientNet-B0 ~5M, 高效\n'
            'efficientnet_b1: EfficientNet-B1 ~8M, 平衡\n'
            'efficientnet_b2: EfficientNet-B2 ~9M, 更高精度'
        )
        form.addRow('模型类型:', self.model_combo)
        
        self.contrastive_check = QCheckBox('启用对比学习')
        self.contrastive_check.setToolTip(
            '对比学习可以更好地区分相似角色\n'
            '同一角色的特征会被拉近，不同角色的特征会被推远\n'
            '可以与局部区域增强同时使用，效果更佳\n'
            '注意: 会增加约30%的训练时间和显存占用'
        )
        form.addRow('', self.contrastive_check)
        
        self.local_region_check = QCheckBox('启用局部区域增强')
        self.local_region_check.setToolTip(
            '提高模型对图片局部区域的识别能力\n'
            '训练时随机裁剪图片的局部区域\n'
            '适用于: 需要识别头发、服装等局部特征\n'
            '可以与对比学习同时使用，效果更佳\n'
            '注意: 会略微增加训练时间'
        )
        form.addRow('', self.local_region_check)

        self.input_size_spin = QSpinBox()
        self.input_size_spin.setRange(32, 512)
        self.input_size_spin.setValue(128)
        self.input_size_spin.setSingleStep(32)
        self.input_size_spin.setToolTip(
            '图片送入模型前的缩放尺寸。必须与预测时一致。\n\n'
            '推荐值:\n'
            '  64~96px: 速度优先, 细节丢失较多\n'
            '  128px:  平衡之选(默认), 动漫人脸细节足够\n'
            '  224~256px: 精度优先, 适合高分辨率原图\n\n'
            '注意: 尺寸越大, 显存/内存占用越大, 训练越慢'
        )
        form.addRow('输入尺寸:', self.input_size_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(32)
        self.batch_size_spin.setToolTip(
            '每批送入模型训练的图片数量。\n\n'
            '推荐范围(按数据总量):\n'
            '  <100张:  4~8\n'
            '  100~500张:  8~16\n'
            '  500~2000张: 16~32\n'
            '  >2000张: 32~64\n\n'
            '小批量: 训练慢但梯度噪声有助于泛化\n'
            '大批量: 训练快但需更多显存/内存'
        )
        form.addRow('批量大小:', self.batch_size_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setToolTip(
            '完整遍历数据集的次数。\n\n'
            '推荐范围(按数据总量):\n'
            '  <50张:  80~120轮\n'
            '  50~200张:  100~150轮\n'
            '  200~1000张: 120~200轮\n'
            '  >1000张: 80~150轮\n\n'
            '程序自动保存验证准确率最高的模型, 轮数多设无害\n'
            'ReduceLROnPlateau会在准确率停滞时自动降低学习率'
        )
        form.addRow('训练轮数:', self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setToolTip(
            '控制模型参数每次更新的步长, 训练最关键参数。\n\n'
            '推荐值:\n'
            '  0.001(默认): 通用值, 适合大多数场景\n'
            '  0.0005: 训练不稳定或数据量很少时\n'
            '  0.0001: 微调已有模型时\n\n'
            '太大(>0.01): 损失震荡, 无法收敛\n'
            '太小(<0.0001): 收敛极慢\n'
            '程序会自动减半学习率(连续5轮验证准确率不涨时)'
        )
        form.addRow('学习率:', self.lr_spin)

        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.05, 0.5)
        self.val_split_spin.setValue(0.2)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setToolTip(
            '从数据集中随机抽出不参与训练的比例, 用于评估模型效果。\n\n'
            '推荐值(按数据总量):\n'
            '  <100张:  0.1~0.15(尽量多留给训练)\n'
            '  100~500张: 0.15~0.2\n'
            '  500~2000张: 0.2\n'
            '  >2000张: 0.2~0.3\n\n'
            '验证集数据模型从未见过, 其准确率能反映真实泛化能力\n'
            '训练准确率高但验证准确率低 = 过拟合'
        )
        form.addRow('验证集比例:', self.val_split_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(-1, 100)
        self.patience_spin.setValue(-1)
        self.patience_spin.setSpecialValueText('不启用')
        self.patience_spin.setToolTip(
            '连续多少轮验证准确率没有提升则弹窗提示。\n'
            ' -1: 不启用(默认)\n'
            ' 5:  5轮无提升弹出提示\n'
            ' 10: 10轮无提升弹出提示\n\n'
            '提示后用户可选择继续训练或终止。\n'
            '与 ReduceLROnPlateau 不同, 这是由用户决策的早停机制。'
        )
        form.addRow('早停耐心值:', self.patience_spin)

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 16)
        self.num_workers_spin.setValue(0)
        self.num_workers_spin.setToolTip(
            '数据加载线程数。\n'
            ' 0: 单线程加载 (Windows推荐)\n'
            ' 2-4: 大数据集可尝试，但Windows可能变慢\n\n'
            '如果训练慢，先尝试设为0'
        )
        form.addRow('数据加载线程:', self.num_workers_spin)

        self.val_freq_spin = QSpinBox()
        self.val_freq_spin.setRange(1, 10)
        self.val_freq_spin.setValue(1)
        self.val_freq_spin.setToolTip(
            '每多少轮进行一次验证。\n'
            ' 1: 每轮都验证 (默认)\n'
            ' 2-3: 加快训练速度\n'
            ' 5: 大数据集推荐\n\n'
            '减少验证频率可加快训练'
        )
        form.addRow('验证频率:', self.val_freq_spin)

        device_layout = QHBoxLayout()
        self.device_combo = QComboBox()
        self._update_device_combo()
        self.device_combo.setToolTip(
            '选择训练使用的设备:\n'
            '自动: 优先使用GPU, 不可用时回退CPU\n'
            'CPU: 强制使用CPU训练\n'
            'CUDA: 强制使用GPU (需要NVIDIA显卡和CUDA环境)'
        )
        refresh_device_btn = QPushButton('🔄')
        refresh_device_btn.setFixedSize(28, 28)
        refresh_device_btn.setToolTip('刷新设备列表')
        refresh_device_btn.clicked.connect(self._update_device_combo)
        device_layout.addWidget(self.device_combo)
        device_layout.addWidget(refresh_device_btn)
        device_layout.addStretch()
        form.addRow('训练设备:', device_layout)

        self.amp_check = QCheckBox('启用混合精度训练 (AMP)')
        self.amp_check.setChecked(False)
        self.amp_check.setToolTip(
            '混合精度训练 (Automatic Mixed Precision):\n'
            ' - 使用FP16进行计算, FP32存储权重\n'
            ' - CPU和GPU均可使用\n'
            ' - 可加速训练30-50%, 减少内存占用\n'
            ' - 对模型精度影响极小\n\n'
            '推荐: 始终启用'
        )
        form.addRow('', self.amp_check)

        resume_layout = QHBoxLayout()
        self.resume_edit = QLineEdit()
        self.resume_edit.setPlaceholderText('选择已有的模型文件继续训练（可选）')
        self.resume_edit.setReadOnly(True)
        resume_browse = QPushButton('选择模型...')
        resume_browse.clicked.connect(self.browse_resume_model)
        resume_clear = QPushButton('清除')
        resume_clear.clicked.connect(self.clear_resume_model)
        resume_layout.addWidget(self.resume_edit)
        resume_layout.addWidget(resume_browse)
        resume_layout.addWidget(resume_clear)
        form.addRow('继续训练:', resume_layout)

        config_group.setLayout(form)
        left_layout.addWidget(config_group)

        info_layout = QHBoxLayout()
        self.hint_btn = QPushButton('💡 数据集要求')
        self.hint_btn.clicked.connect(self.show_data_hint)
        self.ref_btn = QPushButton('📋 参数速查表')
        self.ref_btn.clicked.connect(self.show_ref_table)
        self.hw_detect_btn = QPushButton('🔧 硬件检测')
        self.hw_detect_btn.setStyleSheet(
            'QPushButton { background-color: #2196F3; color: white; border-radius: 3px; padding: 4px 10px; }'
            'QPushButton:hover { background-color: #1976D2; }'
        )
        self.hw_detect_btn.setToolTip('自动检测硬件并测试最优训练配置')
        self.hw_detect_btn.clicked.connect(self.start_hardware_detection)
        info_layout.addWidget(self.hint_btn)
        info_layout.addWidget(self.ref_btn)
        info_layout.addWidget(self.hw_detect_btn)
        info_layout.addStretch()
        left_layout.addLayout(info_layout)

        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton('▶ 开始训练')
        self.train_btn.setMinimumHeight(36)
        self.train_btn.setStyleSheet(
            'QPushButton { background-color: #4CAF50; color: white; font-size: 14px; font-weight: bold; border-radius: 4px; }'
            'QPushButton:hover { background-color: #45a049; }'
            'QPushButton:disabled { background-color: #cccccc; }'
        )
        self.train_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton('■ 停止训练')
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        self.open_dir_btn = QPushButton('📂 打开模型目录')
        self.open_dir_btn.clicked.connect(self.open_model_folder)
        btn_layout.addWidget(self.train_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.open_dir_btn)
        left_layout.addLayout(btn_layout)

        top_splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        plot_group = QGroupBox('训练曲线')
        plot_layout = QVBoxLayout(plot_group)
        self.plot_canvas = TrainingPlotCanvas()
        plot_layout.addWidget(self.plot_canvas)
        right_layout.addWidget(plot_group)

        top_splitter.addWidget(right_panel)
        top_splitter.setSizes([437, 943])
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(top_splitter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(20)
        main_layout.addWidget(self.progress_bar)

        self.time_label = QLabel('已用: 0:00:00   预计剩余: —')
        self.time_label.setFont(QFont('Consolas', 11))
        self.time_label.setStyleSheet('color: #FF9800; font-weight: bold; padding: 2px 8px;')
        self.time_label.setMinimumWidth(300)
        main_layout.addWidget(self.time_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont('Consolas', 9))
        self.log_text.setStyleSheet('QTextEdit { background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #555; }')
        main_layout.addWidget(self.log_text)

        self._train_start_time = None
        self._elapsed_timer = QTimer()
        self._elapsed_timer.timeout.connect(self._update_elapsed)

    def browse_data_dir(self):
        path = QFileDialog.getExistingDirectory(self, '选择数据集目录')
        if path:
            self.data_dir_edit.setText(path)
            self._save_config()

    def browse_model_dir(self):
        path = QFileDialog.getExistingDirectory(self, '选择模型保存目录')
        if path:
            self.model_dir_edit.setText(path)

    def _load_config(self):
        try:
            if self.CONFIG_FILE.exists():
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                if config.get('data_dir') and os.path.isdir(config['data_dir']):
                    self.data_dir_edit.setText(config['data_dir'])
        except Exception:
            pass

    def _save_config(self):
        try:
            config = {'data_dir': self.data_dir_edit.text().strip()}
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _update_device_combo(self):
        import torch
        self.device_combo.clear()
        self.device_combo.addItem('自动 (推荐)')
        self.device_combo.addItem('CPU')
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                self.device_combo.addItem(f'CUDA:{i} ({name})')
        else:
            self.device_combo.addItem('CUDA (不可用)')

    def browse_resume_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择模型文件',
            str(BASE_DIR / 'models'),
            '模型文件 (*.pth);;所有文件 (*)'
        )
        if path:
            self.resume_edit.setText(path)
            self._load_model_params_to_ui(path)

    def _load_model_params_to_ui(self, model_path):
        """从模型文件加载参数并填充到界面"""
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            args = checkpoint.get('args', {})
            
            if not args:
                self.append_log('[继续] 模型文件中无训练参数信息 (旧版本模型)')
                self.append_log('[继续] 请手动配置训练参数')
                return
            
            model_name_map = {
                'tiny': '超轻量模型',
                'small': '轻量模型',
                'standard': '标准模型',
                'large': '大型模型',
                'mobilenet': 'MobileNet模型',
                'resnet18': 'ResNet-18',
                'resnet34': 'ResNet-34',
                'resnet50': 'ResNet-50',
                'efficientnet_b0': 'EfficientNet-B0',
                'efficientnet_b1': 'EfficientNet-B1',
                'efficientnet_b2': 'EfficientNet-B2',
                'standard_se': '标准模型 + SE注意力',
                'standard_cbam': '标准模型 + CBAM注意力',
            }
            
            if 'model_name' in args:
                model_name = args['model_name']
                index = self.model_combo.findText(model_name)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
                else:
                    self.append_log(f'[继续] 警告: 未知模型类型 "{model_name}"')
            else:
                self.append_log('[继续] 警告: 模型中未找到模型类型信息')
            
            if 'input_size' in args:
                self.input_size_spin.setValue(args['input_size'])
            
            if 'batch_size' in args:
                self.batch_size_spin.setValue(args['batch_size'])
            
            if 'lr' in args:
                self.lr_spin.setValue(args['lr'])
            
            if 'val_split' in args:
                self.val_split_spin.setValue(args['val_split'])
            
            use_contrastive = args.get('use_contrastive', False)
            use_local_region = args.get('use_local_region', False)
            self.contrastive_check.setChecked(use_contrastive)
            self.local_region_check.setChecked(use_local_region)
            
            use_amp = args.get('use_amp', False)
            self.amp_check.setChecked(use_amp)
            
            if 'num_workers' in args:
                self.num_workers_spin.setValue(args['num_workers'])
            
            device = args.get('device', 'auto')
            if device == 'auto':
                self.device_combo.setCurrentIndex(0)
            elif device == 'cpu':
                self.device_combo.setCurrentIndex(1)
            elif device.startswith('cuda'):
                for i in range(self.device_combo.count()):
                    if self.device_combo.itemText(i).startswith(f'CUDA:{device.replace("cuda:", "")}'):
                        self.device_combo.setCurrentIndex(i)
                        break
            
            epoch = checkpoint.get('epoch', 0)
            best_acc = checkpoint.get('best_acc', 0)
            
            self.append_log(f'[继续] 从模型内部读取参数:')
            self.append_log(f'  • 模型类型: {model_name_map.get(args.get("model_name", ""), args.get("model_name", "未知"))} (内部标识: {args.get("model_name", "无")})')
            self.append_log(f'  • 输入尺寸: {args.get("input_size", 128)}')
            self.append_log(f'  • 批量大小: {args.get("batch_size", 32)}')
            self.append_log(f'  • 学习率: {args.get("lr", 0.001)}')
            self.append_log(f'  • 对比学习: {"启用" if use_contrastive else "禁用"}')
            self.append_log(f'  • 局部区域增强: {"启用" if use_local_region else "禁用"}')
            self.append_log(f'  • 混合精度(AMP): {"启用" if use_amp else "禁用"}')
            self.append_log(f'  • 数据加载线程: {args.get("num_workers", 0)}')
            self.append_log(f'  • 训练设备: {device}')
            self.append_log(f'  • 已训练轮次: {epoch}, 历史最佳: {best_acc:.2f}%')
            
        except Exception as e:
            self.append_log(f'[继续] 加载模型参数失败: {e}')

    def clear_resume_model(self):
        self.resume_edit.clear()

    def open_model_folder(self):
        path = self.model_dir_edit.text()
        if path and os.path.isdir(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(BASE_DIR / 'models')))

    def start_training(self):
        data_dir = self.data_dir_edit.text().strip()
        if not data_dir or not os.path.isdir(data_dir):
            QMessageBox.warning(self, '提示', '请先选择有效的数据集目录')
            return

        device_text = self.device_combo.currentText()
        if device_text.startswith('自动'):
            device = 'auto'
        elif device_text == 'CPU':
            device = 'cpu'
        elif device_text.startswith('CUDA:'):
            device = device_text.split('(')[0].strip().lower()
        else:
            device = 'auto'

        resume_path = self.resume_edit.text().strip()
        if resume_path and not os.path.isfile(resume_path):
            QMessageBox.warning(self, '提示', '选择的继续训练模型文件不存在')
            return

        config = {
            'data_dir': data_dir,
            'model_dir': self.model_dir_edit.text().strip(),
            'model_name': self.model_combo.currentText(),
            'input_size': self.input_size_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'epochs': self.epochs_spin.value(),
            'lr': self.lr_spin.value(),
            'val_split': self.val_split_spin.value(),
            'patience': self.patience_spin.value(),
            'num_workers': self.num_workers_spin.value(),
            'val_freq': self.val_freq_spin.value(),
            'device': device,
            'use_amp': self.amp_check.isChecked(),
            'resume': resume_path if resume_path else None,
            'skip_gpu_adjust': getattr(self, '_using_hw_config', False),
            'use_contrastive': self.contrastive_check.isChecked(),
            'use_local_region': self.local_region_check.isChecked(),
        }

        self.log_text.clear()
        self._log_lines = []
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(config['epochs'])
        self.time_label.setText('已用: 0:00:00   预计剩余: —')
        self.plot_canvas.set_total_epochs(config['epochs'])
        self.plot_canvas.clear()

        self._train_start_time = time.time()
        self._elapsed_timer.start(1000)

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.training_thread = TrainingThread(config)
        self.training_thread.log_signal.connect(self.append_log)
        self.training_thread.progress_signal.connect(self.update_progress)
        self.training_thread.time_signal.connect(self.update_time)
        self.training_thread.patience_signal.connect(self.on_patience_trigger)
        self.training_thread.finished_signal.connect(self.on_training_finished)
        self.training_thread.plot_data_signal.connect(self.update_plot)
        self.training_thread.start()

    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.request_stop()
            self.append_log('[中断] 正在安全停止...')

    def append_log(self, msg):
        ts = datetime.now().strftime('[%H:%M:%S]')
        log_line = f'{ts} {msg}'
        self.log_text.append(log_line)
        self._log_lines.append(log_line)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, current, total):
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f'第 {current}/{total} 轮 (%p%)')

    def on_training_finished(self, success, msg):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._elapsed_timer.stop()
        
        if self._log_lines:
            if success and msg:
                log_path = msg.replace('.pth', '_log.txt')
            else:
                log_dir = BASE_DIR / 'logs'
                log_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_path = str(log_dir / f'training_{timestamp}_log.txt')
            
            try:
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self._log_lines))
                self.log_text.append(f'[日志] 训练日志已保存: {log_path}')
            except Exception as e:
                self.log_text.append(f'[日志] 保存日志失败: {e}')
        
        if success:
            self.progress_bar.setFormat(f'完成! 模型: {os.path.basename(msg)}')
        else:
            self.progress_bar.setFormat('训练中断')

    def _update_elapsed(self):
        if self._train_start_time and self.training_thread and self.training_thread.isRunning():
            elapsed = time.time() - self._train_start_time
            elapsed_str = _format_duration(elapsed)
            current_text = self.time_label.text()
            if '预计剩余' in current_text:
                parts = current_text.split('   ')
                if len(parts) >= 2:
                    self.time_label.setText(f'已用: {elapsed_str}   {parts[1]}')
                else:
                    self.time_label.setText(f'已用: {elapsed_str}   预计剩余: —')
            else:
                self.time_label.setText(f'已用: {elapsed_str}   预计剩余: —')

    def update_time(self, time_str):
        self.time_label.setText(time_str)

    def update_plot(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, elapsed_time):
        self.plot_canvas.add_data(epoch, train_loss, train_acc, val_loss, val_acc, lr, elapsed_time)

    def on_patience_trigger(self, epoch, best_acc, val_acc):
        from PyQt5.QtWidgets import QMessageBox
        self._elapsed_timer.stop()
        reply = QMessageBox.question(
            self, '训练进度提示',
            f'已经连续 {self.patience_spin.value()} 轮验证准确率没有提升了。\n\n'
            f'当前轮: 第 {epoch} 轮\n'
            f'最佳验证准确率: {best_acc:.2f}%\n'
            f'当前验证准确率: {val_acc:.2f}%\n\n'
            f'模型训练可能已达最优, 是否继续训练？',
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            if self.training_thread:
                self.training_thread.request_stop()
        else:
            if self.training_thread:
                self.training_thread.request_continue()
            self._elapsed_timer.start(1000)

    def show_data_hint(self):
        msg = (
            '📁 数据集目录结构要求:\n\n'
            '选择包含角色子文件夹的上级目录。\n'
            '每个子文件夹名即角色标签，子文件夹内存放该角色的图片。\n\n'
            '示例:\n'
            '  dataset/\n'
            '    ├── 鸣人/          ← 自动识别为"鸣人"\n'
            '    │   ├── 001.jpg\n'
            '    │   └── 002.jpg\n'
            '    ├── 佐助/          ← 自动识别为"佐助"\n'
            '    │   └── 001.jpg\n'
            '    └── ...\n\n'
            '支持的图片格式: .jpg .jpeg .png .bmp .webp .tiff'
        )
        box = QMessageBox(self)
        box.setWindowTitle('数据集要求')
        box.setText(msg)
        box.setFont(QFont('Microsoft YaHei', 9))
        box.exec_()

    def show_ref_table(self):
        msg = (
            '📋 按数据总量推荐训练配置\n\n'
            '  < 100 张         | tiny     | 96px  | 批量8  | 120轮 | 0.001 | 验证0.15\n'
            '  100 ~ 500 张     | small    | 128px | 批量16 | 150轮 | 0.001 | 验证0.15\n'
            '  500 ~ 2000 张    | standard | 224px | 批量32 | 150轮 | 0.001 | 验证0.2\n'
            '  2000 ~ 5000 张   | large    | 256px | 批量32 | 120轮 | 0.001 | 验证0.2\n'
            '  > 5000 张        | large    | 256px | 批量64 | 100轮 | 0.0005| 验证0.2\n\n'
            '提示: 用"智能推荐参数"按钮可自动分析数据集并填入推荐值。'
        )
        box = QMessageBox(self)
        box.setWindowTitle('参数速查表')
        box.setText(msg)
        box.setFont(QFont('Microsoft YaHei', 9))
        box.exec_()

    def start_hardware_detection(self):
        data_dir = self.data_dir_edit.text().strip()
        
        if not data_dir or not os.path.isdir(data_dir):
            QMessageBox.warning(self, '提示', '请先选择有效的数据集目录')
            return
        
        try:
            from pathlib import Path
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
            num_classes = 0
            for subdir in Path(data_dir).iterdir():
                if subdir.is_dir():
                    if any(f.suffix.lower() in image_exts for f in subdir.iterdir()):
                        num_classes += 1
        except Exception:
            num_classes = 86
        
        self.log_text.clear()
        self._log_lines = []
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(4)
        self.progress_bar.setFormat('硬件检测中...')
        
        self.hw_detect_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        
        self.hw_detect_thread = HardwareDetectThread(data_dir, num_classes)
        self.hw_detect_thread.log_signal.connect(self.append_log)
        self.hw_detect_thread.progress_signal.connect(self.update_hw_progress)
        self.hw_detect_thread.finished_signal.connect(self.on_hardware_detection_finished)
        self.hw_detect_thread.start()

    def update_hw_progress(self, current, total):
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f'检测进度: {current}/{total}')

    def on_hardware_detection_finished(self, hw_info, optimal_config):
        self.hw_detect_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.progress_bar.setFormat('检测完成')
        
        if hw_info is None or optimal_config is None:
            QMessageBox.critical(self, '错误', '硬件检测失败，请查看日志')
            return
        
        from utils.hardware_detector import format_hardware_report, format_config_report
        
        hw_report = format_hardware_report(hw_info)
        config_report = format_config_report(optimal_config)
        
        self._last_optimal_config = optimal_config
        
        msg = f'{hw_report}\n\n{config_report}'
        
        box = QMessageBox(self)
        box.setWindowTitle('硬件检测结果')
        box.setText(msg)
        box.setFont(QFont('Consolas', 9))
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.button(QMessageBox.Yes).setText('一键应用配置')
        box.button(QMessageBox.No).setText('关闭')
        box.setDefaultButton(QMessageBox.Yes)
        
        if box.exec_() == QMessageBox.Yes:
            self.apply_optimal_config(optimal_config)

    def apply_optimal_config(self, config):
        self.model_combo.setCurrentText(config.model_name)
        self.input_size_spin.setValue(config.input_size)
        self.batch_size_spin.setValue(config.batch_size)
        self.num_workers_spin.setValue(config.num_workers)
        self.amp_check.setChecked(config.use_amp)
        self._using_hw_config = True
        
        self.append_log(f'[配置已应用] 模型:{config.model_name}, 输入:{config.input_size}, '
                        f'批量:{config.batch_size}, 线程:{config.num_workers}, AMP:{config.use_amp}')
        
        QMessageBox.information(self, '成功', 
            f'已应用最优配置:\n'
            f'• 模型类型: {config.model_name}\n'
            f'• 输入尺寸: {config.input_size}\n'
            f'• 批量大小: {config.batch_size}\n'
            f'• 数据加载线程: {config.num_workers}\n'
            f'• 混合精度: {"启用" if config.use_amp else "禁用"}\n'
            f'• 预计每轮时间: {config.estimated_time_per_epoch_min:.1f}分钟')

    def show_recommendations(self):
        data_dir = self.data_dir_edit.text().strip()
        if not data_dir or not os.path.isdir(data_dir):
            QMessageBox.warning(self, '提示', '请先选择有效的数据集目录')
            return

        try:
            from pathlib import Path
            from PIL import Image

            data_path = Path(data_dir)
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

            total_images = 0
            class_counts = {}
            sizes = []

            for subdir in sorted(data_path.iterdir()):
                if not subdir.is_dir():
                    continue
                imgs = [f for f in subdir.iterdir() if f.suffix.lower() in image_exts]
                count = len(imgs)
                if count > 0:
                    class_counts[subdir.name] = count
                    total_images += count
                    try:
                        with Image.open(imgs[0]) as img:
                            sizes.append(img.size)
                    except Exception:
                        pass

            num_classes = len(class_counts)
            if num_classes == 0:
                QMessageBox.warning(self, '提示', '没有找到有效的角色子文件夹')
                return

            avg_per_class = total_images / num_classes
            min_per_class = min(class_counts.values())
            max_per_class = max(class_counts.values())

            if sizes:
                avg_w = sum(s[0] for s in sizes) / len(sizes)
                avg_h = sum(s[1] for s in sizes) / len(sizes)
                avg_size = int((avg_w + avg_h) / 2)
            else:
                avg_size = 256

            if total_images < 30 and num_classes <= 5:
                rec_model = 'tiny'
            elif total_images < 100:
                rec_model = 'small'
            elif total_images < 2000:
                rec_model = 'standard'
            else:
                rec_model = 'large'

            if avg_size <= 100:
                rec_input = 96
            elif avg_size <= 200:
                rec_input = 128
            else:
                rec_input = min(256, avg_size)
            rec_input = max(32, (rec_input // 32) * 32)

            if total_images < 100:
                rec_batch = 8
            elif total_images < 500:
                rec_batch = 16
            elif total_images < 2000:
                rec_batch = 32
            else:
                rec_batch = 64

            if total_images < 50:
                rec_epochs = 80
            elif total_images < 200:
                rec_epochs = 120
            elif total_images < 1000:
                rec_epochs = 150
            else:
                rec_epochs = 200

            rec_lr = 0.001

            if total_images < 100:
                rec_val = 0.15
            elif total_images < 500:
                rec_val = 0.15
            else:
                rec_val = 0.2

            info_lines = [
                f'📊 数据集分析结果:',
                f'',
                f'  分类数量: {num_classes} 个角色',
                f'  图片总数: {total_images} 张',
                f'  平均每类: {avg_per_class:.1f} 张',
                f'  最少: {min_per_class} 张 ({min(class_counts, key=class_counts.get)})',
                f'  最多: {max_per_class} 张 ({max(class_counts, key=class_counts.get)})',
                f'  图片平均尺寸: {avg_size}×{avg_size} px',
                f'',
                f'📋 推荐训练参数:',
                f'',
                f'  模型类型: {rec_model}',
                f'  输入尺寸: {rec_input}',
                f'  批量大小: {rec_batch}',
                f'  训练轮数: {rec_epochs}',
                f'  学习率: {rec_lr}',
                f'  验证集比例: {rec_val}',
            ]

            msg = '\n'.join(info_lines)
            box = QMessageBox(self)
            box.setWindowTitle('智能推荐参数')
            box.setText(msg)
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.button(QMessageBox.Yes).setText('一键填入')
            box.button(QMessageBox.No).setText('不使用')
            box.setDefaultButton(QMessageBox.Yes)

            if box.exec_() == QMessageBox.Yes:
                self.model_combo.setCurrentText(rec_model)
                self.input_size_spin.setValue(rec_input)
                self.batch_size_spin.setValue(rec_batch)
                self.epochs_spin.setValue(rec_epochs)
                self.lr_spin.setValue(rec_lr)
                self.val_split_spin.setValue(rec_val)

        except Exception as e:
            QMessageBox.warning(self, '错误', f'分析数据集失败: {e}')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('')
        self.setMinimumSize(1100, 650)
        self.resize(1380, 1008)
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)

        self.tabs = QTabWidget()
        self.tabs.setFont(QFont('Microsoft YaHei', 11))
        self.tabs.addTab(TrainTab(), ' 训练模型')
        layout.addWidget(self.tabs)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    translator = QTranslator()
    trans_path = QLibraryInfo.location(QLibraryInfo.TranslationsPath)
    if translator.load(QLocale.system(), 'qt', '_', trans_path):
        app.installTranslator(translator)

    font = QFont('Microsoft YaHei', 10)
    font.setStyleStrategy(QFont.PreferAntialias)
    font.setHintingPreference(QFont.PreferNoHinting)
    app.setFont(font)

    global_stylesheet = '''
        QWidget {
            font-weight: 500;
        }
        QMainWindow {
            background-color: #f5f5f5;
        }
        QLabel {
            font-weight: 500;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 16px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            font-weight: bold;
        }
        QTextEdit, QLineEdit {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            background-color: white;
            font-weight: normal;
        }
        QPushButton {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background-color: #2196F3;
            color: white;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QPushButton:pressed {
            background-color: #0d47a1;
        }
        QPushButton:disabled {
            background-color: #cccccc;
        }
        QSpinBox, QDoubleSpinBox, QComboBox {
            padding: 4px 8px;
            border: 1px solid #ddd;
            border-radius: 3px;
            background-color: white;
            font-weight: normal;
        }
        QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #2196F3;
        }
        QProgressBar {
            border: 1px solid #ddd;
            border-radius: 4px;
            text-align: center;
            font-weight: bold;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
        }
        QTabWidget::pane {
            border: 1px solid #ddd;
            background-color: white;
        }
        QTabBar::tab {
            font-weight: bold;
            background-color: #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #2196F3;
        }
    '''
    app.setStyleSheet(global_stylesheet)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
