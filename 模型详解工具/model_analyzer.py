import sys
import os
import torch
import json
import numpy as np
from collections import OrderedDict

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout, QScrollArea,
    QFrame, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QCheckBox, QTextEdit, QProgressBar, QSpinBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QFontMetrics

MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


STYLES = {
    'main_bg': '#1e1e1e',
    'panel_bg': '#252526',
    'card_bg': '#2d2d30',
    'input_bg': '#3c3c3c',
    'accent': '#0078d4',
    'accent_hover': '#1a8cff',
    'accent_pressed': '#0066b8',
    'text_primary': '#e0e0e0',
    'text_secondary': '#9d9d9d',
    'text_muted': '#6d6d6d',
    'text_highlight': '#4fc3f7',
    'text_success': '#6fcf97',
    'text_warning': '#f2c94c',
    'text_error': '#f28b82',
    'border': '#3c3c3c',
    'border_light': '#4a4a4a',
    'border_focus': '#0078d4',
    'success': '#6fcf97',
    'warning': '#f2c94c',
    'error': '#f28b82',
    'info': '#4fc3f7',
    'selection': '#264f78',
    'hover': '#2a2d2e',
    'scrollbar': '#424242',
    'scrollbar_hover': '#4f4f4f',
}


class LayerVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.layer_info = []
        self.setMinimumSize(600, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def set_layers(self, layer_info):
        self.layer_info = layer_info[:100] if layer_info else []
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        painter.fillRect(self.rect(), QColor(STYLES['main_bg']))
        
        if not self.layer_info:
            painter.setPen(QPen(QColor(STYLES['text_secondary'])))
            font = QFont('Microsoft YaHei UI', 12)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, '请先加载模型文件')
            return
        
        width = self.width()
        height = self.height()
        
        margin = 30
        usable_width = width - 2 * margin
        
        layer_width = min(100, max(60, usable_width // max(len(self.layer_info), 1)))
        layer_height = 50
        spacing = min(15, max(5, (usable_width - len(self.layer_info) * layer_width) // max(len(self.layer_info) - 1, 1)))
        
        total_width = len(self.layer_info) * layer_width + (len(self.layer_info) - 1) * spacing
        start_x = max(margin, (width - total_width) // 2)
        
        y_center = height // 2
        
        type_colors = {
            'Conv': QColor('#6fcf97'),
            'BatchNorm': QColor('#4fc3f7'),
            'ReLU': QColor('#f2c94c'),
            'MaxPool': QColor('#bb86fc'),
            'AvgPool': QColor('#7c4dff'),
            'Linear': QColor('#f28b82'),
            'Dropout': QColor('#9d9d9d'),
            'Flatten': QColor('#6d6d6d'),
            'Sequential': QColor('#03dac6'),
            'Bottleneck': QColor('#cf6679'),
            'Downsample': QColor('#ff7597'),
            'Other': QColor('#6d6d6d'),
        }
        
        for i, layer in enumerate(self.layer_info):
            x = start_x + i * (layer_width + spacing)
            y = y_center - layer_height // 2
            
            layer_type = layer.get('type', 'Other')
            for key in type_colors:
                if key in layer_type:
                    color = type_colors[key]
                    break
            else:
                color = type_colors['Other']
            
            gradient = QLinearGradient(x, y, x, y + layer_height)
            gradient.setColorAt(0, color.lighter(130))
            gradient.setColorAt(1, color)
            
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(color.darker(150), 1))
            painter.drawRoundedRect(x, y, layer_width, layer_height, 6, 6)
            
            painter.setPen(QPen(QColor('white')))
            font = QFont('Microsoft YaHei UI', 7)
            painter.setFont(font)
            
            name = layer.get('name', 'Unknown')
            if len(name) > 10:
                name = name[:8] + '..'
            
            fm = QFontMetrics(font)
            text_rect = fm.boundingRect(name)
            text_x = x + (layer_width - text_rect.width()) // 2
            text_y = y + layer_height // 2 - 5
            painter.drawText(text_x, text_y, name)
            
            painter.setFont(QFont('Microsoft YaHei UI', 6))
            params = layer.get('params', 0)
            if params > 1e6:
                params_str = f'{params/1e6:.1f}M'
            elif params > 1e3:
                params_str = f'{params/1e3:.1f}K'
            else:
                params_str = str(params)
            
            fm2 = QFontMetrics(QFont('Microsoft YaHei UI', 6))
            text_rect2 = fm2.boundingRect(params_str)
            text_x2 = x + (layer_width - text_rect2.width()) // 2
            text_y2 = y + layer_height - 8
            painter.drawText(text_x2, text_y2, params_str)
            
            if i < len(self.layer_info) - 1:
                painter.setPen(QPen(QColor('#555'), 2))
                arrow_start = x + layer_width
                arrow_end = x + layer_width + spacing
                painter.drawLine(arrow_start, y_center, arrow_end, y_center)
        
        painter.setPen(QPen(QColor(STYLES['text_secondary'])))
        painter.setFont(QFont('Microsoft YaHei UI', 9))
        painter.drawText(margin, height - 10, f'共 {len(self.layer_info)} 层')


if MATPLOTLIB_AVAILABLE:
    class WeightHistogram(FigureCanvas):
        def __init__(self):
            self.fig = Figure(figsize=(8, 4), dpi=100)
            self.fig.patch.set_facecolor(STYLES['main_bg'])
            super().__init__(self.fig)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor(STYLES['main_bg'])
            self.setMinimumHeight(250)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._setup_empty_plot()
        
        def _setup_empty_plot(self):
            self.ax.clear()
            self.ax.set_facecolor(STYLES['main_bg'])
            self.ax.text(0.5, 0.5, '请先加载模型文件', 
                        ha='center', va='center', fontsize=12, color='#9d9d9d',
                        transform=self.ax.transAxes)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            for spine in self.ax.spines.values():
                spine.set_color('#3c3c3c')
            self.fig.tight_layout()
            self.draw()
        
        def plot_weight(self, weight_data, layer_name):
            self.ax.clear()
            self.ax.set_facecolor(STYLES['main_bg'])
            
            if weight_data is None:
                self.ax.text(0.5, 0.5, '无权重数据', ha='center', va='center',
                            color='#9d9d9d', transform=self.ax.transAxes, fontsize=12)
                for spine in self.ax.spines.values():
                    spine.set_color('#3c3c3c')
                self.draw()
                return
            
            weight_flat = weight_data.flatten().cpu().numpy()
            
            self.ax.hist(weight_flat, bins=50, color=STYLES['accent'], edgecolor=STYLES['accent_pressed'], alpha=0.8)
            
            display_name = layer_name if len(layer_name) <= 30 else '...' + layer_name[-27:]
            self.ax.set_title(f'{display_name} 权重分布', color='#e0e0e0', fontsize=11, pad=10)
            self.ax.set_xlabel('权重值', color='#e0e0e0', fontsize=10)
            self.ax.set_ylabel('频次', color='#e0e0e0', fontsize=10)
            self.ax.tick_params(colors='#e0e0e0', labelsize=9)
            
            for spine in self.ax.spines.values():
                spine.set_color('#e0e0e0')
            
            self.ax.grid(True, alpha=0.2, color='#4a4a4a')
            self.fig.tight_layout()
            self.draw()
else:
    class WeightHistogram(QLabel):
        def __init__(self):
            super().__init__()
            self.setAlignment(Qt.AlignCenter)
            self.setStyleSheet(f'''
                background-color: {STYLES['card_bg']}; 
                color: {STYLES['text_secondary']};
                font-size: 14px;
                padding: 20px;
            ''')
            self.setText('需要安装 matplotlib 才能显示权重分布图\n\npip install matplotlib')
            self.setMinimumHeight(250)
        
        def plot_weight(self, weight_data, layer_name):
            pass


if MATPLOTLIB_AVAILABLE:
    class TrainingCurveWidget(FigureCanvas):
        def __init__(self):
            self.fig = Figure(figsize=(8, 5), dpi=100)
            self.fig.patch.set_facecolor(STYLES['main_bg'])
            super().__init__(self.fig)
            self.setMinimumHeight(350)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._setup_empty_plot()
        
        def _setup_empty_plot(self):
            self.fig.clear()
            ax = self.fig.add_subplot(1, 1, 1)
            ax.set_facecolor(STYLES['main_bg'])
            ax.text(0.5, 0.5, '请先加载模型文件', 
                   ha='center', va='center', fontsize=12, color='#9d9d9d',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#3c3c3c')
            self.fig.tight_layout(pad=1.0)
            self.draw()
        
        def plot_curves(self, epochs, train_loss, val_loss, train_acc, val_acc, lr_history):
            self.fig.clear()
            
            if not epochs and not train_loss:
                self._setup_empty_plot()
                return
            
            axes = []
            for i in range(4):
                ax = self.fig.add_subplot(2, 2, i + 1)
                ax.set_facecolor(STYLES['main_bg'])
                for spine in ax.spines.values():
                    spine.set_color('#e0e0e0')
                ax.tick_params(colors='#e0e0e0', labelsize=8)
                axes.append(ax)
            
            if train_loss:
                axes[0].plot(epochs, train_loss, color=STYLES['success'], label='训练损失', linewidth=1.5)
                axes[0].plot(epochs, val_loss, color=STYLES['error'], label='验证损失', linewidth=1.5)
                axes[0].set_xlabel('轮次', color='#e0e0e0', fontsize=9)
                axes[0].set_ylabel('损失', color='#e0e0e0', fontsize=9)
                axes[0].set_title('损失曲线', color='#e0e0e0', fontsize=10, pad=8)
                axes[0].legend(facecolor=STYLES['card_bg'], edgecolor=STYLES['border'], 
                              labelcolor='#e0e0e0', fontsize=8)
                axes[0].tick_params(colors='#e0e0e0', labelsize=8)
                axes[0].grid(True, alpha=0.2, color='#4a4a4a')
            
            if train_acc:
                axes[1].plot(epochs, train_acc, color=STYLES['success'], label='训练准确率', linewidth=1.5)
                axes[1].plot(epochs, val_acc, color=STYLES['error'], label='验证准确率', linewidth=1.5)
                axes[1].set_xlabel('轮次', color='#e0e0e0', fontsize=9)
                axes[1].set_ylabel('准确率 (%)', color='#e0e0e0', fontsize=9)
                axes[1].set_title('准确率曲线', color='#e0e0e0', fontsize=10, pad=8)
                axes[1].legend(facecolor=STYLES['card_bg'], edgecolor=STYLES['border'], 
                              labelcolor='#e0e0e0', fontsize=8)
                axes[1].tick_params(colors='#e0e0e0', labelsize=8)
                axes[1].grid(True, alpha=0.2, color='#4a4a4a')
            
            if lr_history:
                axes[2].plot(epochs, lr_history, color=STYLES['info'], linewidth=1.5)
                axes[2].set_xlabel('轮次', color='#e0e0e0', fontsize=9)
                axes[2].set_ylabel('学习率', color='#e0e0e0', fontsize=9)
                axes[2].set_title('学习率变化', color='#e0e0e0', fontsize=10, pad=8)
                axes[2].tick_params(colors='#e0e0e0', labelsize=8)
                axes[2].grid(True, alpha=0.2, color='#4a4a4a')
            
            if train_acc and val_acc:
                gap = [t - v for t, v in zip(train_acc, val_acc)]
                axes[3].plot(epochs, gap, color='#bb86fc', linewidth=1.5)
                axes[3].axhline(y=0, color='#e0e0e0', linestyle='--', alpha=0.5)
                axes[3].set_xlabel('轮次', color='#e0e0e0', fontsize=9)
                axes[3].set_ylabel('准确率差距 (%)', color='#e0e0e0', fontsize=9)
                axes[3].set_title('过拟合检测', color='#e0e0e0', fontsize=10, pad=8)
                axes[3].tick_params(colors='#e0e0e0', labelsize=8)
                axes[3].grid(True, alpha=0.2, color='#4a4a4a')
            
            self.fig.tight_layout(pad=1.5, h_pad=2.0, w_pad=2.0)
            self.draw()
else:
    class TrainingCurveWidget(QLabel):
        def __init__(self):
            super().__init__()
            self.setAlignment(Qt.AlignCenter)
            self.setStyleSheet(f'''
                background-color: {STYLES['card_bg']}; 
                color: {STYLES['text_secondary']};
                font-size: 14px;
                padding: 20px;
            ''')
            self.setText('需要安装 matplotlib 才能显示训练曲线\n\npip install matplotlib')
            self.setMinimumHeight(400)
        
        def plot_curves(self, epochs, train_loss, val_loss, train_acc, val_acc, lr_history):
            pass


class ModelAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.checkpoint = None
        self.model = None
        self.model_path = None
        self.layer_weights = {}
        self.layer_info = []
        
        self.setWindowTitle('模型详解工具 - 深度解析神经网络')
        self.setMinimumSize(1200, 800)
        self.setup_ui()
        self.apply_styles()
    
    def apply_styles(self):
        self.setStyleSheet(f'''
            QMainWindow {{
                background-color: {STYLES['main_bg']};
            }}
            QWidget {{
                color: {STYLES['text_primary']};
                font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;
                font-size: 10pt;
            }}
            QGroupBox {{
                font-weight: bold;
                font-size: 11pt;
                border: 1px solid {STYLES['border']};
                border-radius: 6px;
                margin-top: 12px;
                padding: 15px;
                padding-top: 25px;
                background-color: {STYLES['card_bg']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: {STYLES['text_highlight']};
            }}
            QLabel {{
                color: {STYLES['text_primary']};
                line-height: 1.5;
            }}
            QPushButton {{
                background-color: {STYLES['accent']};
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
                font-size: 10pt;
            }}
            QPushButton:hover {{
                background-color: {STYLES['accent_hover']};
            }}
            QPushButton:pressed {{
                background-color: {STYLES['accent_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {STYLES['scrollbar']};
                color: {STYLES['text_muted']};
            }}
            QLineEdit {{
                background-color: {STYLES['input_bg']};
                border: 1px solid {STYLES['border']};
                padding: 6px 10px;
                border-radius: 4px;
                color: {STYLES['text_primary']};
                font-size: 10pt;
            }}
            QLineEdit:focus {{
                border: 1px solid {STYLES['border_focus']};
            }}
            QLineEdit:read-only {{
                background-color: {STYLES['panel_bg']};
            }}
            QComboBox {{
                background-color: {STYLES['input_bg']};
                border: 1px solid {STYLES['border']};
                padding: 6px 10px;
                border-radius: 4px;
                color: {STYLES['text_primary']};
                min-width: 200px;
            }}
            QComboBox:hover {{
                border: 1px solid {STYLES['border_light']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {STYLES['text_secondary']};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {STYLES['card_bg']};
                color: {STYLES['text_primary']};
                selection-background-color: {STYLES['selection']};
                border: 1px solid {STYLES['border']};
                border-radius: 4px;
            }}
            QTextEdit {{
                background-color: {STYLES['input_bg']};
                border: 1px solid {STYLES['border']};
                border-radius: 4px;
                padding: 8px;
                color: {STYLES['text_primary']};
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 9pt;
                line-height: 1.4;
            }}
            QTextEdit:focus {{
                border: 1px solid {STYLES['border_focus']};
            }}
            QTreeWidget {{
                background-color: {STYLES['panel_bg']};
                border: 1px solid {STYLES['border']};
                border-radius: 4px;
                color: {STYLES['text_primary']};
                font-size: 9pt;
                padding: 4px;
                outline: none;
            }}
            QTreeWidget::item {{
                padding: 4px 6px;
                border-radius: 3px;
            }}
            QTreeWidget::item:selected {{
                background-color: {STYLES['selection']};
                color: {STYLES['text_primary']};
            }}
            QTreeWidget::item:hover {{
                background-color: {STYLES['hover']};
            }}
            QTreeWidget::branch {{
                background-color: transparent;
            }}
            QTableWidget {{
                background-color: {STYLES['panel_bg']};
                border: 1px solid {STYLES['border']};
                border-radius: 4px;
                color: {STYLES['text_primary']};
                font-size: 9pt;
                gridline-color: {STYLES['border']};
                outline: none;
            }}
            QTableWidget::item {{
                padding: 6px 8px;
            }}
            QTableWidget::item:selected {{
                background-color: {STYLES['selection']};
                color: {STYLES['text_primary']};
            }}
            QTableWidget::item:hover {{
                background-color: {STYLES['hover']};
            }}
            QHeaderView::section {{
                background-color: {STYLES['card_bg']};
                color: {STYLES['text_secondary']};
                padding: 8px 10px;
                border: none;
                border-bottom: 1px solid {STYLES['border']};
                border-right: 1px solid {STYLES['border']};
                font-weight: bold;
                font-size: 9pt;
            }}
            QHeaderView::section:last {{
                border-right: none;
            }}
            QTabWidget::pane {{
                border: 1px solid {STYLES['border']};
                background-color: {STYLES['main_bg']};
                border-radius: 4px;
                top: -1px;
            }}
            QTabBar {{
                background-color: transparent;
            }}
            QTabBar::tab {{
                background-color: {STYLES['panel_bg']};
                color: {STYLES['text_secondary']};
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border: 1px solid {STYLES['border']};
                border-bottom: none;
                font-weight: normal;
            }}
            QTabBar::tab:selected {{
                background-color: {STYLES['main_bg']};
                color: {STYLES['text_primary']};
                border-bottom: 1px solid {STYLES['main_bg']};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {STYLES['card_bg']};
                color: {STYLES['text_primary']};
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background-color: transparent;
                width: 10px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background-color: {STYLES['scrollbar']};
                border-radius: 5px;
                min-height: 30px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {STYLES['scrollbar_hover']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background-color: transparent;
            }}
            QScrollBar:horizontal {{
                background-color: transparent;
                height: 10px;
                margin: 0;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {STYLES['scrollbar']};
                border-radius: 5px;
                min-width: 30px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {STYLES['scrollbar_hover']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                background-color: transparent;
            }}
            QSplitter::handle {{
                background-color: {STYLES['border']};
            }}
            QSplitter::handle:horizontal {{
                width: 1px;
            }}
            QSplitter::handle:vertical {{
                height: 1px;
            }}
            QFrame {{
                border: none;
            }}
            QFrame[frameShape="4"] {{
                background-color: {STYLES['border']};
            }}
        ''')
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        top_frame = QFrame()
        top_frame.setStyleSheet(f'''
            QFrame {{
                background-color: {STYLES['card_bg']};
                border-radius: 10px;
                padding: 5px;
            }}
        ''')
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(15, 10, 15, 10)
        
        path_label = QLabel('模型文件:')
        path_label.setStyleSheet(f'color: {STYLES["text_secondary"]}; font-weight: bold;')
        top_layout.addWidget(path_label)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText('请选择模型文件 (.pth)')
        top_layout.addWidget(self.model_path_edit, 1)
        
        browse_btn = QPushButton('选择模型')
        browse_btn.setFixedWidth(100)
        browse_btn.clicked.connect(self.browse_model)
        top_layout.addWidget(browse_btn)
        
        layout.addWidget(top_frame)
        
        self.tabs = QTabWidget()
        
        self.setup_basic_info_tab()
        self.setup_structure_tab()
        self.setup_weights_tab()
        self.setup_training_tab()
        self.setup_raw_data_tab()
        self.setup_export_tab()
        
        layout.addWidget(self.tabs, 1)
        
        self.status_label = QLabel('就绪 - 请选择模型文件开始分析')
        self.status_label.setStyleSheet(f'''
            color: {STYLES['text_secondary']}; 
            padding: 10px 15px;
            background-color: {STYLES['card_bg']};
            border-radius: 6px;
        ''')
        layout.addWidget(self.status_label)
    
    def setup_basic_info_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f'''
            QScrollArea {{
                border: none;
                background-color: {STYLES['main_bg']};
            }}
            QScrollArea > QWidget > QWidget {{
                background-color: {STYLES['main_bg']};
            }}
        ''')
        
        content = QWidget()
        content.setStyleSheet(f'background-color: {STYLES["main_bg"]};')
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(15)
        
        file_group = QGroupBox('文件信息')
        file_layout = QFormLayout(file_group)
        file_layout.setSpacing(12)
        file_layout.setLabelAlignment(Qt.AlignRight)
        
        self.file_path_label = self._create_value_label()
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setStyleSheet(f'color: {STYLES["success"]};')
        file_layout.addRow('文件路径:', self.file_path_label)
        
        self.file_size_label = self._create_value_label()
        file_layout.addRow('文件大小:', self.file_size_label)
        
        self.create_time_label = self._create_value_label()
        file_layout.addRow('创建时间:', self.create_time_label)
        
        content_layout.addWidget(file_group)
        
        model_group = QGroupBox('模型信息')
        model_layout = QFormLayout(model_group)
        model_layout.setSpacing(12)
        model_layout.setLabelAlignment(Qt.AlignRight)
        
        self.model_type_label = self._create_value_label(highlight=True)
        model_layout.addRow('模型类型:', self.model_type_label)
        
        self.input_size_label = self._create_value_label()
        model_layout.addRow('输入尺寸:', self.input_size_label)
        
        self.class_count_label = self._create_value_label(highlight=True)
        model_layout.addRow('类别数量:', self.class_count_label)
        
        self.total_params_label = self._create_value_label(highlight=True)
        model_layout.addRow('总参数量:', self.total_params_label)
        
        self.trainable_params_label = self._create_value_label()
        model_layout.addRow('可训练参数:', self.trainable_params_label)
        
        self.model_size_label = self._create_value_label()
        model_layout.addRow('模型大小:', self.model_size_label)
        
        content_layout.addWidget(model_group)
        
        layers_group = QGroupBox('层统计')
        layers_layout = QFormLayout(layers_group)
        layers_layout.setSpacing(12)
        layers_layout.setLabelAlignment(Qt.AlignRight)
        
        self.layer_count_label = self._create_value_label()
        layers_layout.addRow('总层数:', self.layer_count_label)
        
        self.conv_count_label = self._create_value_label()
        layers_layout.addRow('卷积层:', self.conv_count_label)
        
        self.bn_count_label = self._create_value_label()
        layers_layout.addRow('归一化层:', self.bn_count_label)
        
        self.linear_count_label = self._create_value_label()
        layers_layout.addRow('全连接层:', self.linear_count_label)
        
        self.pool_count_label = self._create_value_label()
        layers_layout.addRow('池化层:', self.pool_count_label)
        
        content_layout.addWidget(layers_group)
        
        classes_group = QGroupBox('类别列表')
        classes_layout = QVBoxLayout(classes_group)
        
        self.class_list = QTextEdit()
        self.class_list.setReadOnly(True)
        self.class_list.setMinimumHeight(150)
        self.class_list.setMaximumHeight(250)
        classes_layout.addWidget(self.class_list)
        
        content_layout.addWidget(classes_group)
        
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        self.tabs.addTab(tab, '基本信息')
    
    def setup_structure_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_frame = QFrame()
        left_frame.setStyleSheet(f'background-color: {STYLES["panel_bg"]}; border-radius: 8px;')
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        left_title = QLabel('层结构树')
        left_title.setStyleSheet(f'color: {STYLES["text_highlight"]}; font-weight: bold; font-size: 11pt;')
        left_layout.addWidget(left_title)
        
        self.layer_tree = QTreeWidget()
        self.layer_tree.setHeaderLabels(['层名称', '类型', '参数量', '输出形状'])
        self.layer_tree.setColumnWidth(0, 200)
        self.layer_tree.setColumnWidth(1, 100)
        self.layer_tree.setColumnWidth(2, 100)
        self.layer_tree.itemClicked.connect(self.on_layer_selected)
        left_layout.addWidget(self.layer_tree)
        
        splitter.addWidget(left_frame)
        
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        right_title = QLabel('网络结构可视化')
        right_title.setStyleSheet(f'color: {STYLES["text_highlight"]}; font-weight: bold; font-size: 11pt;')
        right_layout.addWidget(right_title)
        
        self.layer_visualizer = LayerVisualizer()
        self.layer_visualizer.setMinimumHeight(200)
        right_layout.addWidget(self.layer_visualizer, 1)
        
        detail_title = QLabel('层详情')
        detail_title.setStyleSheet(f'color: {STYLES["text_highlight"]}; font-weight: bold; font-size: 11pt; margin-top: 10px;')
        right_layout.addWidget(detail_title)
        
        self.layer_detail_table = QTableWidget()
        self.layer_detail_table.setColumnCount(2)
        self.layer_detail_table.setHorizontalHeaderLabels(['属性', '值'])
        self.layer_detail_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.layer_detail_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.layer_detail_table.setMaximumHeight(180)
        right_layout.addWidget(self.layer_detail_table)
        
        splitter.addWidget(right_frame)
        splitter.setSizes([350, 650])
        
        layout.addWidget(splitter)
        self.tabs.addTab(tab, '网络结构')
    
    def setup_weights_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        top_frame = QFrame()
        top_frame.setStyleSheet(f'background-color: {STYLES["card_bg"]}; border-radius: 8px; padding: 10px;')
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(15, 10, 15, 10)
        
        layer_label = QLabel('选择层:')
        layer_label.setStyleSheet('font-weight: bold;')
        top_layout.addWidget(layer_label)
        
        self.weight_layer_combo = QComboBox()
        self.weight_layer_combo.setMinimumWidth(350)
        self.weight_layer_combo.currentTextChanged.connect(self.on_weight_layer_changed)
        top_layout.addWidget(self.weight_layer_combo)
        
        top_layout.addStretch()
        
        self.weight_stats_label = QLabel('')
        self.weight_stats_label.setStyleSheet(f'color: {STYLES["text_secondary"]};')
        top_layout.addWidget(self.weight_stats_label)
        
        layout.addWidget(top_frame)
        
        self.weight_histogram = WeightHistogram()
        layout.addWidget(self.weight_histogram, 1)
        
        stats_group = QGroupBox('权重统计')
        stats_layout = QFormLayout(stats_group)
        stats_layout.setSpacing(10)
        stats_layout.setLabelAlignment(Qt.AlignRight)
        
        self.weight_shape_label = self._create_value_label()
        stats_layout.addRow('形状:', self.weight_shape_label)
        
        self.weight_min_label = self._create_value_label()
        stats_layout.addRow('最小值:', self.weight_min_label)
        
        self.weight_max_label = self._create_value_label()
        stats_layout.addRow('最大值:', self.weight_max_label)
        
        self.weight_mean_label = self._create_value_label()
        stats_layout.addRow('均值:', self.weight_mean_label)
        
        self.weight_std_label = self._create_value_label()
        stats_layout.addRow('标准差:', self.weight_std_label)
        
        self.weight_zeros_label = self._create_value_label()
        stats_layout.addRow('零值数量:', self.weight_zeros_label)
        
        layout.addWidget(stats_group)
        
        self.tabs.addTab(tab, '权重参数')
    
    def setup_training_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.training_curve_widget = TrainingCurveWidget()
        layout.addWidget(self.training_curve_widget, 2)
        
        stats_group = QGroupBox('训练统计')
        stats_layout = QFormLayout(stats_group)
        stats_layout.setSpacing(10)
        stats_layout.setLabelAlignment(Qt.AlignRight)
        
        self.train_epochs_label = self._create_value_label()
        stats_layout.addRow('训练轮次:', self.train_epochs_label)
        
        self.best_train_acc_label = self._create_value_label(highlight=True)
        stats_layout.addRow('最佳训练准确率:', self.best_train_acc_label)
        
        self.best_val_acc_label = self._create_value_label(highlight=True)
        stats_layout.addRow('最佳验证准确率:', self.best_val_acc_label)
        
        self.final_loss_label = self._create_value_label()
        stats_layout.addRow('最终损失:', self.final_loss_label)
        
        self.final_lr_label = self._create_value_label()
        stats_layout.addRow('最终学习率:', self.final_lr_label)
        
        layout.addWidget(stats_group)
        
        log_group = QGroupBox('训练日志')
        log_layout = QVBoxLayout(log_group)
        
        self.training_log_display = QTextEdit()
        self.training_log_display.setReadOnly(True)
        self.training_log_display.setMinimumHeight(100)
        self.training_log_display.setMaximumHeight(200)
        log_layout.addWidget(self.training_log_display)
        
        layout.addWidget(log_group)
        
        self.tabs.addTab(tab, '训练曲线')
    
    def setup_raw_data_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.raw_data_tree = QTreeWidget()
        self.raw_data_tree.setHeaderLabels(['键', '类型', '大小/值'])
        self.raw_data_tree.setColumnWidth(0, 250)
        self.raw_data_tree.setColumnWidth(1, 120)
        layout.addWidget(self.raw_data_tree, 1)
        
        detail_group = QGroupBox('详细内容')
        detail_layout = QVBoxLayout(detail_group)
        
        self.raw_content_display = QTextEdit()
        self.raw_content_display.setReadOnly(True)
        self.raw_content_display.setMinimumHeight(120)
        self.raw_content_display.setMaximumHeight(180)
        detail_layout.addWidget(self.raw_content_display)
        
        layout.addWidget(detail_group)
        
        self.raw_data_tree.itemClicked.connect(self.show_raw_item_content)
        
        self.tabs.addTab(tab, '原始数据')
    
    def setup_export_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        export_group = QGroupBox('导出选项')
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(15)
        
        btn_layout1 = QHBoxLayout()
        btn_layout1.setSpacing(15)
        
        export_structure_btn = QPushButton('导出网络结构')
        export_structure_btn.clicked.connect(self.export_structure)
        btn_layout1.addWidget(export_structure_btn)
        
        export_weights_btn = QPushButton('导出权重统计')
        export_weights_btn.clicked.connect(self.export_weights_stats)
        btn_layout1.addWidget(export_weights_btn)
        
        export_classes_btn = QPushButton('导出类别列表')
        export_classes_btn.clicked.connect(self.export_classes)
        btn_layout1.addWidget(export_classes_btn)
        
        export_layout.addLayout(btn_layout1)
        
        btn_layout2 = QHBoxLayout()
        btn_layout2.setSpacing(15)
        
        export_summary_btn = QPushButton('导出完整报告')
        export_summary_btn.clicked.connect(self.export_summary)
        btn_layout2.addWidget(export_summary_btn)
        
        export_raw_btn = QPushButton('导出原始数据')
        export_raw_btn.clicked.connect(self.export_raw_data)
        btn_layout2.addWidget(export_raw_btn)
        
        export_layout.addLayout(btn_layout2)
        
        layout.addWidget(export_group)
        
        log_group = QGroupBox('导出日志')
        log_layout = QVBoxLayout(log_group)
        
        self.export_log = QTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setMinimumHeight(150)
        log_layout.addWidget(self.export_log)
        
        layout.addWidget(log_group, 1)
        
        self.tabs.addTab(tab, '导出')
    
    def _create_value_label(self, highlight=False):
        label = QLabel('-')
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if highlight:
            label.setStyleSheet(f'''
                color: {STYLES['text_highlight']}; 
                font-weight: bold;
                padding: 2px 8px;
                background-color: rgba(78, 204, 163, 0.1);
                border-radius: 4px;
            ''')
        else:
            label.setStyleSheet(f'color: {STYLES["text_primary"]};')
        return label
    
    def browse_model(self):
        default_models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models'
        )
        
        path, _ = QFileDialog.getOpenFileName(
            self, '选择模型文件',
            default_models_dir,
            '模型文件 (*.pth);;所有文件 (*)'
        )
        
        if path:
            self.load_model(path)
    
    def load_model(self, path):
        try:
            self.status_label.setText('正在加载模型...')
            self.status_label.setStyleSheet(f'''
                color: {STYLES['warning']}; 
                padding: 10px 15px;
                background-color: {STYLES['card_bg']};
                border-radius: 6px;
            ''')
            QApplication.processEvents()
            
            self.model_path = path
            
            file_stat = os.stat(path)
            file_size = file_stat.st_size
            create_time = file_stat.st_ctime
            
            self.checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            self.analyze_model()
            
            self.file_path_label.setText(path)
            self.file_size_label.setText(self._format_size(file_size))
            self.create_time_label.setText(self._format_time(create_time))
            
            self.status_label.setText(f'加载完成: {os.path.basename(path)}')
            self.status_label.setStyleSheet(f'''
                color: {STYLES['success']}; 
                padding: 10px 15px;
                background-color: {STYLES['card_bg']};
                border-radius: 6px;
            ''')
            
        except Exception as e:
            QMessageBox.warning(self, '错误', f'加载模型失败: {str(e)}')
            self.status_label.setText('加载失败')
            self.status_label.setStyleSheet(f'''
                color: {STYLES['error']}; 
                padding: 10px 15px;
                background-color: {STYLES['card_bg']};
                border-radius: 6px;
            ''')
    
    def analyze_model(self):
        if not self.checkpoint:
            return
        
        args = self.checkpoint.get('args', {})
        model_name = args.get('model_name', '未知')
        input_size = args.get('input_size', 128)
        class_names = self.checkpoint.get('class_names', [])
        
        if not class_names:
            classes_path = self.model_path.replace('.pth', '_classes.json')
            if os.path.exists(classes_path):
                with open(classes_path, 'r', encoding='utf-8') as f:
                    class_names = json.load(f)
        
        self.model_type_label.setText(model_name)
        self.input_size_label.setText(f'{input_size} × {input_size}')
        self.class_count_label.setText(str(len(class_names)))
        
        self.class_list.clear()
        for i, name in enumerate(class_names):
            self.class_list.append(f'[{i:3d}] {name}')
        
        state_dict = self.checkpoint.get('model_state_dict', self.checkpoint.get('state_dict', {}))
        
        if state_dict:
            self.analyze_state_dict(state_dict)
        
        self.analyze_training_history()
        self.analyze_raw_data()
    
    def analyze_state_dict(self, state_dict):
        total_params = 0
        layer_info = []
        layer_weights = {}
        
        conv_count = 0
        bn_count = 0
        linear_count = 0
        pool_count = 0
        
        for name, param in state_dict.items():
            param_count = param.numel()
            total_params += param_count
            
            layer_type = self._get_layer_type(name)
            
            if 'conv' in name.lower() or 'Conv' in layer_type:
                conv_count += 1
            elif 'bn' in name.lower() or 'batch' in name.lower() or 'BatchNorm' in layer_type:
                bn_count += 1
            elif 'fc' in name.lower() or 'linear' in name.lower() or 'Linear' in layer_type:
                linear_count += 1
            elif 'pool' in name.lower() or 'Pool' in layer_type:
                pool_count += 1
            
            layer_info.append({
                'name': name,
                'type': layer_type,
                'params': param_count,
                'shape': list(param.shape)
            })
            
            layer_weights[name] = param
        
        self.layer_info = layer_info
        self.layer_weights = layer_weights
        
        self.total_params_label.setText(f'{total_params:,} ({total_params/1e6:.2f}M)')
        self.trainable_params_label.setText(f'{total_params:,}')
        self.model_size_label.setText(f'{total_params * 4 / 1024 / 1024:.2f} MB (FP32)')
        
        self.layer_count_label.setText(str(len(layer_info)))
        self.conv_count_label.setText(str(conv_count // 2 if conv_count > 0 else 0))
        self.bn_count_label.setText(str(bn_count // 2 if bn_count > 0 else 0))
        self.linear_count_label.setText(str(linear_count // 2 if linear_count > 0 else 0))
        self.pool_count_label.setText(str(pool_count))
        
        self.update_layer_tree(layer_info)
        self.update_layer_combo(layer_info)
        self.layer_visualizer.set_layers(layer_info[:50])
    
    def _get_layer_type(self, name):
        name_lower = name.lower()
        if 'weight' in name_lower:
            if 'conv' in name_lower:
                return 'Conv2d'
            elif 'bn' in name_lower or 'batch' in name_lower:
                return 'BatchNorm'
            elif 'fc' in name_lower or 'linear' in name_lower:
                return 'Linear'
        elif 'bias' in name_lower:
            return 'Bias'
        elif 'running' in name_lower:
            return 'RunningStats'
        return 'Other'
    
    def update_layer_tree(self, layer_info):
        self.layer_tree.clear()
        
        root = QTreeWidgetItem(['模型结构', '', '', ''])
        self.layer_tree.addTopLevelItem(root)
        
        current_parent = root
        current_prefix = ''
        
        for layer in layer_info:
            name = layer['name']
            parts = name.split('.')
            
            parent = root
            for i, part in enumerate(parts[:-1]):
                prefix = '.'.join(parts[:i+1])
                found = None
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    if child.data(0, Qt.UserRole) == prefix:
                        found = child
                        break
                
                if not found:
                    found = QTreeWidgetItem([part, '', '', ''])
                    found.setData(0, Qt.UserRole, prefix)
                    parent.addChild(found)
                
                parent = found
            
            item = QTreeWidgetItem([
                parts[-1],
                layer['type'],
                f"{layer['params']:,}",
                str(layer['shape'])
            ])
            item.setData(0, Qt.UserRole, name)
            parent.addChild(item)
        
        self.layer_tree.expandAll()
    
    def update_layer_combo(self, layer_info):
        self.weight_layer_combo.clear()
        
        for layer in layer_info:
            if 'weight' in layer['name'].lower():
                self.weight_layer_combo.addItem(layer['name'])
    
    def on_layer_selected(self, item):
        name = item.data(0, Qt.UserRole)
        if name and name in self.layer_weights:
            self.weight_layer_combo.setCurrentText(name)
    
    def on_weight_layer_changed(self, name):
        if not name or name not in self.layer_weights:
            return
        
        weight = self.layer_weights[name]
        
        self.weight_histogram.plot_weight(weight, name)
        
        weight_np = weight.flatten().cpu().numpy()
        
        self.weight_shape_label.setText(str(list(weight.shape)))
        self.weight_min_label.setText(f'{weight_np.min():.6f}')
        self.weight_max_label.setText(f'{weight_np.max():.6f}')
        self.weight_mean_label.setText(f'{weight_np.mean():.6f}')
        self.weight_std_label.setText(f'{weight_np.std():.6f}')
        self.weight_zeros_label.setText(f'{(weight_np == 0).sum():,} ({(weight_np == 0).sum() / len(weight_np) * 100:.2f}%)')
        
        self.layer_detail_table.setRowCount(0)
        
        details = [
            ('层名称', name),
            ('数据类型', str(weight.dtype)),
            ('设备', str(weight.device)),
            ('是否需要梯度', str(weight.requires_grad)),
            ('参数数量', f'{weight.numel():,}'),
        ]
        
        for key, value in details:
            row = self.layer_detail_table.rowCount()
            self.layer_detail_table.insertRow(row)
            self.layer_detail_table.setItem(row, 0, QTableWidgetItem(key))
            self.layer_detail_table.setItem(row, 1, QTableWidgetItem(str(value)))
    
    def analyze_training_history(self):
        training_history = self.checkpoint.get('training_history', {})
        
        if not training_history and self.model_path:
            history_path = self.model_path.replace('.pth', '_history.json')
            if os.path.exists(history_path):
                try:
                    import json
                    with open(history_path, 'r', encoding='utf-8') as f:
                        training_history = json.load(f)
                except Exception:
                    pass
        
        if not training_history:
            self.training_curve_widget.plot_curves([], [], [], [], [], [])
            return
        
        train_loss = training_history.get('train_loss', [])
        val_loss = training_history.get('val_loss', [])
        train_acc = training_history.get('train_acc', [])
        val_acc = training_history.get('val_acc', [])
        lr_history = training_history.get('lr', [])
        
        num_epochs = max(len(train_loss), len(val_loss), len(train_acc), len(val_acc))
        epochs = training_history.get('epochs', list(range(1, num_epochs + 1)))
        
        if not epochs and num_epochs > 0:
            epochs = list(range(1, num_epochs + 1))
        
        self.training_curve_widget.plot_curves(
            epochs, train_loss, val_loss, train_acc, val_acc, lr_history
        )
        
        if epochs:
            self.train_epochs_label.setText(str(len(epochs)))
        
        if train_acc:
            self.best_train_acc_label.setText(f'{max(train_acc):.2f}%')
        
        if val_acc:
            self.best_val_acc_label.setText(f'{max(val_acc):.2f}%')
        
        if train_loss:
            self.final_loss_label.setText(f'{train_loss[-1]:.4f}')
        
        if lr_history:
            self.final_lr_label.setText(f'{lr_history[-1]:.6f}')
        
        training_logs = self.checkpoint.get('training_logs', [])
        if training_logs:
            self.training_log_display.setText('\n'.join(training_logs[-100:]))
        elif self.model_path:
            log_path = self.model_path.replace('.pth', '_log.txt')
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    lines = log_content.strip().split('\n')
                    self.training_log_display.setText('\n'.join(lines[-100:]))
                except Exception:
                    self.training_log_display.setText('无法读取日志文件')
            else:
                self.training_log_display.setText('无训练日志')
        else:
            self.training_log_display.setText('无训练日志')
    
    def analyze_raw_data(self):
        self.raw_data_tree.clear()
        
        self._add_dict_to_tree(self.checkpoint, self.raw_data_tree.invisibleRootItem())
    
    def _add_dict_to_tree(self, data, parent, prefix=''):
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f'{prefix}.{key}' if prefix else key
                
                if isinstance(value, (dict, list)) and len(str(value)) > 100:
                    type_str = type(value).__name__
                    size_str = f'长度: {len(value)}'
                    item = QTreeWidgetItem([str(key), type_str, size_str])
                    item.setData(0, Qt.UserRole, full_key)
                    parent.addChild(item)
                    self._add_dict_to_tree(value, item, full_key)
                elif isinstance(value, torch.Tensor):
                    item = QTreeWidgetItem([
                        str(key),
                        'Tensor',
                        f'{list(value.shape)} ({value.dtype})'
                    ])
                    item.setData(0, Qt.UserRole, full_key)
                    parent.addChild(item)
                else:
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:50] + '...'
                    item = QTreeWidgetItem([str(key), type(value).__name__, value_str])
                    item.setData(0, Qt.UserRole, full_key)
                    parent.addChild(item)
        elif isinstance(data, list):
            for i, value in enumerate(data):
                full_key = f'{prefix}[{i}]'
                
                if isinstance(value, (dict, list)) and len(str(value)) > 100:
                    type_str = type(value).__name__
                    size_str = f'长度: {len(value)}'
                    item = QTreeWidgetItem([f'[{i}]', type_str, size_str])
                    item.setData(0, Qt.UserRole, full_key)
                    parent.addChild(item)
                    self._add_dict_to_tree(value, item, full_key)
                elif isinstance(value, torch.Tensor):
                    item = QTreeWidgetItem([
                        f'[{i}]',
                        'Tensor',
                        f'{list(value.shape)} ({value.dtype})'
                    ])
                    item.setData(0, Qt.UserRole, full_key)
                    parent.addChild(item)
                else:
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:50] + '...'
                    item = QTreeWidgetItem([f'[{i}]', type(value).__name__, value_str])
                    item.setData(0, Qt.UserRole, full_key)
                    parent.addChild(item)
    
    def show_raw_item_content(self, item):
        key = item.data(0, Qt.UserRole)
        if not key:
            return
        
        try:
            value = self._get_nested_value(self.checkpoint, key)
            
            if isinstance(value, torch.Tensor):
                self.raw_content_display.setText(f'Tensor:\n形状: {list(value.shape)}\n数据类型: {value.dtype}\n\n前100个值:\n{value.flatten()[:100]}')
            elif isinstance(value, (dict, list)):
                self.raw_content_display.setText(json.dumps(value, ensure_ascii=False, indent=2)[:5000])
            else:
                self.raw_content_display.setText(str(value)[:5000])
        except Exception as e:
            self.raw_content_display.setText(f'无法显示: {str(e)}')
    
    def _get_nested_value(self, data, key):
        parts = key.replace('[', '.').replace(']', '').split('.')
        
        for part in parts:
            if part.isdigit():
                data = data[int(part)]
            else:
                data = data[part]
        
        return data
    
    def export_structure(self):
        if not self.layer_info:
            QMessageBox.warning(self, '提示', '请先加载模型')
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, '导出网络结构',
            'model_structure.txt',
            '文本文件 (*.txt)'
        )
        
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('=' * 60 + '\n')
                f.write('网络结构报告\n')
                f.write('=' * 60 + '\n\n')
                
                for layer in self.layer_info:
                    f.write(f"层: {layer['name']}\n")
                    f.write(f"  类型: {layer['type']}\n")
                    f.write(f"  参数量: {layer['params']:,}\n")
                    f.write(f"  形状: {layer['shape']}\n\n")
            
            self.export_log.append(f'[完成] 网络结构已导出到: {path}')
    
    def export_weights_stats(self):
        if not self.layer_weights:
            QMessageBox.warning(self, '提示', '请先加载模型')
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, '导出权重统计',
            'weights_stats.csv',
            'CSV文件 (*.csv)'
        )
        
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('层名称,类型,参数量,最小值,最大值,均值,标准差\n')
                
                for name, weight in self.layer_weights.items():
                    weight_np = weight.flatten().cpu().numpy()
                    layer_type = self._get_layer_type(name)
                    f.write(f'{name},{layer_type},{weight.numel()},{weight_np.min():.6f},{weight_np.max():.6f},{weight_np.mean():.6f},{weight_np.std():.6f}\n')
            
            self.export_log.append(f'[完成] 权重统计已导出到: {path}')
    
    def export_classes(self):
        class_names = self.checkpoint.get('class_names', [])
        
        if not class_names:
            QMessageBox.warning(self, '提示', '模型中没有类别信息')
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, '导出类别列表',
            'class_names.txt',
            '文本文件 (*.txt)'
        )
        
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                for i, name in enumerate(class_names):
                    f.write(f'[{i}] {name}\n')
            
            self.export_log.append(f'[完成] 类别列表已导出到: {path}')
    
    def export_summary(self):
        if not self.checkpoint:
            QMessageBox.warning(self, '提示', '请先加载模型')
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, '导出完整报告',
            'model_report.txt',
            '文本文件 (*.txt)'
        )
        
        if path:
            args = self.checkpoint.get('args', {})
            class_names = self.checkpoint.get('class_names', [])
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write('=' * 60 + '\n')
                f.write('模型完整分析报告\n')
                f.write('=' * 60 + '\n\n')
                
                f.write('【基本信息】\n')
                f.write(f'  文件路径: {self.model_path}\n')
                f.write(f'  模型类型: {args.get("model_name", "未知")}\n')
                f.write(f'  输入尺寸: {args.get("input_size", "未知")}\n')
                f.write(f'  类别数量: {len(class_names)}\n\n')
                
                total_params = sum(p.numel() for p in self.layer_weights.values())
                f.write('【参数统计】\n')
                f.write(f'  总参数量: {total_params:,} ({total_params/1e6:.2f}M)\n')
                f.write(f'  总层数: {len(self.layer_info)}\n\n')
                
                f.write('【类别列表】\n')
                for i, name in enumerate(class_names):
                    f.write(f'  [{i:3d}] {name}\n')
                
                f.write('\n【训练参数】\n')
                for key, value in args.items():
                    f.write(f'  {key}: {value}\n')
            
            self.export_log.append(f'[完成] 完整报告已导出到: {path}')
    
    def export_raw_data(self):
        if not self.checkpoint:
            QMessageBox.warning(self, '提示', '请先加载模型')
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, '导出原始数据',
            'model_raw_data.json',
            'JSON文件 (*.json)'
        )
        
        if path:
            def convert_to_serializable(obj):
                if isinstance(obj, torch.Tensor):
                    return f'<Tensor {list(obj.shape)}>'
                elif isinstance(obj, (np.ndarray, np.integer, np.floating)):
                    return str(obj)
                return str(obj)
            
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self.checkpoint, f, ensure_ascii=False, indent=2, default=convert_to_serializable)
                self.export_log.append(f'[完成] 原始数据已导出到: {path}')
            except Exception as e:
                self.export_log.append(f'[错误] 导出失败: {str(e)}')
    
    def _format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f'{size:.2f} {unit}'
            size /= 1024
        return f'{size:.2f} TB'
    
    def _format_time(self, timestamp):
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont('Microsoft YaHei UI', 10)
    app.setFont(font)
    
    window = ModelAnalyzer()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
