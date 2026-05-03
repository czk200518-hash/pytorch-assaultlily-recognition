import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QTextEdit, QGroupBox, QFormLayout, QMessageBox, QSplitter,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTranslator, QLibraryInfo, QLocale
from PyQt5.QtGui import QFont, QPixmap

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / 'config.json'

class PredictThread(QThread):
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)

    def __init__(self, model, class_names, device, image_path, input_size):
        super().__init__()
        self.model = model
        self.class_names = class_names
        self.device = device
        self.image_path = image_path
        self.input_size = input_size

    def run(self):
        try:
            from predict import predict_image
            
            self.log_signal.emit('[识别] 使用整图识别模式')
            
            result = predict_image(
                self.model, self.image_path, self.class_names,
                self.device, input_size=self.input_size,
            )
            self.result_signal.emit({
                'result': result,
                'image_path': self.image_path,
                'class_names': self.class_names,
            })
            self.log_signal.emit(f'[完成] 识别结束')
        except Exception as e:
            self.log_signal.emit(f'[错误] {traceback.format_exc()}')

class RecognizeTool(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.class_names = []
        self.device = None
        self.input_size = 128
        self.predict_thread = None
        self.setup_ui()
        self.load_config_and_auto_load()

    def setup_ui(self):
        self.setWindowTitle('角色识别工具')
        self.setMinimumSize(750, 600)
        self.resize(900, 720)

        layout = QVBoxLayout(self)

        self._setup_model_section(layout)
        self._setup_model_info(layout)
        self._setup_image_section(layout)
        self._setup_content_area(layout)
        self._setup_log_area(layout)

    def _setup_model_section(self, layout):
        model_group = QGroupBox('① 加载模型')
        model_layout = QHBoxLayout()
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText('选择训练好的 .pth 模型文件')
        
        model_browse = QPushButton('浏览...')
        model_browse.clicked.connect(self.browse_model)
        
        self.load_btn = QPushButton('加载模型')
        self.load_btn.setStyleSheet(
            'QPushButton { background-color: #FF9800; color: white; font-weight: bold; border-radius: 4px; padding: 6px 16px; }'
            'QPushButton:hover { background-color: #F57C00; }'
            'QPushButton:disabled { background-color: #cccccc; }'
        )
        self.load_btn.clicked.connect(self.load_model_file)
        
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(model_browse)
        model_layout.addWidget(self.load_btn)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

    def _setup_model_info(self, layout):
        model_info_group = QGroupBox('模型信息')
        info_layout = QHBoxLayout()
        
        self.model_info_label = QLabel('尚未加载模型')
        self.model_info_label.setStyleSheet('color: #888; font-size: 12px; padding: 4px;')
        
        info_layout.addWidget(self.model_info_label)
        info_layout.addStretch()
        model_info_group.setLayout(info_layout)
        layout.addWidget(model_info_group)

        size_group = QGroupBox('图片尺寸信息')
        size_layout = QHBoxLayout()
        
        self.input_size_label = QLabel('输入尺寸: —')
        self.input_size_label.setStyleSheet('color: #2196F3; font-size: 12px; font-weight: bold; padding: 2px 8px;')
        
        size_layout.addWidget(self.input_size_label)
        size_layout.addStretch()
        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

    def _setup_image_section(self, layout):
        image_group = QGroupBox('② 选择识别图片')
        image_layout = QHBoxLayout()
        
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText('选择一张角色图片进行识别')
        
        image_browse = QPushButton('浏览...')
        image_browse.clicked.connect(self.browse_image)
        
        self.predict_btn = QPushButton('▶ 开始识别')
        self.predict_btn.setMinimumHeight(36)
        self.predict_btn.setStyleSheet(
            'QPushButton { background-color: #4CAF50; color: white; font-size: 14px; font-weight: bold; border-radius: 4px; }'
            'QPushButton:hover { background-color: #45a049; }'
            'QPushButton:disabled { background-color: #cccccc; }'
        )
        self.predict_btn.clicked.connect(self.start_prediction)
        
        image_layout.addWidget(self.image_path_edit)
        image_layout.addWidget(image_browse)
        image_layout.addWidget(self.predict_btn)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

    def _setup_content_area(self, layout):
        content_splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel('图片预览区域\n\n选择图片后自动显示')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(250, 250)
        self.image_label.setStyleSheet(
            'QLabel { border: 2px dashed #aaa; color: #888; font-size: 14px; }'
        )
        left_layout.addWidget(self.image_label)
        content_splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        result_group = QGroupBox('③ 识别结果')
        result_form = QFormLayout()

        self.predicted_label = QLabel('—')
        self.predicted_label.setFont(QFont('Microsoft YaHei', 18, QFont.Bold))
        self.predicted_label.setStyleSheet('color: #2196F3; padding: 4px;')

        self.confidence_label = QLabel('—')
        self.confidence_label.setFont(QFont('Microsoft YaHei', 13))
        self.confidence_label.setStyleSheet('color: #4CAF50; padding: 4px;')

        self.top3_label = QLabel('—')
        self.top3_label.setFont(QFont('Consolas', 11))
        self.top3_label.setWordWrap(True)
        self.top3_label.setStyleSheet('padding: 4px;')

        result_form.addRow('角色:', self.predicted_label)
        result_form.addRow('置信度:', self.confidence_label)
        result_form.addRow('Top-3:', self.top3_label)
        result_group.setLayout(result_form)
        right_layout.addWidget(result_group)
        right_layout.addStretch()
        content_splitter.addWidget(right_panel)

        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)

        layout.addWidget(content_splitter, 1)

    def _setup_log_area(self, layout):
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setFont(QFont('Consolas', 9))
        self.log_text.setStyleSheet('QTextEdit { background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #555; }')
        layout.addWidget(self.log_text)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择模型文件', str(BASE_DIR.parent / 'models'),
            '模型文件 (*.pth);;所有文件 (*)'
        )
        if path:
            self.model_path_edit.setText(path)

    def load_model_file(self):
        model_path = self.model_path_edit.text().strip()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, '提示', '请选择有效的 .pth 模型文件')
            return

        try:
            import torch
            from model import create_model
            
            self.log_text.clear()
            self.append_log(f'[加载] 正在加载模型: {os.path.basename(model_path)}')

            checkpoint = torch.load(model_path, map_location='cpu')
            class_names = checkpoint.get('class_names', [])
            if not class_names:
                classes_path = model_path.replace('.pth', '_classes.json')
                if os.path.exists(classes_path):
                    with open(classes_path, 'r', encoding='utf-8') as f:
                        class_names = json.load(f)

            args = checkpoint.get('args', {})
            model_name = args.get('model_name', 'standard')
            self.input_size = args.get('input_size', 128)
            num_classes = len(class_names)

            if num_classes == 0:
                raise ValueError('无法确定类别数量')

            model = create_model(model_name, num_classes, self.input_size)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = model.to(self.device)
            self.model.eval()
            self.class_names = class_names

            total_params = sum(p.numel() for p in model.parameters())
            info_text = (
                f'模型: {model_name}  |  输入尺寸: {self.input_size}×{self.input_size}  |  '
                f'分类数: {num_classes}  |  参数量: {total_params/1e6:.1f}M  |  '
                f'设备: {"GPU" if self.device.type=="cuda" else "CPU"}'
            )
            self.model_info_label.setText(info_text)
            self.input_size_label.setText(f'输入尺寸: {self.input_size}×{self.input_size}')
            self.append_log(f'[加载] 完成! 输入尺寸自动设为 {self.input_size}')
            self.append_log(f'[加载] 分类: {class_names}')
            self.save_config(model_path)

        except Exception as e:
            QMessageBox.warning(self, '错误', f'加载模型失败: {str(e)}')
            self.append_log(f'[错误] {traceback.format_exc()}')

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择图片',
            '', '图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)'
        )
        if path:
            self.image_path_edit.setText(path)
            self.show_image_preview(path)

            if self.model is not None:
                self.start_prediction()

    def show_image_preview(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            self.image_label.setStyleSheet('')

    def start_prediction(self):
        if self.model is None:
            QMessageBox.warning(self, '提示', '请先加载模型')
            return

        image_path = self.image_path_edit.text().strip()
        if not image_path or not os.path.isfile(image_path):
            QMessageBox.warning(self, '提示', '请先选择有效的图片文件')
            return

        self.predicted_label.setText('识别中...')
        self.confidence_label.setText('—')
        self.top3_label.setText('—')
        self.predict_btn.setEnabled(False)

        self.predict_thread = PredictThread(
            self.model, self.class_names, self.device,
            image_path, self.input_size,
        )
        self.predict_thread.log_signal.connect(self.append_log)
        self.predict_thread.result_signal.connect(self.show_result)
        self.predict_thread.finished.connect(lambda: self.predict_btn.setEnabled(True))
        self.predict_thread.start()

    def append_log(self, msg):
        ts = datetime.now().strftime('[%H:%M:%S]')
        self.log_text.append(f'{ts} {msg}')
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def show_result(self, data):
        r = data['result']
        self.predicted_label.setText(r['predicted_class'])
        self.confidence_label.setText(f'{r["confidence"]*100:.1f}%')
        top3_text = '\n'.join(
            f'  [{i+1}] {name} — {prob*100:.1f}%'
            for i, (name, prob) in enumerate(r['top3'])
        )
        self.top3_label.setText(top3_text)

    def save_config(self, model_path=None):
        config = {}
        if os.path.exists(str(CONFIG_PATH)):
            try:
                with open(str(CONFIG_PATH), 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except Exception:
                pass
        if model_path:
            config['last_model_path'] = model_path
        try:
            with open(str(CONFIG_PATH), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_config_and_auto_load(self):
        if not os.path.exists(str(CONFIG_PATH)):
            return
        try:
            with open(str(CONFIG_PATH), 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            return
        model_path = config.get('last_model_path', '')
        if model_path and os.path.isfile(model_path):
            self.model_path_edit.setText(model_path)
            self.load_model_file()

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

    window = RecognizeTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
