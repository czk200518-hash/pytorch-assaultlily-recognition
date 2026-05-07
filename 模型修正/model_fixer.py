import sys
import os
import torch
import json
from datetime import datetime

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout, QSplitter,
    QFrame, QAbstractItemView, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor


class EditDialog(QDialog):
    def __init__(self, current_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle('修改类别名称')
        self.setMinimumWidth(400)
        self.setup_ui(current_name)
    
    def setup_ui(self, current_name):
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel('当前名称:'))
        self.current_label = QLabel(current_name)
        self.current_label.setStyleSheet('font-weight: bold; color: #666; padding: 5px;')
        layout.addWidget(self.current_label)
        
        layout.addWidget(QLabel('新名称:'))
        self.new_name_edit = QLineEdit(current_name)
        self.new_name_edit.selectAll()
        layout.addWidget(self.new_name_edit)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_new_name(self):
        return self.new_name_edit.text().strip()


class ModelFixer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.checkpoint = None
        self.class_names = []
        self.model_path = None
        self.modified = False
        self.original_class_names = []
        
        self.setWindowTitle('模型修正工具 - 类别名称修改')
        self.setMinimumSize(900, 600)
        self.setup_ui()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        
        model_group = QGroupBox('模型文件')
        model_layout = QHBoxLayout(model_group)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText('请选择要修改的模型文件 (.pth)')
        model_layout.addWidget(self.model_path_edit)
        
        browse_btn = QPushButton('浏览...')
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_btn)
        
        layout.addWidget(model_group)
        
        info_group = QGroupBox('模型信息')
        info_layout = QFormLayout(info_group)
        
        self.model_type_label = QLabel('-')
        self.model_type_label.setStyleSheet('font-weight: bold;')
        info_layout.addRow('模型类型:', self.model_type_label)
        
        self.input_size_label = QLabel('-')
        info_layout.addRow('输入尺寸:', self.input_size_label)
        
        self.class_count_label = QLabel('-')
        self.class_count_label.setStyleSheet('font-weight: bold; color: #2196F3;')
        info_layout.addRow('类别数量:', self.class_count_label)
        
        self.modified_label = QLabel('未修改')
        self.modified_label.setStyleSheet('color: #666;')
        info_layout.addRow('修改状态:', self.modified_label)
        
        layout.addWidget(info_group)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        left_layout.addWidget(QLabel('类别列表 (双击修改):'))
        
        self.class_list = QListWidget()
        self.class_list.setAlternatingRowColors(True)
        self.class_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.class_list.itemDoubleClicked.connect(self.edit_class_name)
        self.class_list.currentRowChanged.connect(self.on_select_class)
        left_layout.addWidget(self.class_list)
        
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel('搜索:'))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText('输入关键字筛选类别...')
        self.search_edit.textChanged.connect(self.filter_classes)
        search_layout.addWidget(self.search_edit)
        left_layout.addLayout(search_layout)
        
        splitter.addWidget(left_frame)
        
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        right_layout.addWidget(QLabel('修改操作:'))
        
        edit_group = QGroupBox('选中的类别')
        edit_layout = QVBoxLayout(edit_group)
        
        self.selected_index_label = QLabel('索引: -')
        edit_layout.addWidget(self.selected_index_label)
        
        self.selected_name_label = QLabel('名称: -')
        self.selected_name_label.setWordWrap(True)
        edit_layout.addWidget(self.selected_name_label)
        
        edit_btn_layout = QHBoxLayout()
        
        self.edit_btn = QPushButton('修改名称')
        self.edit_btn.clicked.connect(self.edit_selected_class)
        self.edit_btn.setEnabled(False)
        edit_btn_layout.addWidget(self.edit_btn)
        
        edit_layout.addLayout(edit_btn_layout)
        right_layout.addWidget(edit_group)
        
        batch_group = QGroupBox('批量操作')
        batch_layout = QVBoxLayout(batch_group)
        
        batch_edit_layout = QFormLayout()
        
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText('要查找的文本')
        batch_edit_layout.addRow('查找:', self.find_edit)
        
        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText('替换为')
        batch_edit_layout.addRow('替换:', self.replace_edit)
        
        batch_layout.addLayout(batch_edit_layout)
        
        batch_btn_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton('预览')
        self.preview_btn.clicked.connect(self.preview_batch_replace)
        batch_btn_layout.addWidget(self.preview_btn)
        
        self.apply_batch_btn = QPushButton('应用')
        self.apply_batch_btn.clicked.connect(self.apply_batch_replace)
        batch_btn_layout.addWidget(self.apply_batch_btn)
        
        batch_layout.addLayout(batch_btn_layout)
        right_layout.addWidget(batch_group)
        
        right_layout.addStretch()
        
        splitter.addWidget(right_frame)
        
        splitter.setSizes([600, 300])
        layout.addWidget(splitter)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.save_btn = QPushButton('保存修改')
        self.save_btn.clicked.connect(self.save_model)
        self.save_btn.setEnabled(False)
        self.save_btn.setFixedWidth(120)
        btn_layout.addWidget(self.save_btn)
        
        self.save_as_btn = QPushButton('另存为...')
        self.save_as_btn.clicked.connect(self.save_model_as)
        self.save_as_btn.setEnabled(False)
        self.save_as_btn.setFixedWidth(120)
        btn_layout.addWidget(self.save_as_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.log_label = QLabel('请选择模型文件开始')
        self.log_label.setStyleSheet('color: #666; padding: 5px;')
        layout.addWidget(self.log_label)
    
    def browse_model(self):
        if self.modified:
            reply = QMessageBox.question(
                self, '确认', '当前有未保存的修改，确定要加载新模型吗？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
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
            self.log_label.setText('正在加载模型...')
            self.log_label.setStyleSheet('color: #FF9800; padding: 5px;')
            QApplication.processEvents()
            
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            class_names = checkpoint.get('class_names', [])
            if not class_names:
                classes_path = path.replace('.pth', '_classes.json')
                if os.path.exists(classes_path):
                    with open(classes_path, 'r', encoding='utf-8') as f:
                        class_names = json.load(f)
            
            if not class_names:
                QMessageBox.warning(self, '错误', '模型文件中没有找到类别信息')
                return
            
            self.checkpoint = checkpoint
            self.class_names = list(class_names)
            self.original_class_names = list(class_names)
            self.model_path = path
            self.modified = False
            
            args = checkpoint.get('args', {})
            model_name = args.get('model_name', '未知')
            input_size = args.get('input_size', 128)
            
            self.model_type_label.setText(model_name)
            self.input_size_label.setText(f'{input_size}×{input_size}')
            self.class_count_label.setText(str(len(self.class_names)))
            self.modified_label.setText('未修改')
            self.modified_label.setStyleSheet('color: #666;')
            
            self.update_class_list()
            
            self.save_btn.setEnabled(False)
            self.save_as_btn.setEnabled(True)
            
            self.log_label.setText(f'已加载: {os.path.basename(path)}')
            self.log_label.setStyleSheet('color: #4CAF50; padding: 5px;')
            
        except Exception as e:
            QMessageBox.warning(self, '错误', f'加载模型失败: {str(e)}')
            self.log_label.setText('加载失败')
            self.log_label.setStyleSheet('color: #F44336; padding: 5px;')
    
    def update_class_list(self, filter_text=''):
        self.class_list.clear()
        
        for i, name in enumerate(self.class_names):
            if filter_text and filter_text.lower() not in name.lower():
                continue
            
            item = QListWidgetItem(f'[{i}] {name}')
            item.setData(Qt.UserRole, i)
            
            if name != self.original_class_names[i]:
                item.setForeground(QColor('#F44336'))
            
            self.class_list.addItem(item)
    
    def filter_classes(self, text):
        self.update_class_list(text)
    
    def on_select_class(self, row):
        if row < 0:
            self.selected_index_label.setText('索引: -')
            self.selected_name_label.setText('名称: -')
            self.edit_btn.setEnabled(False)
            return
        
        item = self.class_list.item(row)
        index = item.data(Qt.UserRole)
        
        self.selected_index_label.setText(f'索引: {index}')
        self.selected_name_label.setText(f'名称: {self.class_names[index]}')
        self.edit_btn.setEnabled(True)
    
    def edit_class_name(self, item):
        index = item.data(Qt.UserRole)
        self.do_edit_class(index)
    
    def edit_selected_class(self):
        row = self.class_list.currentRow()
        if row < 0:
            return
        
        item = self.class_list.item(row)
        index = item.data(Qt.UserRole)
        self.do_edit_class(index)
    
    def do_edit_class(self, index):
        current_name = self.class_names[index]
        
        dialog = EditDialog(current_name, self)
        if dialog.exec_() == QDialog.Accepted:
            new_name = dialog.get_new_name()
            
            if not new_name:
                QMessageBox.warning(self, '提示', '名称不能为空')
                return
            
            if new_name == current_name:
                return
            
            if new_name in self.class_names:
                reply = QMessageBox.question(
                    self, '确认',
                    f'名称 "{new_name}" 已存在，确定要重复吗？',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
            
            self.class_names[index] = new_name
            self.modified = True
            
            self.update_modified_status()
            self.update_class_list(self.search_edit.text())
            
            self.log_label.setText(f'已修改: [{index}] "{current_name}" → "{new_name}"')
            self.log_label.setStyleSheet('color: #FF9800; padding: 5px;')
    
    def preview_batch_replace(self):
        find_text = self.find_edit.text()
        replace_text = self.replace_edit.text()
        
        if not find_text:
            QMessageBox.warning(self, '提示', '请输入要查找的文本')
            return
        
        matches = []
        for i, name in enumerate(self.class_names):
            if find_text in name:
                new_name = name.replace(find_text, replace_text)
                matches.append((i, name, new_name))
        
        if not matches:
            QMessageBox.information(self, '结果', f'没有找到包含 "{find_text}" 的类别')
            return
        
        msg = f'将替换 {len(matches)} 个类别:\n\n'
        for i, old, new in matches[:10]:
            msg += f'[{i}] {old} → {new}\n'
        if len(matches) > 10:
            msg += f'\n... 还有 {len(matches) - 10} 个'
        
        QMessageBox.information(self, '预览结果', msg)
    
    def apply_batch_replace(self):
        find_text = self.find_edit.text()
        replace_text = self.replace_edit.text()
        
        if not find_text:
            QMessageBox.warning(self, '提示', '请输入要查找的文本')
            return
        
        count = 0
        for i, name in enumerate(self.class_names):
            if find_text in name:
                self.class_names[i] = name.replace(find_text, replace_text)
                count += 1
        
        if count == 0:
            QMessageBox.information(self, '结果', f'没有找到包含 "{find_text}" 的类别')
            return
        
        self.modified = True
        self.update_modified_status()
        self.update_class_list(self.search_edit.text())
        
        self.log_label.setText(f'批量替换完成: 修改了 {count} 个类别')
        self.log_label.setStyleSheet('color: #FF9800; padding: 5px;')
    
    def update_modified_status(self):
        if self.modified:
            self.modified_label.setText('已修改 (未保存)')
            self.modified_label.setStyleSheet('color: #F44336; font-weight: bold;')
            self.save_btn.setEnabled(True)
        else:
            self.modified_label.setText('未修改')
            self.modified_label.setStyleSheet('color: #666;')
            self.save_btn.setEnabled(False)
    
    def save_model(self):
        if not self.model_path:
            return
        
        self._save_to_path(self.model_path)
    
    def save_model_as(self):
        if not self.checkpoint:
            return
        
        default_name = os.path.basename(self.model_path)
        if default_name.endswith('.pth'):
            default_name = default_name[:-4]
        default_name += '_modified.pth'
        
        default_dir = os.path.dirname(self.model_path)
        
        path, _ = QFileDialog.getSaveFileName(
            self, '保存模型', 
            os.path.join(default_dir, default_name),
            '模型文件 (*.pth)'
        )
        
        if path:
            self._save_to_path(path)
    
    def _save_to_path(self, path):
        try:
            self.log_label.setText('正在保存...')
            self.log_label.setStyleSheet('color: #FF9800; padding: 5px;')
            QApplication.processEvents()
            
            self.checkpoint['class_names'] = self.class_names
            
            torch.save(self.checkpoint, path)
            
            self.modified = False
            self.original_class_names = list(self.class_names)
            self.update_modified_status()
            self.update_class_list(self.search_edit.text())
            
            self.log_label.setText(f'保存成功: {os.path.basename(path)}')
            self.log_label.setStyleSheet('color: #4CAF50; padding: 5px;')
            
            QMessageBox.information(self, '成功', f'保存成功: {os.path.basename(path)}')
            
        except Exception as e:
            QMessageBox.warning(self, '错误', f'保存失败: {str(e)}')
            self.log_label.setText('保存失败')
            self.log_label.setStyleSheet('color: #F44336; padding: 5px;')


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont('Microsoft YaHei UI', 9)
    app.setFont(font)
    
    window = ModelFixer()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
