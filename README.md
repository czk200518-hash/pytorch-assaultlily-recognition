# 突击莉莉角色识别项目
## 一、项目概述
本项目是一个基于pytorch的动漫角色识别系统，专门针对《突击莉莉》(Assault Lily) 系列作品中的角色进行识别。项目包含完整的模型训练功能。
### 项目结构
```
突击莉莉角色识别/
├── core/                    # 核心模块
│   ├── model.py            # 模型定义
│   ├── train.py            # 训练逻辑
│   ├── dataset_loader.py   # 数据集加载
│   └── predict.py          # 预测模块
├── utils/                   # 工具模块
│   ├── face_detector.py    # 动漫人脸检测
│   ├── hardware_detector.py # 硬件检测与优化
│   ├── cpu_optimizer.py    # CPU优化
│   └── gpu_optimizer.py    # GPU优化
├── models/                  # 训练好的模型文件
│   └── 模型/
├── gui.py                   # PyQt5图形界面
├── main.py                  # 命令行入口
├── config.json              # 配置文件
└── requirements.txt         # 依赖列表
```

---

## 二、核心模块技术细节
### 2.1 模型架构 (core/model.py)
#### 2.1.1 自定义CNN模型系列
项目实现了多款针对动漫人脸识别优化的CNN模型：

| 模型名称 | 参数量 | 适用场景 | 特点 |
|---------|--------|---------|------|
| tiny | ~500K | <50张总图 | 3层卷积，极速训练 |
| small | ~3M | <100张/类 | 4层卷积，轻量级 |
| standard | ~10M | 通用场景 | 5层卷积+3层全连接 |
| large | ~20M | >2000张总图 | 更深更宽的网络 |
| mobilenet | ~2M | CPU高效 | 深度可分离卷积 |

#### 2.1.2 注意力机制
实现了两种注意力模块：

**SEBlock (Squeeze-and-Excitation)**
- 通过全局信息压缩和激励增强通道间特征表达
- 压缩比: reduction=16

**CBAM (Convolutional Block Attention Module)**
- 结合通道注意力和空间注意力
- 通道注意力: 同时使用AvgPool和MaxPool
- 空间注意力: 7x7卷积核

#### 2.1.3 预训练模型支持

支持基于ImageNet预训练的模型：
- ResNet18/34/50
- EfficientNet-B0/B1/B2

### 2.2 训练模块 (core/train.py)
#### 2.2.1 监督对比学习损失 (SupConLoss)
```python
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        # 温度参数控制分布平滑度
```

#### 2.2.2 训练优化技术
**混合精度训练 (AMP)**
- 自动检测CUDA可用性
- 使用torch.amp.autocast和GradScaler
- 显著降低显存占用，提升训练速度
**梯度累积**
- 支持自定义累积步数
- 等效于增大batch size
- 适用于显存受限场景
**学习率调度**
- ReduceLROnPlateau: 验证准确率停滞时降低学习率
- factor=0.5, patience=5
**梯度裁剪**
- max_norm=1.0
- 防止梯度爆炸

#### 2.2.3 训练流程
```
1. 设备检测与优化配置
2. 数据集加载与预处理
3. 模型初始化
4. 检查点恢复(可选)
5. 训练循环:
   - 前向传播
   - 损失计算(分类+对比学习)
   - 反向传播
   - 优化器更新
   - 验证评估
   - 学习率调整
   - 检查点保存
```

### 2.3 数据集加载模块 (core/dataset_loader.py)
#### 2.3.1 数据增强策略
**标准增强**
```python
transforms.Compose([
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    RandomErasing(p=0.3, scale=(0.02, 0.1)),
])
```

**局部区域增强 (LocalRegionTransform)**
- 45%: 整图 + 标准增强
- 40%: 随机裁剪 (50%-90%区域，增强对角色特征的识别)
- 15%: 五裁剪 (左上/右上/左下/右下/中心)

**对比学习增强 (ContrastiveTransform)**
- 对同一张图片生成两个不同的增强版本
- 更强的颜色抖动和几何变换

#### 2.3.2 数据集类
| 类名 | 用途 | 返回值 |
|-----|------|-------|
| ContrastiveDataset | 对比学习 | (view1, view2, label) |
| CombinedDataset | 分类+对比学习 | (img_cls, view1, view2, label) |
| CombinedContrastiveLocalDataset | 组合增强 | (img_cls, view1, view2, label) |
| LocalRegionDataset | 局部增强 | (image, label) |

#### 2.3.3 数据加载优化
```python
DataLoader(
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,           # 锁页内存，加速GPU传输
    persistent_workers=True,   # 持久化工作进程
    prefetch_factor=2,         # 预取因子
)
```

---

## 三、工具模块技术细节
### 3.1 硬件检测与优化 (utils/hardware_detector.py)
#### 3.1.1 硬件信息采集
```python
@dataclass
class HardwareInfo:
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_name: str
    gpu_total_memory_gb: float
    gpu_available_memory_gb: float
    gpu_compute_capability: str
    platform_info: str
```

#### 3.1.2 性能基准测试
- 模型前向传播基准测试
- 最大批量大小探测 (二分查找)
- 数据加载器性能测试
- 最优工作进程数确定

#### 3.1.3 自动配置推荐
```python
@dataclass
class OptimalConfig:
    model_name: str
    input_size: int
    batch_size: int
    num_workers: int
    use_amp: bool
    accum_steps: int
    estimated_time_per_epoch_min: float
    confidence: str
```

### 3.2 CPU优化 (utils/cpu_optimizer.py)
三级自适应优化策略：
| 等级 | 优化方式 | 提升幅度 | 条件 |
|-----|---------|---------|------|
| 1 | Intel IPEX | 2~4倍 | Intel CPU |
| 2 | torch.compile | 15~30% | PyTorch>=2.0 |
| 3 | 线程优化+channels_last | 10~20% | 通用 |

针对Intel混合架构(12/13/14代)的特殊优化：
- 自动检测P核/E核配置
- 调整线程数为物理核心数的75%

### 3.3 GPU优化 (utils/gpu_optimizer.py)
#### 3.3.1 显存管理
```python
def setup_gpu_optimizations():
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.95)
    # 低显存GPU额外配置
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

#### 3.3.2 自动批量调整（为保守推荐，实际可以提高）
根据显存大小推荐配置：
| 显存 | 推荐输入尺寸 | 批量系数 |
|-----|------------|---------|
| >=12GB | 224 | 2.0x |
| 8-12GB | 224 | 1.5x |
| 6-8GB | 192 | 1.0x |
| 4-6GB | 160 | 0.75x |
| <4GB | 128 | 0.5x |

#### 3.3.3 梯度累积计算
```python
def get_gradient_accumulation_steps(batch_size, target_effective_batch=32):
    # 自动计算梯度累积步数以达到目标等效批量
```

---
## 四、图形界面 (gui.py)
### 4.1 技术栈
- PyQt5: GUI框架
- matplotlib: 训练曲线可视化
- 多线程: QThread实现异步训练

### 4.2 功能模块
1. **训练配置面板**
   - 数据集选择
   - 模型类型选择
   - 超参数配置
   - 硬件检测
2. **训练监控**
   - 实时日志输出
   - 训练曲线绘制
   - 进度显示
   - 时间预估

### 4.3 多线程架构
```python
class TrainingThread(QThread):
    log_signal = pyqtSignal(str)           # 日志信号
    progress_signal = pyqtSignal(int, int) # 进度信号
    finished_signal = pyqtSignal(bool, str)# 完成信号
    plot_data_signal = pyqtSignal(...)     # 绘图数据信号
```

## 六、训练成果
### 6.1 最佳模型
**ResNet34**
- 数据集: 39,270张图片 (86个角色)
- 验证集：7447张图片
- 验证准确率: 99.99%
- 训练配置:
  - 输入尺寸: 256*256
  - 批量大小: 32
  - 混合精度: 启用
  - 对比学习: 启用
  - 局部区域增强: 启用
**ResNet50**
- 数据集: 39,270张图片
- 验证集：7447张图片
- 验证准确率: 100%
- 更强的特征提取能力
- 训练配置:
  - 输入尺寸: 256*256
  - 批量大小: 32
  - 混合精度: 启用
  - 对比学习: 启用
  - 局部区域增强: 启用

### 6.2 训练日志分析
```
[设备] CUDA:0 (NVIDIA GeForce RTX 3050 Laptop GPU)
[AMP] 混合精度训练已启用
[组合模式] 已启用对比学习 + 局部区域增强
[数据] 分类数: 86, 训练集: 39270 张, 验证集: 6930 张
第34/50 轮|训练准确率:98.21%|验证准确率:99.99%
峰值显存: 1.17GB
```

---
## 七、依赖环境
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=10.0.0
opencv-python-headless>=4.5.0
PyQt5>=5.15.0
matplotlib>=3.7.0
```

可选依赖：
- psutil: 硬件信息采集
- intel_extension_for_pytorch: Intel CPU加速

---

## 八、技术亮点
1. **多模型架构支持**: 从轻量级到大型模型，适应不同场景
2. **对比学习**: 提升特征表示能力，增强模型泛化性
3. **局部区域增强**: 使模型能从局部特征识别角色
4. **智能硬件优化**: 自动检测硬件配置并推荐最优参数（该功能相对保守）
6. **动漫人脸检测**: 专门优化的检测算法，支持全身立绘
7. **完善的GUI**: 实时监控、可视化、一键操作

*文档生成时间: 2026-05-03*
*项目作者: 突击莉莉角色识别项目组*
