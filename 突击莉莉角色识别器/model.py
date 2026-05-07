import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stderr is None:sys.stderr = sys.stdout
try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
        )
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_out = self.shared_mlp(self.channel_attention[0](x).view(b, c))
        max_out = self.shared_mlp(self.channel_attention[1](x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = 512, out_features: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features),
        )
    
    def forward(self, x):
        return self.projection(x)

class AnimeFaceCNN(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 128):
        super().__init__()
        self.input_size = input_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        feature_size = 512 * 4 * 4

        self.fc = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)

        for layer in self.fc[:-1]:
            x = layer(x)
        return x

class AnimeFaceCNNTiny(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 128):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        feature_size = 64 * 4 * 4

        self.fc = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class AnimeFaceCNNLarge(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 128):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        feature_size = 512 * 4 * 4

        self.fc = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class _InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class AnimeFaceMobileNet(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            _InvertedResidual(32, 16, 1, 1),
            _InvertedResidual(16, 24, 2, 6),
            _InvertedResidual(24, 24, 1, 6),
            _InvertedResidual(24, 32, 2, 6),
            _InvertedResidual(32, 32, 1, 6),
            _InvertedResidual(32, 32, 1, 6),
            _InvertedResidual(32, 64, 2, 6),
            _InvertedResidual(64, 64, 1, 6),
            _InvertedResidual(64, 64, 1, 6),
            _InvertedResidual(64, 64, 1, 6),
            _InvertedResidual(64, 96, 1, 6),
            _InvertedResidual(96, 96, 1, 6),
            _InvertedResidual(96, 96, 1, 6),
            _InvertedResidual(96, 160, 2, 6),
            _InvertedResidual(160, 160, 1, 6),
            _InvertedResidual(160, 160, 1, 6),
            _InvertedResidual(160, 320, 1, 6),

            nn.Conv2d(320, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AnimeFaceCNNSmall(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 128):
        super().__init__()
        self.input_size = input_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        feature_size = 256 * 4 * 4

        self.fc = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)

        for layer in self.fc[:-1]:
            x = layer(x)
        return x

class AnimeFaceCNNWithAttention(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 128, attention_type: str = 'cbam', use_contrastive: bool = False):
        super().__init__()
        self.input_size = input_size
        self.use_contrastive = use_contrastive
        self.attention_type = attention_type
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )
        self.att1 = self._make_attention(32)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )
        self.att2 = self._make_attention(64)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        self.att3 = self._make_attention(128)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.att4 = self._make_attention(256)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.att5 = self._make_attention(512)
        
        feature_size = 512 * 4 * 4
        
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        
        if use_contrastive:
            self.projection_head = ProjectionHead(256, 512, 128)
        
        self._initialize_weights()
    
    def _make_attention(self, channels: int):
        if self.attention_type == 'se':
            return SEBlock(channels)
        elif self.attention_type == 'cbam':
            return CBAM(channels)
        else:
            return nn.Identity()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features: bool = False):
        x = self.conv1(x)
        x = self.att1(x)
        
        x = self.conv2(x)
        x = self.att2(x)
        
        x = self.conv3(x)
        x = self.att3(x)
        
        x = self.conv4(x)
        x = self.att4(x)
        
        x = self.conv5(x)
        x = self.att5(x)
        
        x = torch.flatten(x, 1)
        
        for i, layer in enumerate(self.fc):
            x = layer(x)
            if i == 5:
                features = x
        
        logits = self.fc[6](x) if len(self.fc) > 6 else x
        
        if return_features and self.use_contrastive:
            proj = self.projection_head(features)
            return logits, proj
        
        if return_features:
            return logits, features
        
        return logits
    
    def extract_features(self, x):
        x = self.conv1(x)
        x = self.att1(x)
        x = self.conv2(x)
        x = self.att2(x)
        x = self.conv3(x)
        x = self.att3(x)
        x = self.conv4(x)
        x = self.att4(x)
        x = self.conv5(x)
        x = self.att5(x)
        x = torch.flatten(x, 1)
        
        for layer in self.fc[:-1]:
            x = layer(x)
        return x

class ContrastiveWrapper(nn.Module):
    """对比学习包装器 - 为预训练模型添加对比学习支持
    
    参数:
        base_model: 基础模型
        feature_dim: 特征维度
        num_classes: 分类数
        use_contrastive: 是否启用对比学习
    """
    
    def __init__(self, base_model: nn.Module, feature_dim: int, num_classes: int, use_contrastive: bool = False):
        super().__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_contrastive = use_contrastive
        
        if use_contrastive:
            self.projection_head = ProjectionHead(feature_dim, 256, 128)
    
    def forward(self, x, return_features: bool = False):
        if hasattr(self.base_model, 'forward'):
            if 'resnet' in str(type(self.base_model)).lower():
                features = self.base_model.fc(x) if x.dim() == 2 else self._extract_resnet_features(x)
                logits = features
            elif 'efficientnet' in str(type(self.base_model)).lower():
                features = self.base_model.classifier(x) if x.dim() == 2 else self._extract_efficientnet_features(x)
                logits = features
            elif 'vit' in str(type(self.base_model)).lower():
                features = self.base_model.heads.head(x) if x.dim() == 2 else self._extract_vit_features(x)
                logits = features
            else:
                logits = self.base_model(x)
                features = logits
        else:
            logits = self.base_model(x)
            features = logits
        
        if return_features and self.use_contrastive:
            proj = self.projection_head(features)
            return logits, proj
        
        if return_features:
            return logits, features
        
        return logits
    
    def _extract_resnet_features(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = torch.nn.functional.relu(x, inplace=False)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        features = torch.flatten(x, 1)
        return self.base_model.fc(features)
    
    def _extract_efficientnet_features(self, x):
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        features = torch.flatten(x, 1)
        return self.base_model.classifier(features)
    
    def _extract_vit_features(self, x):
        x = self.base_model._process_input(x)
        batch_class_token = self.base_model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.base_model.encoder(x)
        x = x[:, 0]
        return self.base_model.heads.head(x)


def create_model(model_name: str, num_classes: int, input_size: int = 128, pretrained: bool = True, 
                  attention_type: str = None, use_contrastive: bool = False) -> nn.Module:
    if model_name == 'tiny':
        return AnimeFaceCNNTiny(num_classes, input_size)
    elif model_name == 'small':
        return AnimeFaceCNNSmall(num_classes, input_size)
    elif model_name == 'standard':
        if attention_type or use_contrastive:
            return AnimeFaceCNNWithAttention(
                num_classes, input_size, 
                attention_type=attention_type or 'none',
                use_contrastive=use_contrastive
            )
        return AnimeFaceCNN(num_classes, input_size)
    elif model_name == 'large':
        return AnimeFaceCNNLarge(num_classes, input_size)
    elif model_name == 'mobilenet':
        return AnimeFaceMobileNet(num_classes, input_size)
    
    elif model_name == 'standard_se':
        return AnimeFaceCNNWithAttention(num_classes, input_size, attention_type='se', use_contrastive=use_contrastive)
    elif model_name == 'standard_cbam':
        return AnimeFaceCNNWithAttention(num_classes, input_size, attention_type='cbam', use_contrastive=use_contrastive)
    
    elif model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        if not HAS_TORCHVISION:
            raise ImportError("需要安装 torchvision: pip install torchvision")
        
        weights = None
        if pretrained:
            weights_map = {
                'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
                'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
                'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,
                'resnet101': models.ResNet101_Weights.IMAGENET1K_V1,
                'resnet152': models.ResNet152_Weights.IMAGENET1K_V1,
            }
            weights = weights_map.get(model_name)
        
        resnet_configs = {
            'resnet18': (models.resnet18, 512),
            'resnet34': (models.resnet34, 512),
            'resnet50': (models.resnet50, 2048),
            'resnet101': (models.resnet101, 2048),
            'resnet152': (models.resnet152, 2048),
        }
        
        try:
            model_fn, fc_features = resnet_configs[model_name]
            model = model_fn(weights=weights)
            model.fc = nn.Linear(fc_features, num_classes)
        except Exception as e:
            print(f'[警告] 加载预训练权重失败: {e}')
            print(f'[警告] 使用随机初始化权重')
            model_fn, fc_features = resnet_configs[model_name]
            model = model_fn(weights=None)
            model.fc = nn.Linear(fc_features, num_classes)
        
        if use_contrastive:
            model = ContrastiveWrapper(model, num_classes, num_classes, use_contrastive=True)
        
        return model
    
    elif model_name.startswith('efficientnet_'):
        if not HAS_TORCHVISION:
            raise ImportError("需要安装 torchvision: pip install torchvision")
        
        efficientnet_configs = {
            'efficientnet_b0': (models.efficientnet_b0, 1280, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
            'efficientnet_b1': (models.efficientnet_b1, 1280, models.EfficientNet_B1_Weights.IMAGENET1K_V1),
            'efficientnet_b2': (models.efficientnet_b2, 1408, models.EfficientNet_B2_Weights.IMAGENET1K_V1),
            'efficientnet_b3': (models.efficientnet_b3, 1536, models.EfficientNet_B3_Weights.IMAGENET1K_V1),
            'efficientnet_b4': (models.efficientnet_b4, 1792, models.EfficientNet_B4_Weights.IMAGENET1K_V1),
            'efficientnet_b5': (models.efficientnet_b5, 2048, models.EfficientNet_B5_Weights.IMAGENET1K_V1),
            'efficientnet_b6': (models.efficientnet_b6, 2304, models.EfficientNet_B6_Weights.IMAGENET1K_V1),
            'efficientnet_b7': (models.efficientnet_b7, 2560, models.EfficientNet_B7_Weights.IMAGENET1K_V1),
        }
        
        if model_name not in efficientnet_configs:
            raise ValueError(f"不支持的 EfficientNet: {model_name}")
        
        model_fn, classifier_features, weights_cls = efficientnet_configs[model_name]
        weights = weights_cls if pretrained else None
        
        try:
            model = model_fn(weights=weights)
            model.classifier[1] = nn.Linear(classifier_features, num_classes)
        except Exception as e:
            print(f'[警告] 加载预训练权重失败: {e}')
            print(f'[警告] 使用随机初始化权重')
            model = model_fn(weights=None)
            model.classifier[1] = nn.Linear(classifier_features, num_classes)
        
        if use_contrastive:
            model = ContrastiveWrapper(model, num_classes, num_classes, use_contrastive=True)
        
        return model
    
    elif model_name.startswith('vit_'):
        if not HAS_TORCHVISION:
            raise ImportError("需要安装 torchvision: pip install torchvision")
        
        vit_configs = {
            'vit_b_16': (models.vit_b_16, 768, models.ViT_B_16_Weights.IMAGENET1K_V1),
            'vit_b_32': (models.vit_b_32, 768, models.ViT_B_32_Weights.IMAGENET1K_V1),
            'vit_l_16': (models.vit_l_16, 1024, models.ViT_L_16_Weights.IMAGENET1K_V1),
            'vit_l_32': (models.vit_l_32, 1024, models.ViT_L_32_Weights.IMAGENET1K_V1),
        }
        
        if model_name not in vit_configs:
            raise ValueError(f"不支持的 ViT: {model_name}")
        
        model_fn, head_features, weights_cls = vit_configs[model_name]
        weights = weights_cls if pretrained else None
        
        try:
            model = model_fn(weights=weights)
            model.heads.head = nn.Linear(head_features, num_classes)
        except Exception as e:
            print(f'[警告] 加载预训练权重失败: {e}')
            print(f'[警告] 使用随机初始化权重')
            model = model_fn(weights=None)
            model.heads.head = nn.Linear(head_features, num_classes)
        
        if use_contrastive:
            model = ContrastiveWrapper(model, num_classes, num_classes, use_contrastive=True)
        
        return model
    
    else:
        raise ValueError(
            f"未知的模型名称: {model_name}\n"
            f"可选模型:\n"
            f"  自定义CNN: tiny, small, standard, large, mobilenet, standard_se, standard_cbam\n"
            f"  ResNet: resnet18, resnet34, resnet50, resnet101, resnet152\n"
            f"  EfficientNet: efficientnet_b0~b7\n"
            f"  ViT: vit_b_16, vit_b_32, vit_l_16, vit_l_32"
        )
