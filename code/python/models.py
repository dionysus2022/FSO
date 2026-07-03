"""
models.py — FSO 大气湍流畸变修复项目的 网络架构模块
========================================================
定义核心网络:
  1. ConstellationCNN      — CNN 分类器: 识别调制格式 (5分类)
  2. ConstellationUNet     — GNN 去噪修复网络: 畸变图 → 修复图 (U-Net 结构)
  3. ConstellationViT      — Vision Transformer 分类器
  4. UNet_ResNet18_E2E     — U-Net + 改造 ResNet-18 端到端级联
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, swin_t


# ================================================================
# 第一部分: ConstellationCNN — 调制格式分类器
# ================================================================

class ConstellationCNN(nn.Module):
    """
    星座密度图调制格式分类器 (CNN)

    输入:  (B, 1, 64, 64)  单通道星座密度图
    输出:  (B, 5)          5个调制格式类别的 Logits

    网络结构概览 (4层卷积 + 2层全连接):
      Block  |  操作                |  输出维度
      -------|----------------------|---------------
      输入   |  -                   |  (B, 1, 64, 64)
      Conv1  |  Conv+BN+ReLU+Pool   |  (B, 16, 32, 32)
      Conv2  |  Conv+BN+ReLU+Pool   |  (B, 32, 16, 16)
      Conv3  |  Conv+BN+ReLU+Pool   |  (B, 64, 8, 8)
      Conv4  |  Conv+BN+ReLU+Pool   |  (B, 128, 4, 4)
      Flatten|  展平                |  (B, 128*4*4) = (B, 2048)
      FC1    |  Linear+ReLU+Dropout |  (B, 256)
      FC2    |  Linear              |  (B, 5)  ← 5类Logits
    """

    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.3, in_channels: int = 1):
        """
        参数:
          num_classes:   分类类别数 (默认5: BPSK/QPSK/16QAM/64QAM/256QAM)
          dropout_rate:  Dropout 概率, 用于防止过拟合
          in_channels:   输入图像通道数 (1=单通道CDM, 2=CDM+GAF双通道)
        """
        super(ConstellationCNN, self).__init__()

        # ---- 卷积特征提取部分 ----

        # Conv Block 1: in_channels → 16 通道, 空间尺寸减半 (64→32)
        # 使用较大的 kernel=5 来捕获星座图中的局部聚类模式
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, padding=2),
            # 输入 (B, 1, 64, 64) → 输出 (B, 16, 64, 64)
            nn.BatchNorm2d(16),         # 归一化加速收敛
            nn.ReLU(inplace=True),      # 非线性激活
            nn.MaxPool2d(kernel_size=2),  # 下采样 → (B, 16, 32, 32)
        )

        # Conv Block 2: 16 → 32 通道, 空间尺寸减半 (32→16)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            # 输入 (B, 16, 32, 32) → 输出 (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 下采样 → (B, 32, 16, 16)
        )

        # Conv Block 3: 32 → 64 通道, 空间尺寸减半 (16→8)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # 输入 (B, 32, 16, 16) → 输出 (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 下采样 → (B, 64, 8, 8)
        )

        # Conv Block 4: 64 → 128 通道, 空间尺寸减半 (8→4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            # 输入 (B, 64, 8, 8) → 输出 (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 下采样 → (B, 128, 4, 4)
        )

        # ---- 全连接分类部分 ----

        # 计算展平后的维度: 128通道 × 4×4空间 = 2048
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),   # 2048 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),    # Dropout 正则化
        )

        # 最终分类头: 256 → num_classes (5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
          x: Tensor 形状 (B, 1, 64, 64)  输入星座密度图

        返回:
          Tensor 形状 (B, num_classes)   5类 Logits (未经 softmax)
        """
        # 卷积特征提取: (B,1,64,64) → (B,128,4,4)
        x = self.conv1(x)   # → (B, 16, 32, 32)
        x = self.conv2(x)   # → (B, 32, 16, 16)
        x = self.conv3(x)   # → (B, 64, 8, 8)
        x = self.conv4(x)   # → (B, 128, 4, 4)

        # 展平: (B, 128, 4, 4) → (B, 2048)
        x = x.view(x.size(0), -1)

        # 全连接分类: (B, 2048) → (B, 256) → (B, 5)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


# ================================================================
# 第二部分: ConstellationUNet — U-Net 去噪修复网络
# ================================================================

class _DoubleConv(nn.Module):
    """
    U-Net 的基础构建块: 两次 Conv+BN+ReLU

    输入 → Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU → 输出

    这种"双卷积"设计是 U-Net 的标准做法,
    两个连续的 3×3 卷积等效于一个 5×5 卷积的感受野, 但参数量更少。
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """
        参数:
          in_channels:  输入通道数
          out_channels: 输出通道数
          mid_channels: 中间通道数 (默认与 out_channels 相同)
        """
        super(_DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            # 第一个卷积: 通道数从 in_channels → mid_channels
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 第二个卷积: 通道数从 mid_channels → out_channels
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class _DownSample(nn.Module):
    """
    编码器下采样模块: MaxPool(2) + DoubleConv

    每次下采样将空间尺寸减半, 同时通道数翻倍。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(_DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),           # 尺寸减半
            _DoubleConv(in_channels, out_channels),  # 通道加倍
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class _UpSample(nn.Module):
    """
    解码器上采样模块: 转置卷积(或双线性插值) + DoubleConv

    每次上采样将空间尺寸加倍, 同时通道数减半。
    Skip Connection 将编码器对应层的特征与上采样结果拼接,
    从而保留高分辨率细节信息, 这是 U-Net 恢复图像质量的关键设计。
    """

    def __init__(self, x1_channels: int, x2_channels: int, out_channels: int, bilinear: bool = True):
        """
        参数:
          x1_channels:  来自解码器上一层的特征图通道数
          x2_channels:  来自编码器对应层的skip connection通道数
          out_channels: 输出通道数
          bilinear:     是否使用双线性插值上采样 (True) 还是转置卷积 (False)
        """
        super(_UpSample, self).__init__()
        in_channels = x1_channels + x2_channels  # skip connection拼接后的总通道数

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = _DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(x1_channels, x1_channels, kernel_size=2, stride=2)
            self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        参数:
          x1: 来自解码器上一层的特征图 (较低分辨率)
          x2: 来自编码器对应层的特征图 (较高分辨率, skip connection)

        返回:
          上采样并拼接后的特征图
        """
        # 步骤1: 上采样 x1, 使其空间尺寸与 x2 一致
        x1 = self.up(x1)

        # 步骤2: 处理可能因尺寸不匹配导致的边界问题 (padding 裁剪)
        # 例如输入 64→32→16→8, 上采样 8→16→32→64 可能因除法取整产生 1px 差异
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [
            diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2,
        ])

        # 步骤3: 沿通道维度拼接 skip connection 的特征 (concat)
        # Encoder特征(浅层) + Decoder特征(深层) → 融合高低层语义
        x = torch.cat([x2, x1], dim=1)

        # 步骤4: DoubleConv 融合特征
        return self.conv(x)


class ConstellationUNet(nn.Module):
    """
    星座密度图去噪修复网络 (轻量级 U-Net)

    输入:  (B, 1, 64, 64)  带湍流畸变的星座密度图  Distorted_CDM
    输出:  (B, 1, 64, 64)  修复后的纯净星座密度图  (值域 [0, 1])

    网络结构概览 (Encoder-Bottleneck-Decoder):

      Encoder (下采样路径):
        enc1: DoubleConv( 1→16 )         → (B, 16, 64, 64)   ← 与 dec3 做 skip
        enc2: DownSample( 16→32 )        → (B, 32, 32, 32)   ← 与 dec2 做 skip
        enc3: DownSample( 32→64 )        → (B, 64, 16, 16)   ← 与 dec1 做 skip
        enc4: DownSample( 64→128 )       → (B, 128, 8, 8)

      Bottleneck (瓶颈层):
        bottleneck: DoubleConv(128→128)  → (B, 128, 8, 8)

      Decoder (上采样路径):
        dec1: UpSample(128+64→64)        → (B, 64, 16, 16)
        dec2: UpSample(64+32→32)         → (B, 32, 32, 32)
        dec3: UpSample(32+16→16)         → (B, 16, 64, 64)

      Output (输出层):
        out_conv: Conv2d(16→1, k=1)     → (B, 1, 64, 64)
        sigmoid: 将输出压缩到 [0, 1] 范围

    参数总量约: ~0.5M (非常轻量, 适合中规模数据集)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, bilinear: bool = True):
        """
        参数:
          in_channels:  输入通道数 (默认1, 灰度图)
          out_channels: 输出通道数 (默认1, 灰度图)
          bilinear:     上采样方式 (True=双线性插值, False=转置卷积)
        """
        super(ConstellationUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # ---- Encoder (编码器/下采样路径) ----
        # 第一层不做下采样, 先用双卷积提取初始特征
        self.enc1 = _DoubleConv(in_channels, 16)     # (1→16, 尺寸不变)
        self.enc2 = _DownSample(16, 32)               # (16→32, 尺寸减半)
        self.enc3 = _DownSample(32, 64)               # (32→64, 尺寸减半)
        self.enc4 = _DownSample(64, 128)              # (64→128, 尺寸减半)

        # ---- Bottleneck (瓶颈层) ----
        # 最深层的特征提取, 拥有最大的感受野
        factor = 2 if bilinear else 1
        self.bottleneck = _DoubleConv(128, 256 // factor)  # (128→128, 尺寸不变)

        # ---- Decoder (解码器/上采样路径) ----
        # 每次上采样通道数减半, 并将对应 encoder 层的特征拼接过来 (skip connection)
        # UpSample(x1_channels, x2_channels, out_channels): x1=解码器输入, x2=skip连接
        self.dec1 = _UpSample(128, 128, 64, bilinear)   # 128(bottleneck) + 128(skip from enc4) → 64
        self.dec2 = _UpSample(64, 64, 32, bilinear)     # 64(dec1) + 64(skip from enc3) → 32
        self.dec3 = _UpSample(32, 16, 16, bilinear)     # 32(dec2) + 16(skip from enc1) → 16

        # ---- Output (输出层) ----
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

        # Sigmoid 激活: 将输出值映射到 [0, 1] 区间
        # 因为 CDM 星座密度图已被归一化到 0~1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 畸变图 → 修复图

        参数:
          x: Tensor 形状 (B, 1, 64, 64)  — Distorted_CDM 畸变星座密度图

        返回:
          Tensor 形状 (B, 1, 64, 64)  — Repaired_CDM 修复后的星座密度图, 值域 [0, 1]
        """
        # ============ Encoder: 逐层提取特征并保存 skip 特征 ============

        # enc1: (B, 1, 64, 64)  → (B, 16, 64, 64)
        e1 = self.enc1(x)

        # enc2: (B, 16, 64, 64) → (B, 32, 32, 32)
        e2 = self.enc2(e1)

        # enc3: (B, 32, 32, 32) → (B, 64, 16, 16)
        e3 = self.enc3(e2)

        # enc4: (B, 64, 16, 16) → (B, 128, 8, 8)
        e4 = self.enc4(e3)

        # ============ Bottleneck: 最深层特征 ============
        # bottleneck: (B, 128, 8, 8) → (B, 128, 8, 8)
        b = self.bottleneck(e4)

        # ============ Decoder: 逐层上采样 + skip connection ============

        # dec1: (B, 128, 8, 8) + skip(e4, 128ch) → (B, 64, 16, 16)
        d1 = self.dec1(b, e4)

        # dec2: (B, 64, 16, 16) + skip(e3, 64ch)  → (B, 32, 32, 32)
        d2 = self.dec2(d1, e3)

        # dec3: (B, 32, 32, 32) + skip(e1, 16ch)  → (B, 16, 64, 64)
        # 注意: skip到enc1而不是enc2, 因为enc2是32×32而我们需要64×64输出
        d3 = self.dec3(d2, e1)

        # ============ Output: 1×1 卷积 + Sigmoid ============

        # out_conv: (B, 16, 64, 64) → (B, 1, 64, 64)
        out = self.out_conv(d3)

        # sigmoid: 将输出值压缩到 [0, 1] 范围
        out = self.sigmoid(out)

        return out


# ================================================================
# 自测代码: 直接运行 python models.py 可以快速验证网络结构
# ================================================================

# ================================================================
# 第三部分: ConstellationViT — Vision Transformer 分类器
# ================================================================

class PatchEmbed(nn.Module):
    """将 64x64 图像分割为 8x8 patch, 每个 patch 投射到 embedding dim"""
    def __init__(self, img_size=64, patch_size=16, in_channels=1, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 1, 64, 64) → (B, embed_dim, 4, 4) → (B, 16, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """单层 Transformer Encoder: Multi-Head Attention + MLP + LayerNorm + Residual"""
    def __init__(self, dim, num_heads=8, mlp_ratio=2., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ConstellationViT(nn.Module):
    """
    轻量级 Vision Transformer 调制格式分类器

    输入:  (B, 1, 64, 64)  星座密度图
    输出:  (B, 5)          5类 Logits

    结构:
      PatchEmbed(1→256, patch=16)  → (B, 16, 256)
      CLS Token + Position Embedding
      Transformer Encoder ×4
      CLS → MLP Head → 5 Logits

    参数量: ~1.8M
    """

    def __init__(self, img_size=64, patch_size=16, in_channels=1, num_classes=5,
                 embed_dim=256, depth=4, num_heads=8, mlp_ratio=2., dropout=0.15):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                    # (B, 16, 256)
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, 256)
        x = torch.cat([cls, x], dim=1)             # (B, 17, 256)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x[:, 0]                                 # CLS token only
        x = self.head(x)                            # (B, 5)
        return x


# ================================================================
# 第四部分: UNet_ResNet18_E2E — U-Net + 改造 ResNet-18 端到端
# ================================================================


class UNet_Swin_E2E(nn.Module):
    """
    U-Net + 改造 Swin-T 端到端级联分类器

    输入:  (B, 1, 64, 64)  Distorted CDM
    输出:  (B, 5)          5类调制格式 Logits

    数据流:
      Distorted CDM → U-Net → Repaired CDM → Swin-T(adapt) → Logits

    Swin-T 适配改造:
      - features[0][0]: 3→1 输入通道的 Conv2d(k=4,s=4)
      - head: 1000→5 输出类别

    参数量: ~29.1M (U-Net 0.83M + Swin-T 28.29M)
    """

    def __init__(self, num_classes: int = 5, in_channels: int = 1):
        super(UNet_Swin_E2E, self).__init__()

        # ---- U-Net 前端 (可接受1/2通道输入, 输出单通道修复图) ----
        self.unet = ConstellationUNet(in_channels=in_channels, out_channels=1, bilinear=True)

        # ---- Swin-T 后端 (始终接收 U-Net 输出的单通道修复图) ----
        self.swin = swin_t(weights=None)

        # 适配 1: 首层 Conv2d 始终保持单通道 (U-Net 输出为 1 通道)
        self.swin.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)

        # 适配 2: 分类头改造为 5 类输出
        self.swin.head = nn.Linear(self.swin.head.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: Distorted CDM → U-Net 修复 → Swin-T 分类
        """
        repaired = self.unet(x)          # (B, in_channels, 64, 64)
        logits = self.swin(repaired)     # (B, 5)
        return logits


class UNet_ResNet18_E2E(nn.Module):
    """
    U-Net + 改造 ResNet-18 端到端级联分类器

    输入:  (B, 1, 64, 64)  Distorted CDM 畸变星座密度图
    输出:  (B, 5)          5类调制格式 Logits

    数据流:
      Distorted CDM → U-Net → Repaired CDM → ResNet-18(adapt) → Logits

    ResNet-18 适配改造:
      - conv1: 3→1 输入通道 (接受单通道 CDM)
      - fc: 1000→5 输出类别

    参数量: ~12.0M (U-Net 0.83M + ResNet-18 11.17M)
    """

    def __init__(self, num_classes: int = 5, in_channels: int = 1):
        super(UNet_ResNet18_E2E, self).__init__()

        # ---- U-Net 前端 (可接受1/2通道输入, 输出单通道修复图) -------
        self.unet = ConstellationUNet(in_channels=in_channels, out_channels=1, bilinear=True)

        # ---- ResNet-18 后端 (始终接收 U-Net 输出的单通道修复图) -------
        self.resnet = resnet18(weights=None)

        # 适配 1: 首层卷积始终保持单通道 (U-Net 输出为 1 通道)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False,
        )

        # 适配 2: 全连接层改造为 5 类输出
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: Distorted CDM → U-Net 修复 → ResNet-18 分类

        参数:
          x: Tensor 形状 (B, in_channels, 64, 64) — Distorted CDM / CDM+GAF

        返回:
          Tensor 形状 (B, num_classes)   — 5类 Logits
        """
        repaired = self.unet(x)       # (B, 1, 64, 64)
        logits = self.resnet(repaired)  # (B, 5)
        return logits


if __name__ == "__main__":
    print("=" * 60)
    print("  网络架构自测")
    print("=" * 60)

    # 模拟一个 batch 的输入: batch_size=4, 单通道 64×64
    dummy_input = torch.randn(4, 1, 64, 64)

    # --- 测试 ConstellationCNN ---
    print("\n>>> [1/2] 测试 ConstellationCNN (分类器)")
    cnn = ConstellationCNN(num_classes=5)
    cnn_out = cnn(dummy_input)
    print(f"  输入形状:  {dummy_input.shape}")
    print(f"  输出形状:  {cnn_out.shape}   (期望: [4, 5])")
    print(f"  输出 Logits: {cnn_out}")

    # 计算参数量
    cnn_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print(f"  可训练参数: {cnn_params:,}")

    # --- 测试 ConstellationUNet ---
    print("\n>>> [2/2] 测试 ConstellationUNet (去噪修复)")
    unet = ConstellationUNet(in_channels=1, out_channels=1, bilinear=True)
    unet_out = unet(dummy_input)
    print(f"  输入形状:  {dummy_input.shape}")
    print(f"  输出形状:  {unet_out.shape}   (期望: [4, 1, 64, 64])")
    print(f"  输出范围:  min={unet_out.min().item():.4f}, max={unet_out.max().item():.4f}")

    unet_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"  可训练参数: {unet_params:,}")

    # --- 测试 UNet_ResNet18_E2E ---
    print("\n>>> [3/3] 测试 UNet_ResNet18_E2E (端到端级联)")
    e2e_resnet = UNet_ResNet18_E2E(num_classes=5)
    e2e_resnet_out = e2e_resnet(dummy_input)
    print(f"  输入形状:  {dummy_input.shape}")
    print(f"  输出形状:  {e2e_resnet_out.shape}   (期望: [4, 5])")
    print(f"  输出 Logits: {e2e_resnet_out}")

    e2e_resnet_params = sum(p.numel() for p in e2e_resnet.parameters() if p.requires_grad)
    print(f"  可训练参数: {e2e_resnet_params:,}")

    print("\n[OK] 自测通过! 全部网络前向传播正常。")
