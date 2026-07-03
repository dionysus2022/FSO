"""
debug_cdm_statistics.py — CDM 统计矩诊断脚本
============================================
目的: 验证"弱湍流质心漂移 > 强湍流"的物理假说,
      解释 Baseline CNN 在弱湍流下准确率反低于强湍流的反常现象。

分析维度:
  1. 质心漂移 (Center of Mass Shift) — 图像亮度重心偏离中心 (32,32) 的距离
  2. 能量截断率 (Clipping Ratio) — 边界像素占总能量的比例
  3. 像素散布程度 (Spatial Spread) — 有效像素的分布区域占比

用法: python debug_cdm_statistics.py
"""

import os, sys, re, glob, random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import defaultdict

# ==================== 配置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
NUM_SAMPLES = 1000          # 每组统计样本数
TARGET_MODS = [4, 6]        # 16QAM (mod=4) 和 64QAM (mod=6) — 最容易混淆
TURBULENCE_GROUPS = ["weak", "moderate", "strong"]

FILENAME_PATTERN = re.compile(
    r'screen_(weak|moderate|strong)_(\d+)_frame_(\d+)(?:_snr_(\d+))?_mod_(\d+)\.mat$'
)

# ==================== 文件收集 ====================
print("=" * 60)
print("  debug_cdm_statistics.py — CDM 统计矩诊断")
print("=" * 60)
print(f"[数据] {DATA_DIR}")

all_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
files_by_group = {g: [] for g in TURBULENCE_GROUPS}

for f in all_files:
    m = FILENAME_PATTERN.search(os.path.basename(f))
    if not m: continue
    turb = m.group(1)
    mod = int(m.group(5))
    if mod not in TARGET_MODS: continue
    files_by_group[turb].append(f)

for g in TURBULENCE_GROUPS:
    print(f"  {g}: {len(files_by_group[g])} 个候选文件")

# ==================== 统计函数 ====================
def compute_centroid_shift(cdm):
    """计算质心漂移: 像素亮度重心偏离图像中心 (32,32) 的欧氏距离"""
    h, w = cdm.shape
    yy, xx = np.mgrid[0:h, 0:w]
    total = cdm.sum()
    if total == 0:
        return 0.0
    cx = (xx * cdm).sum() / total
    cy = (yy * cdm).sum() / total
    return np.sqrt((cx - 31.5) ** 2 + (cy - 31.5) ** 2)

def compute_clipping_ratio(cdm):
    """计算能量截断率: 最外圈像素能量 / 全图总能量"""
    total = cdm.sum()
    if total == 0:
        return 0.0
    boundary = np.concatenate([cdm[0, :], cdm[-1, :], cdm[1:-1, 0], cdm[1:-1, -1]])
    return boundary.sum() / total

def compute_spread(cdm):
    """计算空间散布: 能量 > 0.01 的像素占比"""
    return (cdm > 0.01).sum() / cdm.size

# ==================== 主统计循环 ====================
results = {}
for turb in TURBULENCE_GROUPS:
    files = files_by_group[turb]
    if len(files) == 0:
        results[turb] = None
        continue
    
    n = min(NUM_SAMPLES, len(files))
    sampled = random.sample(files, n)
    
    centroids = []
    clips = []
    spreads = []
    
    for fp in sampled:
        try:
            data = sio.loadmat(fp)
            d = data["Distorted_CDM"]
            centroids.append(float(compute_centroid_shift(d)))
            clips.append(float(compute_clipping_ratio(d)))
            spreads.append(float(compute_spread(d)))
        except:
            continue
    
    centroids = np.array(centroids)
    clips = np.array(clips)
    spreads = np.array(spreads)
    
    results[turb] = {
        "n": len(centroids),
        "centroid_mean": float(centroids.mean()),
        "centroid_std": float(centroids.std()),
        "centroid_max": float(centroids.max()),
        "clipping_mean": float(clips.mean()),
        "clipping_std": float(clips.std()),
        "spread_mean": float(spreads.mean()),
        "spread_std": float(spreads.std()),
    }
    
    print(f"\n--- {turb.upper()} 湍流 (N={len(centroids)}) ---")
    print(f"  质心漂移: {centroids.mean():.3f} ± {centroids.std():.3f} px  (max={centroids.max():.2f})")
    print(f"  能量截断率: {clips.mean()*100:.2f}% ± {clips.std()*100:.2f}%")
    print(f"  空间散布: {spreads.mean()*100:.1f}% ± {spreads.std()*100:.1f}%")

# ==================== 对比报告 ====================
print("\n" + "=" * 60)
print("  [结论] 弱湍流 vs 强湍流 对比")
print("=" * 60)

if results["weak"] and results["strong"]:
    w = results["weak"]
    s = results["strong"]
    
    centroid_ratio = w["centroid_mean"] / max(s["centroid_mean"], 1e-6)
    clipping_ratio = w["clipping_mean"] / max(s["clipping_mean"], 1e-6)
    
    print(f"  质心漂移: Weak={w['centroid_mean']:.3f} px  vs  Strong={s['centroid_mean']:.3f} px")
    print(f"  → 弱湍流质心漂移 {centroid_ratio:.1f}x {'>' if centroid_ratio > 1.1 else '≈' if centroid_ratio > 0.9 else '<'} 强湍流")
    print()
    print(f"  能量截断率: Weak={w['clipping_mean']*100:.2f}%  vs  Strong={s['clipping_mean']*100:.2f}%")
    print(f"  → 弱湍流边界截断率 {clipping_ratio:.1f}x {'>' if clipping_ratio > 1.1 else '≈' if clipping_ratio > 0.9 else '<'} 强湍流")
    print()
    print(f"  空间散布: Weak={w['spread_mean']*100:.1f}%  vs  Strong={w['spread_mean']*100:.1f}%")
    
    print()
    if centroid_ratio > 1.3:
        print("  [实锤] 弱湍流存在显著质心漂移（>30%），高密度星座点被推出 CNN 局部感受野。")
        print("  → 物理根因: 弱湍流下光束漂移为主、散射为辅，导致整体 CDM 移位但未弥散。")
        print("  → CNN 失效: 局部卷积核只能看到位移后的周围背景，无法感知全局拓扑。")
    elif clipping_ratio > 1.3:
        print("  [实锤] 弱湍流存在显著边界截断（>30%），大量有效像素被 clip 到图像边缘。")
        print("  → 物理根因: 弱湍流造成光束漂移，星座点被 AGC 归一化推至 [-2, 2] 之外。")
        print("  → 解决方案: 扩大 histcounts2 的 edges 范围（如 linspace(-3, 3, 65)）。")
    else:
        print("  [待查] 弱/强湍流的统计矩差异不显著，建议检查其他因素（如 256QAM 先验坍塌影响）。")

print("=" * 60)
