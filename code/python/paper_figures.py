"""
论文图表生成脚本
- 图4: 消融实验柱状图
- 图3: 模型结构图
- 图1: 真实星座图矩阵
- 图2: CDM密度图矩阵
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os, sys, pickle, glob

OUT_DIR = r"D:\Project_code\FSO_research\results\docs\figs\训练结果09"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 图4: 消融实验柱状图
# ============================================================
def draw_ablation_bar():
    models = ['stats', 'cdm', 'iq', 'cdm_stats', 'fusion']
    labels = ['Stats', 'CDM', 'IQ', 'CDM+Stats', 'Fusion']
    
    weak_acc  = [72.22, 87.78, 53.33, 90.89, 87.78]
    strong_acc = [77.78, 84.44, 51.11, 86.67, 80.00]
    weak_f1   = [71.66, 86.53, 50.98, 90.32, 87.94]
    strong_f1 = [77.11, 79.43, 45.31, 86.47, 74.61]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Accuracy
    bars1 = ax1.bar(x - width/2, weak_acc, width, label='Weak Turbulence', color='#4C72B0', edgecolor='white', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, strong_acc, width, label='Strong Turbulence', color='#DD8452', edgecolor='white', linewidth=0.8)
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Test Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f'{bar.get_height():.1f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f'{bar.get_height():.1f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # F1
    bars3 = ax2.bar(x - width/2, weak_f1, width, label='Weak Turbulence', color='#4C72B0', edgecolor='white', linewidth=0.8)
    bars4 = ax2.bar(x + width/2, strong_f1, width, label='Strong Turbulence', color='#DD8452', edgecolor='white', linewidth=0.8)
    ax2.set_ylabel('Macro F1 (%)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Macro F1-Score', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f'{bar.get_height():.1f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in bars4:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f'{bar.get_height():.1f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(OUT_DIR, 'fig_ablation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] fig_ablation.png")


# ============================================================
# 图3: cdm_stats 模型结构图
# ============================================================
def draw_model_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    def draw_box(ax, x, y, w, h, text, color='#E8F0FE', edge='#1a73e8', fontsize=10, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                              facecolor=color, edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
                fontweight=weight, wrap=True)
    
    def draw_arrow(ax, x1, y1, x2, y2, color='gray'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
    
    # Input layer
    draw_box(ax, 0.5, 3.5, 2.5, 1.0, 'IQ Samples\n(1×1 complex)', color='#FFF3CD', edge='#FFC107')
    draw_box(ax, 0.5, 5.5, 2.5, 1.0, 'RX Signal\n(Time Domain)', color='#FFF3CD', edge='#FFC107')
    
    # CDM branch
    draw_box(ax, 3.8, 5.5, 2.5, 1.0, 'Constellation\nDensity Map\n(64×64)', color='#D4EDDA', edge='#28A745')
    draw_box(ax, 3.8, 6.7, 2.5, 0.6, 'CDM Generation', color='#E2E3E5', edge='#6C757D', fontsize=8)
    draw_arrow(ax, 3.0, 6.0, 3.8, 6.0)
    
    draw_box(ax, 7.0, 5.5, 2.5, 1.0, 'CNN Backbone\n(Conv2D + BN + ReLU\n+ MaxPool)', color='#D4EDDA', edge='#28A745')
    draw_arrow(ax, 6.3, 6.0, 7.0, 6.0)
    
    # Stats branch
    draw_box(ax, 3.8, 3.5, 2.5, 1.0, 'Blind Statistics\n(16-dim)', color='#D1ECF1', edge='#17A2B8')
    draw_box(ax, 3.8, 4.7, 2.5, 0.6, 'Stats Extraction\n(Cumulants, Moments)', color='#E2E3E5', edge='#6C757D', fontsize=8)
    draw_arrow(ax, 3.0, 4.0, 3.8, 4.0)
    
    draw_box(ax, 7.0, 3.5, 2.5, 0.8, 'MLP Projection\n(16→128)', color='#D1ECF1', edge='#17A2B8')
    draw_arrow(ax, 6.3, 4.0, 7.0, 4.0)
    
    # Feature vectors
    draw_box(ax, 7.0, 6.7, 2.5, 0.5, 'Flatten → 256-d', color='#E2E3E5', edge='#6C757D', fontsize=8)
    draw_arrow(ax, 8.25, 6.0, 8.25, 6.7)
    
    # Fusion
    draw_box(ax, 10.0, 3.5, 2.5, 1.2, 'Feature Fusion\n(Concat + FC\n256+128→128)', color='#F8D7DA', edge='#DC3545', fontsize=10, bold=True)
    draw_arrow(ax, 9.5, 6.0, 10.0, 4.5)
    draw_arrow(ax, 9.5, 4.0, 10.0, 4.0)
    
    # Classifier
    draw_box(ax, 10.0, 1.8, 2.5, 1.0, 'Classifier\n(128→64→6)', color='#F8D7DA', edge='#DC3545', fontsize=10, bold=True)
    draw_arrow(ax, 11.25, 3.5, 11.25, 2.8)
    
    # Output
    draw_box(ax, 10.0, 0.3, 2.5, 0.8, 'Softmax\nQPSK/16/32/64/128/256QAM', color='#E8DAEF', edge='#8E44AD', fontsize=10, bold=True)
    draw_arrow(ax, 11.25, 1.8, 11.25, 1.1)
    
    # Section labels
    ax.text(5.05, 7.5, 'CDM Branch', ha='center', fontsize=12, fontweight='bold', color='#28A745')
    ax.text(5.05, 5.2, 'Statistics Branch', ha='center', fontsize=12, fontweight='bold', color='#17A2B8')
    ax.text(11.25, 5.2, 'Fusion &', ha='center', fontsize=12, fontweight='bold', color='#DC3545')
    ax.text(11.25, 4.9, 'Classification', ha='center', fontsize=12, fontweight='bold', color='#DC3545')
    
    plt.tight_layout(pad=1)
    plt.savefig(os.path.join(OUT_DIR, 'fig_model_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] fig_model_architecture.png")


# ============================================================
# 图1: CDM 星座密度图矩阵 (从预处理的 .mat 文件中读取)
# ============================================================
def draw_cdm_matrix():
    """
    从预处理 .mat 文件中读取 cdm64 数据，绘制星座密度图矩阵
    weak/strong × 6 modulations
    """
    import scipy.io as sio
    preproc_dir = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\preprocessed_uniform_qam_rx_lowmem\2026.06.28"
    mod_names = ['QPSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']
    turb_names = ['weak', 'strong']
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
    
    for i, turb in enumerate(turb_names):
        for j, mod in enumerate(mod_names):
            ax = axes[i, j]
            mod_dir = os.path.join(preproc_dir, mod, turb)
            
            if not os.path.isdir(mod_dir):
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
            else:
                sig_dirs = sorted(os.listdir(mod_dir))
                if not sig_dirs:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
                else:
                    # Use first signal directory, first frame
                    frame_path = os.path.join(mod_dir, sig_dirs[0], 'frame_01.mat')
                    try:
                        mat = sio.loadmat(frame_path)
                        cdm = mat['cdm64']  # (64, 64)
                        ax.imshow(cdm, cmap='hot', aspect='auto', origin='lower')
                    except Exception as e:
                        ax.text(0.5, 0.5, 'Err', ha='center', va='center', transform=ax.transAxes, fontsize=11, color='red')
            
            if i == 0:
                ax.set_title(mod, fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{turb.capitalize()}\nTurbulence', fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('Constellation Density Map (CDM) Matrix', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_cdm_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] fig_cdm_matrix.png")


# ============================================================
# 图2: 真实星座图矩阵 (从预处理的 .mat 文件中读取 rx_sc)
# ============================================================
def draw_raw_constellation():
    """
    从预处理 .mat 文件中读取 rx_sc (均衡后 I/Q)，绘制星座图矩阵
    weak/strong × 6 modulations
    """
    import scipy.io as sio
    preproc_dir = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\preprocessed_uniform_qam_rx_lowmem\2026.06.28"
    mod_names = ['QPSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']
    turb_names = ['weak', 'strong']
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
    
    for i, turb in enumerate(turb_names):
        for j, mod in enumerate(mod_names):
            ax = axes[i, j]
            mod_dir = os.path.join(preproc_dir, mod, turb)
            
            if not os.path.isdir(mod_dir):
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
            else:
                sig_dirs = sorted(os.listdir(mod_dir))
                if not sig_dirs:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
                else:
                    frame_path = os.path.join(mod_dir, sig_dirs[0], 'frame_01.mat')
                    try:
                        mat = sio.loadmat(frame_path)
                        rx_sc = mat['rx_sc'].flatten()  # complex
                        # Downsample if too large
                        if len(rx_sc) > 5000:
                            idx = np.linspace(0, len(rx_sc)-1, 5000, dtype=int)
                            rx_sc = rx_sc[idx]
                        ax.scatter(rx_sc.real, rx_sc.imag, s=1, alpha=0.5, color='#1a73e8', edgecolors='none')
                        # Auto limits based on data
                        lim = max(abs(rx_sc.real).max(), abs(rx_sc.imag).max()) * 1.2
                        ax.set_xlim(-lim, lim)
                        ax.set_ylim(-lim, lim)
                    except Exception as e:
                        ax.text(0.5, 0.5, 'Err', ha='center', va='center', transform=ax.transAxes, fontsize=11, color='red')
            
            if i == 0:
                ax.set_title(mod, fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{turb.capitalize()}\nTurbulence', fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('Received Constellation Diagrams (After Equalization)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_constellation_raw.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] fig_constellation_raw.png")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    draw_ablation_bar()
    draw_model_architecture()
    draw_cdm_matrix()
    draw_raw_constellation()
    print("\nAll figures saved to:", OUT_DIR)