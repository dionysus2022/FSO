"""
论文补充分析脚本
1. 每类 Precision/Recall/F1 表格 (5 seed 平均)
2. SNR/EVM 分布箱线图
3. 模型复杂度表格 (参数量/训练时间/推理时间)
"""
import os, sys, json, glob, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

OUT_DIR = r"D:\Project_code\FSO_research\results\docs\figs\训练结果09"
os.makedirs(OUT_DIR, exist_ok=True)

SEED_DIRS = [2026, 2027, 2028, 2029, 2030]
BASE = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\recognition_mrf_lowmem"
MOD_NAMES = ['QPSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']
TURB_NAMES = ['weak', 'strong']

# ============================================================
# 1. 每类 Precision/Recall/F1 表格 (5 seed 平均)
# ============================================================
def aggregate_per_class_metrics():
    """从 5 seed 的 metrics.json 中提取 test 分类报告并平均"""
    results = {}
    
    for turb in TURB_NAMES:
        per_mod = {m: {'precision': [], 'recall': [], 'f1-score': []} for m in MOD_NAMES}
        overall_acc = []
        overall_f1 = []
        
        for seed in SEED_DIRS:
            mpath = os.path.join(BASE, f"seed_{seed}", f"{turb}_cdm_stats", "metrics.json")
            if not os.path.exists(mpath):
                print(f"[WARN] Missing: {mpath}")
                continue
            with open(mpath, 'r') as f:
                m = json.load(f)
            cr = m['metrics']['test']['classification_report']
            for mod in MOD_NAMES:
                if mod in cr:
                    per_mod[mod]['precision'].append(cr[mod]['precision'])
                    per_mod[mod]['recall'].append(cr[mod]['recall'])
                    per_mod[mod]['f1-score'].append(cr[mod]['f1-score'])
            overall_acc.append(m['metrics']['test']['accuracy'])
            overall_f1.append(m['metrics']['test']['f1_macro'])
        
        results[turb] = {
            'per_mod': per_mod,
            'acc_mean': np.mean(overall_acc) * 100,
            'acc_std': np.std(overall_acc) * 100,
            'f1_mean': np.mean(overall_f1),
            'f1_std': np.std(overall_f1),
        }
    
    return results

def print_per_class_table(results):
    """生成 Markdown 表格"""
    lines = []
    lines.append("## 每类 Precision / Recall / F1 (5 Seed 平均)\n")
    lines.append("| 湍流 | 调制格式 | Precision | Recall | F1-score |")
    lines.append("|:----:|:--------:|:---------:|:------:|:--------:|")
    
    for turb in TURB_NAMES:
        r = results[turb]
        for mod in MOD_NAMES:
            pm = r['per_mod'][mod]
            p = np.mean(pm['precision']) * 100
            rec = np.mean(pm['recall']) * 100
            f1 = np.mean(pm['f1-score']) * 100
            p_std = np.std(pm['precision']) * 100
            rec_std = np.std(pm['recall']) * 100
            f1_std = np.std(pm['f1-score']) * 100
            lines.append(f"| {turb} | {mod} | {p:.2f}+-{p_std:.2f} | {rec:.2f}+-{rec_std:.2f} | {f1:.2f}+-{f1_std:.2f} |")
    
    lines.append("")
    for turb in TURB_NAMES:
        r = results[turb]
        lines.append(f"\n**{turb} Overall**: Acc = {r['acc_mean']:.2f}+-{r['acc_std']:.2f}%, F1 = {r['f1_mean']:.4f}+-{r['f1_std']:.4f}")
    
    return '\n'.join(lines)


# ============================================================
# 2. SNR/EVM 分布箱线图
# ============================================================
def get_ideal_constellation(mod_name):
    """返回归一化到单位平均功率的理想星座点 (N, 2)"""
    if mod_name == 'QPSK':
        pts = np.array([(1,1),(1,-1),(-1,1),(-1,-1)], dtype=float)
    elif mod_name == '16QAM':
        levels = np.array([-3,-1,1,3], dtype=float)
        pts = np.array([(i,q) for i in levels for q in levels], dtype=float)
    elif mod_name == '32QAM':
        # 32QAM cross: remove 4 corner points from 6x6 grid
        levels = np.array([-5,-3,-1,1,3,5], dtype=float)
        pts = np.array([(i,q) for i in levels for q in levels], dtype=float)
        # Remove 4 corners
        corners = [(-5,-5),(-5,5),(5,-5),(5,5)]
        pts = np.array([p for p in pts if tuple(p) not in corners], dtype=float)
    elif mod_name == '64QAM':
        levels = np.array([-7,-5,-3,-1,1,3,5,7], dtype=float)
        pts = np.array([(i,q) for i in levels for q in levels], dtype=float)
    elif mod_name == '128QAM':
        # 128QAM cross: remove 8 corner points from 12x12 grid
        levels = np.array([-11,-9,-7,-5,-3,-1,1,3,5,7,9,11], dtype=float)
        pts = np.array([(i,q) for i in levels for q in levels], dtype=float)
        # Remove outer corners (16 points to get 128)
        mask = []
        for p in pts:
            if abs(p[0]) == 11 and abs(p[1]) == 11:
                mask.append(False)
            elif abs(p[0]) == 11 and abs(p[1]) == 9:
                mask.append(False)
            elif abs(p[0]) == 9 and abs(p[1]) == 11:
                mask.append(False)
            else:
                mask.append(True)
        pts = pts[mask]
    elif mod_name == '256QAM':
        levels = np.array([-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15], dtype=float)
        pts = np.array([(i,q) for i in levels for q in levels], dtype=float)
    else:
        raise ValueError(f"Unknown modulation: {mod_name}")
    
    # Normalize to unit average power
    power = np.mean(pts[:,0]**2 + pts[:,1]**2)
    pts = pts / np.sqrt(power)
    return pts


def compute_evm_snr(rx_sc, mod_name):
    """从均衡后星座点计算 EVM 和等效 SNR
    rx_sc: (2, N) I/Q samples
    Returns: (evm, snr_db)
    """
    I = rx_sc[0, :].flatten().astype(float)
    Q = rx_sc[1, :].flatten().astype(float)
    
    # Normalize to unit average power
    power = np.mean(I**2 + Q**2)
    if power < 1e-12:
        return np.nan, np.nan
    I = I / np.sqrt(power)
    Q = Q / np.sqrt(power)
    
    ideal = get_ideal_constellation(mod_name)
    
    # For each received point, find nearest ideal point (vectorized)
    # Shape: (N_rx, 1) - (1, N_ideal) = (N_rx, N_ideal)
    dx = I[:, None] - ideal[None, :, 0]
    dy = Q[:, None] - ideal[None, :, 1]
    dists_sq = dx**2 + dy**2
    min_dists = np.sqrt(np.min(dists_sq, axis=1))
    
    evm = np.sqrt(np.mean(min_dists**2))
    snr_db = -20 * np.log10(evm) if evm > 0 else np.nan
    return evm, snr_db


def draw_snr_boxplot():
    """从预处理 .mat 文件中读取 rx_sc，计算 EVM/SNR 绘制箱线图"""
    preproc_dir = r"D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\preprocessed_uniform_qam_rx_lowmem\2026.06.28"
    
    # First collect all SNR data (compute from EVM)
    snr_data = {}
    for turb in TURB_NAMES:
        for mod in MOD_NAMES:
            mod_dir = os.path.join(preproc_dir, mod, turb)
            snr_values = []
            if os.path.isdir(mod_dir):
                for sig_dir in sorted(os.listdir(mod_dir)):
                    sig_path = os.path.join(mod_dir, sig_dir)
                    if not os.path.isdir(sig_path):
                        continue
                    for frame_file in sorted(glob.glob(os.path.join(sig_path, "frame_*.mat"))):
                        try:
                            mat = sio.loadmat(frame_file)
                            rx_sc = mat['rx_sc']  # (2, N)
                            evm, snr = compute_evm_snr(rx_sc, mod)
                            if not np.isnan(snr):
                                snr_values.append(snr)
                        except Exception as ex:
                            pass
            snr_data[(turb, mod)] = snr_values
            print(f"  {turb}/{mod}: {len(snr_values)} frames, SNR mean={np.mean(snr_values):.2f} dB" if snr_values else f"  {turb}/{mod}: no data")
    
    # Draw boxplot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    positions = []
    data_list = []
    colors = []
    pos = 1
    
    for turb in TURB_NAMES:
        for mod in MOD_NAMES:
            data_list.append(snr_data[(turb, mod)])
            positions.append(pos)
            colors.append('#4C72B0' if turb == 'weak' else '#DD8452')
            pos += 1
        pos += 0.8  # gap between groups
    
    bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # X-axis: show modulation names under each group
    tick_positions = []
    tick_labels = []
    pos = 1
    for turb in TURB_NAMES:
        for mod in MOD_NAMES:
            tick_positions.append(pos)
            tick_labels.append(mod)
            pos += 1
        pos += 0.8
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9, rotation=30, ha='right')
    
    # Group annotations
    mid_weak = (1 + 6) / 2
    mid_strong = (7.8 + 12.8) / 2
    y_top = max(max(d) for d in data_list if d) + 2
    ax.text(mid_weak, y_top, 'Weak Turbulence', ha='center', fontsize=12, fontweight='bold', color='#4C72B0')
    ax.text(mid_strong, y_top, 'Strong Turbulence', ha='center', fontsize=12, fontweight='bold', color='#DD8452')
    
    ax.set_ylabel('Frame SNR (dB)', fontsize=13, fontweight='bold')
    ax.set_title('SNR Distribution by Modulation and Turbulence', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    import matplotlib.patches as mpatches
    weak_patch = mpatches.Patch(color='#4C72B0', alpha=0.7, label='Weak')
    strong_patch = mpatches.Patch(color='#DD8452', alpha=0.7, label='Strong')
    ax.legend(handles=[weak_patch, strong_patch], fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_snr_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] fig_snr_boxplot.png")
    
    # Return SNR statistics
    snr_stats = []
    for turb in TURB_NAMES:
        for mod in MOD_NAMES:
            vals = snr_data[(turb, mod)]
            if vals:
                snr_stats.append({
                    'turb': turb, 'mod': mod,
                    'mean': np.mean(vals), 'std': np.std(vals),
                    'min': np.min(vals), 'max': np.max(vals),
                    'count': len(vals)
                })
    return pd.DataFrame(snr_stats)


# ============================================================
# 3. 模型复杂度表格
# ============================================================
def compute_model_complexity():
    """计算各模型的参数量、训练时间、推理时间"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # Read model definitions from the training script
    with open(r"D:\Project_code\FSO_research\code\python\train_mrf_lowmem.py", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Model section is lines 575-720 (0-indexed 574-719)
    model_code = ''.join(lines[574:720])
    
    exec_globals = {
        'torch': torch, 'nn': nn, 'F': F, 'np': np,
    }
    exec(model_code, exec_globals)
    
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_mode in ['stats', 'cdm', 'iq', 'cdm_stats', 'fusion']:
        try:
            model = exec_globals['MultiRepNet'](model_mode=model_mode, num_classes=6, iq_in_ch=1)
            model = model.to(device)
            model.eval()
            
            n_params = sum(p.numel() for p in model.parameters())
            
            batch_size = 1
            cdm = torch.randn(batch_size, 1, 64, 64).to(device)
            iq = torch.randn(batch_size, 1, 1).to(device)
            stats = torch.randn(batch_size, 16).to(device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(cdm, iq, stats)
            
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(cdm, iq, stats)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            
            infer_time = np.mean(times)
            
            for turb in ['weak', 'strong']:
                if model_mode == 'cdm_stats':
                    accs = []
                    train_secs = []
                    for seed in SEED_DIRS:
                        sp = os.path.join(BASE, f"seed_{seed}", f"summary_all_turbulence_cdm_stats.csv")
                        if os.path.exists(sp):
                            df = pd.read_csv(sp)
                            row = df[df['turb_name'] == turb]
                            if len(row) > 0:
                                accs.append(float(row['test_accuracy'].values[0]))
                                train_secs.append(float(row['train_seconds'].values[0]))
                    acc = np.mean(accs) * 100 if accs else 0
                    train_sec = np.mean(train_secs) if train_secs else 0
                else:
                    sp = os.path.join(BASE, "ablation", f"summary_all_turbulence_{model_mode}.csv")
                    if os.path.exists(sp):
                        df = pd.read_csv(sp)
                        row = df[df['turb_name'] == turb]
                        acc = float(row['test_accuracy'].values[0]) * 100 if len(row) > 0 else 0
                        train_sec = float(row['train_seconds'].values[0]) if len(row) > 0 else 0
                    else:
                        acc = 0
                        train_sec = 0
                
                results.append({
                    'model': model_mode,
                    'turb': turb,
                    'params': n_params,
                    'train_sec': round(train_sec, 2),
                    'infer_ms': round(infer_time, 3),
                    'acc': round(acc, 2)
                })
        except Exception as e:
            print(f"[WARN] {model_mode}: {e}")
            import traceback
            traceback.print_exc()
    
    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("1. Per-class P/R/F1 aggregation")
    print("=" * 60)
    results = aggregate_per_class_metrics()
    table_md = print_per_class_table(results)
    print(table_md)
    with open(os.path.join(OUT_DIR, 'per_class_metrics.md'), 'w', encoding='utf-8') as f:
        f.write(table_md)
    
    print("\n" + "=" * 60)
    print("2. SNR/EVM boxplot")
    print("=" * 60)
    snr_df = draw_snr_boxplot()
    print(snr_df.to_string(index=False))
    snr_df.to_csv(os.path.join(OUT_DIR, 'snr_stats.csv'), index=False)
    
    print("\n" + "=" * 60)
    print("3. Model complexity")
    print("=" * 60)
    complexity_df = compute_model_complexity()
    print(complexity_df.to_string(index=False))
    complexity_df.to_csv(os.path.join(OUT_DIR, 'model_complexity.csv'), index=False)
    
    print("\n\nAll supplementary analysis saved to:", OUT_DIR)