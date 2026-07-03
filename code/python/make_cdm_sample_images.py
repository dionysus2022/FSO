# -*- coding: utf-8 -*-
"""
make_cdm_sample_images.py — 生成 6 种调制格式的 CDM 组合图（2×3 子图 + 共享 colorbar）。

CDM 已由 make_cdm_from_rx_sc 预计算（log1p + max 归一化，64×64）。
输出：单张组合图 PDF（矢量）+ 600 dpi PNG，符合正式论文图规范。
  - 删除每子图独立 colorbar，改为右侧单一共享 colorbar
  - 标注 "Log-normalized density"（已做 log1p 变换）
  - 子图标题：(a) QPSK … (f) 256QAM
  - 字号：标题 8 pt / 刻度 7 pt / 轴标签 8 pt / colorbar 7 pt
  - 每个子图正方形（aspect 1:1），colorbar 不挤压子图宽度
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = Path(r"D:\Project_code\FSO_research\results\stage2\fast_publish\real_cache_ab.npz")
OUT_DIR = Path(r"D:\Project_code\FSO_research\results\docs\paper_acp\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MOD_NAMES = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

# 论文字号（pt）
FS_TITLE = 8
FS_TICK = 7
FS_LABEL = 8
FS_CBAR = 7
DPI = 600


def main():
    data = np.load(CACHE, allow_pickle=True)
    X_cdm = data["X_cdm"]      # (N, 1, 64, 64)
    y = data["y"]
    turbs = data["turbs"]
    sources = data["sources"]
    split = data["split"]

    # Windows 上 Times New Roman 可用，回退 DejaVu Serif
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    # ---- 选取每种调制一张代表性 CDM（A 域 test weak 优先）----
    cdms = []
    meta = []
    for label, mod_name in enumerate(MOD_NAMES):
        mask = (y == label) & (sources == "A") & (split == "test") & (turbs == "weak")
        idx_arr = np.where(mask)[0]
        if len(idx_arr) == 0:
            mask = (y == label) & (sources == "A") & (split == "test")
            idx_arr = np.where(mask)[0]
        if len(idx_arr) == 0:
            mask = (y == label) & (sources == "A")
            idx_arr = np.where(mask)[0]
        if len(idx_arr) == 0:
            print(f"[WARN] No sample found for {mod_name}")
            cdms.append(None)
            meta.append(None)
            continue
        sel = int(idx_arr[0])
        cdms.append(X_cdm[sel, 0])
        meta.append((sel, str(turbs[sel]), str(sources[sel]), str(split[sel])))

    # ---- 组合图：2×3，每个子图正方形，右侧共享 colorbar ----
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    # 为右侧 colorbar 预留空间，colorbar 用独立 axes 不挤压子图
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.13, top=0.93,
                        wspace=0.32, hspace=0.42)

    im_last = None
    for k, ax in enumerate(axes.flat):
        if cdms[k] is None:
            ax.axis("off")
            continue
        # 正方形子图（box aspect 1:1）
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass
        im_last = ax.imshow(cdms[k], cmap="viridis", origin="lower",
                            extent=[-3.0, 3.0, -3.0, 3.0],
                            aspect="equal", vmin=0.0, vmax=1.0)
        ax.set_title(f"{PANEL_LABELS[k]} {MOD_NAMES[k]}", fontsize=FS_TITLE, pad=3)
        ax.tick_params(labelsize=FS_TICK)
        row, col = divmod(k, 3)
        # 仅外边缘放坐标轴标签，避免冗余
        if row == 1:
            ax.set_xlabel("In-phase", fontsize=FS_LABEL)
        if col == 0:
            ax.set_ylabel("Quadrature", fontsize=FS_LABEL)

    # 共享 colorbar（独立 axes，不挤压子图）
    cax = fig.add_axes([0.88, 0.13, 0.022, 0.80])
    cbar = fig.colorbar(im_last, cax=cax)
    cbar.set_label("Log-normalized density", fontsize=FS_CBAR)
    cbar.ax.tick_params(labelsize=FS_CBAR)

    out_pdf = OUT_DIR / "cdm_samples.pdf"
    out_png = OUT_DIR / "cdm_samples.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    for k, m in enumerate(meta):
        if m is not None:
            sel, turb, src, spl = m
            cdm = cdms[k]
            print(f"[OK] {MOD_NAMES[k]}: idx={sel}, turb={turb}, src={src}, split={spl} | "
                  f"cdm min={cdm.min():.4f} max={cdm.max():.4f}")
    print(f"     -> {out_pdf}")
    print(f"     -> {out_png}")


if __name__ == "__main__":
    main()
