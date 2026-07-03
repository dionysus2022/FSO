"""Regenerate confusion matrix plots with Chinese font support."""
import os, sys, numpy as np, torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "python"))

from dataset import FSODataset, TURBULENCE_GROUPS
from models import ConstellationCNN, ConstellationUNet

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CKPT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
CLASS_NAMES = ["BPSK", "QPSK", "16QAM", "64QAM", "256QAM"]

os.makedirs(SAVE_DIR, exist_ok=True)

# Load models
cnn = ConstellationCNN(NUM_CLASSES).to(DEVICE)
unet = ConstellationUNet(1, 1, True).to(DEVICE)
cnn.load_state_dict(torch.load(os.path.join(CKPT_DIR, "cnn_ideal_best.pth"), map_location=DEVICE, weights_only=False)["model_state_dict"])
unet.load_state_dict(torch.load(os.path.join(CKPT_DIR, "gnn_denoiser_best.pth"), map_location=DEVICE, weights_only=False)["model_state_dict"])
cnn.eval(); unet.eval()
print("Models loaded")

# Data
ds = FSODataset(DATA_DIR)
loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

file_turb_map = []
for f in ds.files:
    fn = os.path.basename(f)
    t = next((t for t in TURBULENCE_GROUPS if t in fn), None)
    file_turb_map.append(t)

grp_cascade = {g: {"preds": [], "labels": []} for g in TURBULENCE_GROUPS}
grp_baseline = {g: {"preds": [], "labels": []} for g in TURBULENCE_GROUPS}
sample_idx = 0

@torch.no_grad()
def run():
    global sample_idx
    for distorted, ideal, labels, snr_values in loader:
        d = distorted.to(DEVICE)
        l = labels.to(DEVICE)
        repaired = unet(d)
        pc = cnn(repaired).argmax(dim=1).cpu().tolist()
        pb = cnn(d).argmax(dim=1).cpu().tolist()
        lc = l.cpu().tolist()
        for i in range(d.size(0)):
            if sample_idx >= len(file_turb_map): return
            turb = file_turb_map[sample_idx]
            sample_idx += 1
            if turb is None: continue
            grp_cascade[turb]["preds"].append(pc[i])
            grp_cascade[turb]["labels"].append(lc[i])
            grp_baseline[turb]["preds"].append(pb[i])
            grp_baseline[turb]["labels"].append(lc[i])

run()
n = sum(len(grp_cascade[t]["preds"]) for t in TURBULENCE_GROUPS)
print(f"Collected {n} samples")

# Plot
for data, prefix, fn in [
    (grp_cascade, "U-Net + CNN (Proposed)", "confusion_cascade_turbulence.png"),
    (grp_baseline, "CNN (Baseline)", "confusion_baseline_turbulence.png"),
]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    turb_names = {"weak": "Weak", "moderate": "Moderate", "strong": "Strong"}
    for ax, turb in zip(axes, TURBULENCE_GROUPS):
        total = len(data[turb]["preds"])
        if total == 0: continue
        cm = confusion_matrix(data[turb]["labels"], data[turb]["preds"], labels=list(range(NUM_CLASSES)))
        acc = np.trace(cm) / cm.sum() * 100
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
        ax.set_title(f"{turb_names[turb]}\nAcc: {acc:.1f}%", fontsize=12)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.suptitle(f"{prefix} - Per-Turbulence Confusion Matrices", fontweight="bold", fontsize=14)
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, fn)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fn}")

print("Done")
