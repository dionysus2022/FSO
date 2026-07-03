"""
plot_confusion_matrix.py - Load trained model + test data, plot confusion matrix
Usage:
  python plot_confusion_matrix.py --model_arch cnn --ckpt ./results/checkpoints/exp_cnn_best.pth
"""
import os, re, glob, argparse, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models import ConstellationCNN, ConstellationUNet, ConstellationViT, UNet_ResNet18_E2E

CLASS_NAMES = ['BPSK', 'QPSK', '16QAM', '64QAM', '256QAM']
LABEL_MAP = {1:0, 2:1, 4:2, 6:3, 8:4}

class ExpDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.mat')))
        print(f'[Dataset] {len(self.files)} files')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat = sio.loadmat(self.files[idx])
        if 'Dual_CDM' in mat:
            cdm = mat['Dual_CDM'].astype(np.float32)
        else:
            cdm = mat['Distorted_CDM'].astype(np.float32)
            cdm = np.stack([cdm, cdm], axis=0)

        label = LABEL_MAP.get(int(mat['Label_Bits'][0,0]), 0)
        return torch.from_numpy(cdm), label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=r'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\dataset_cdm_exp_all')
    parser.add_argument('--model_arch', default='cnn', choices=['cnn','vit','unet_resnet18'])
    parser.add_argument('--ckpt', default='./results/checkpoints/exp_cnn_best.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    dataset = ExpDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Build model
    in_ch = 2 if args.model_arch in ['vit', 'unet_resnet18'] else 1
    if args.model_arch == 'cnn':
        model = ConstellationCNN(in_channels=in_ch).to(device)
    elif args.model_arch == 'vit':
        model = ConstellationViT(in_channels=in_ch).to(device)
    elif args.model_arch == 'unet_resnet18':
        model = UNet_ResNet18_E2E(in_channels=in_ch, num_classes=5).to(device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    print(f'Model loaded: {args.ckpt}')

    # Evaluate
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

    acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
    print(f'\nAccuracy: {acc*100:.2f}% ({len(all_labels)} samples)')

    # Per-class
    for c in range(5):
        total = sum(1 for l in all_labels if l == c)
        correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l == c)
        print(f'  {CLASS_NAMES[c]}: {correct}/{total} = {correct/max(total,1)*100:.2f}%')

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds, labels=range(5))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix ({args.model_arch}, Acc={acc*100:.2f}%)')
    plt.tight_layout()

    os.makedirs('./results/figures', exist_ok=True)
    out = f'./results/figures/cm_{args.model_arch}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'\nSaved: {out}')

if __name__ == '__main__':
    main()
