"""
train_exp.py - Train on experimental data (clean version)
Usage:
  python train_exp.py --model_arch cnn --epochs 50
  python train_exp.py --model_arch vit --epochs 30
  python train_exp.py --model_arch unet_resnet18 --epochs 50
"""
import os, re, glob, argparse, numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.io as sio

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import ConstellationCNN, ConstellationViT, UNet_ResNet18_E2E
from utils import set_seed, get_device, setup_logger

CLASS_NAMES = ['BPSK', 'QPSK', '16QAM', '64QAM', '256QAM']
LABEL_MAP = {1:0, 2:1, 4:2, 6:3, 8:4}

class ExpDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.mat')))
        self.augment = augment
        self.label_map = LABEL_MAP

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat = sio.loadmat(self.files[idx])
        if 'Dual_CDM' in mat:
            cdm = mat['Dual_CDM'].astype(np.float32)
        elif 'Distorted_CDM' in mat:
            cdm = mat['Distorted_CDM'].astype(np.float32)
            cdm = np.stack([cdm, cdm], axis=0)
        else:
            raise KeyError(f'No CDM found in {self.files[idx]}')

        label = self.label_map.get(int(mat['Label_Bits'][0,0]), 0)

        if self.augment:
            from scipy.ndimage import rotate
            angle = np.random.uniform(-5.0, 5.0)
            ch1 = rotate(cdm[0], angle, axes=(-2,-1), reshape=False, order=1, mode='constant', cval=0.0)
            ch2 = rotate(cdm[1], angle, axes=(-2,-1), reshape=False, order=1, mode='constant', cval=0.0)
            cdm = np.stack([ch1, ch2], axis=0)

        return torch.from_numpy(cdm), label

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=r'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\dataset_cdm_exp_all')
    p.add_argument('--model_arch', default='cnn', choices=['cnn','vit','unet_resnet18'])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split_ratio', type=float, default=0.7)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    logger = setup_logger(f'exp_{args.model_arch}')

    logger.info('='*60)
    logger.info(f'Model: {args.model_arch} | Data: {args.data_dir}')
    logger.info(f'Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}')
    logger.info('='*60)

    # Load dataset
    full = ExpDataset(args.data_dir, augment=False)
    n_total = len(full)
    n_train = int(n_total * args.split_ratio)
    train_ds, test_ds = torch.utils.data.random_split(full, [n_train, n_total-n_train])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    logger.info(f'Train: {n_train} | Test: {n_total-n_train}')

    # Model
    in_ch = 2 if args.model_arch in ['vit','unet_resnet18'] else 1
    if args.model_arch == 'cnn':
        model = ConstellationCNN(in_channels=in_ch).to(device)
    elif args.model_arch == 'vit':
        model = ConstellationViT(in_channels=in_ch).to(device)
    elif args.model_arch == 'unet_resnet18':
        model = UNet_ResNet18_E2E(in_channels=in_ch, num_classes=5).to(device)

    logger.info(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_acc = 0.0
    early_stop = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        train_correct, train_total = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        train_acc = train_correct / train_total

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                preds = logits.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        test_acc = test_correct / test_total

        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs('./results/checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'./results/checkpoints/exp_{args.model_arch}_best.pth')
            logger.info(f'Epoch {epoch}: Train={train_acc:.4f} Test={test_acc:.4f} [BEST]')
            early_stop = 0
        else:
            logger.info(f'Epoch {epoch}: Train={train_acc:.4f} Test={test_acc:.4f}')
            early_stop += 1

        if early_stop >= 10:
            logger.info(f'Early stop at epoch {epoch}')
            break

    logger.info(f'Best test accuracy: {best_acc*100:.2f}%')

    # Confusion matrix on best model
    model.load_state_dict(torch.load(f'./results/checkpoints/exp_{args.model_arch}_best.pth'))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(all_labels, all_preds, labels=range(5))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix ({args.model_arch}, Acc={best_acc*100:.2f}%)')
    plt.tight_layout()
    os.makedirs('./results/figures', exist_ok=True)
    plt.savefig(f'./results/figures/cm_{args.model_arch}.png', dpi=150)
    plt.close()
    logger.info(f'Confusion matrix saved to ./results/figures/cm_{args.model_arch}.png')

if __name__ == '__main__':
    main()
