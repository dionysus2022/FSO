import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio

LABEL_MAPPING = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4}
MODULATION_NAMES = {1: "BPSK", 2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}
NUM_CLASSES = 5
TURBULENCE_GROUPS = ["weak", "moderate", "strong"]

FILENAME_PATTERN = re.compile(
    r'screen_(weak|moderate|strong)_(\d+)_frame_(\d+)(?:_snr_(-?\d+))?_mod_(\d+)\.mat$'
)


class FSODataset(Dataset):
    def __init__(self, data_dir: str, use_gaf: bool = False, augment: bool = False):
        all_files = glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True)

        self.use_gaf = use_gaf
        self.augment = augment
        self.files = []
        skipped_old = 0
        for f in all_files:
            basename = os.path.basename(f)
            m = FILENAME_PATTERN.search(basename)
            if not m:
                continue
            mod_bits = int(m.group(5))
            if mod_bits not in LABEL_MAPPING:
                continue
            # GAF 模式下过滤旧格式 (无 _snr_ 字段 = 无 rx_IQ)
            if use_gaf and m.group(4) is None:
                skipped_old += 1
                continue
            self.files.append(f)
        if skipped_old:
            print(f"[GAF] 已过滤 {skipped_old} 个旧格式文件 (无 I/Q 数据)")

        self.files.sort()
        print(f"成功加载数据集，总样本数: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _parse_filename(mat_path: str):
        m = FILENAME_PATTERN.search(os.path.basename(mat_path))
        if m:
            turb = m.group(1)
            idx = int(m.group(2))
            frame = int(m.group(3))
            # group(4) is optional (_snr_N); absent for old-format files
            has_snr = m.group(4) is not None
            snr = int(m.group(4)) if has_snr else 15
            mod = int(m.group(5))
            return turb, idx, frame, snr, mod, has_snr
        return None

    @staticmethod
    def _augment_cdm(cdm_np: np.ndarray) -> np.ndarray:
        """数据扩增: 随机相位旋转 (±5°) + 微弱高斯白噪声注入"""
        from scipy.ndimage import rotate

        # 1. 随机相位旋转，模拟激光器线宽漂移
        angle = np.random.uniform(-5.0, 5.0)
        cdm_aug = rotate(cdm_np, angle, axes=(-2, -1), reshape=False,
                         order=1, mode='constant', cval=0.0)

        # 2. 微弱高斯白噪声，逼迫网络学主干特征
        noise = np.random.randn(*cdm_aug.shape).astype(np.float32) * 0.005
        cdm_aug = cdm_aug.astype(np.float32) + noise

        # 裁剪: CDM 是密度图，保持非负
        cdm_aug = np.clip(cdm_aug, 0.0, None)
        return cdm_aug

    def __getitem__(self, idx: int):
        mat_path = self.files[idx]
        mat_data = sio.loadmat(mat_path)
        distorted = mat_data["Distorted_CDM"].astype(np.float32)
        ideal = mat_data["Ideal_CDM"].astype(np.float32)

        label_bits_raw = mat_data["Label_Bits"]
        label_bits = int(np.squeeze(label_bits_raw))

        label = LABEL_MAPPING.get(label_bits, -1)
        parsed = FSODataset._parse_filename(mat_path)
        snr_value = parsed[3] if parsed else -1

        # ---- 数据扩增 (仅在训练时开启) ----
        if self.augment:
            distorted = self._augment_cdm(distorted)
            ideal = self._augment_cdm(ideal)

        # ---- GAF 双通道模式: CDM[0] + GAF[1] ----
        if self.use_gaf:
            from time_series_utils import complex_to_gaf

            # 优先读取 (N,2) 格式的 I/Q; 回退到复数格式
            if "rx_IQ" in mat_data:
                rx_iq = mat_data["rx_IQ"].astype(np.float64)
            elif "rx_symbols_flat" in mat_data:
                rx_flat = mat_data["rx_symbols_flat"].flatten()
                rx_iq = np.column_stack([rx_flat.real, rx_flat.imag])
            elif "rx_flat" in mat_data:
                rx_flat = mat_data["rx_flat"].flatten()
                rx_iq = np.column_stack([rx_flat.real, rx_flat.imag])
            else:
                rx_iq = None

            if rx_iq is not None and len(rx_iq) >= 64:
                gaf_img = complex_to_gaf(rx_iq, image_size=64)
                gaf_tensor = torch.from_numpy(gaf_img).unsqueeze(0)  # (1, 64, 64)
            else:
                # 无 I/Q 数据时使用零张量 (极少残余旧文件)
                gaf_tensor = torch.zeros(1, 64, 64, dtype=torch.float32)

            # 双通道拼接: (2, 64, 64)
            distorted_gaf = torch.cat([
                torch.from_numpy(distorted).unsqueeze(0),
                gaf_tensor,
            ], dim=0)
            ideal_gaf = torch.from_numpy(ideal).unsqueeze(0)  # 仅 CDM
            label = torch.tensor(label, dtype=torch.long)
            return distorted_gaf, ideal_gaf, label, snr_value

        # ---- 纯 CDM 模式 (向后兼容) ----
        distorted = torch.from_numpy(distorted).unsqueeze(0)
        ideal = torch.from_numpy(ideal).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return distorted, ideal, label, snr_value


def get_dataloaders(data_dir: str, batch_size: int = 64, split_ratio: float = 0.8,
                    num_workers: int = None, seed: int = 42,
                    use_gaf: bool = False, augment: bool = True):
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    from torch.utils.data import DataLoader, Subset
    import random

    # pin_memory 只在 GPU 模式下有意义
    use_pin = torch.cuda.is_available()

    # 先创建无扩增的完整数据集获取文件列表
    full_dataset = FSODataset(data_dir, use_gaf=use_gaf, augment=False)
    n_total = len(full_dataset)
    n_train = int(n_total * split_ratio)

    # 按固定种子打乱并划分索引
    indices = list(range(n_total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # 用 Subset + 独立 dataset 实例实现训练集扩增、测试集不做扩增
    train_dataset = FSODataset(data_dir, use_gaf=use_gaf, augment=augment)
    test_dataset = FSODataset(data_dir, use_gaf=use_gaf, augment=False)
    train_ds = Subset(train_dataset, train_idx)
    test_ds = Subset(test_dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin)

    return train_loader, test_loader


if __name__ == "__main__":
    ds = FSODataset(r"./data")
    print(f"Dataset len: {len(ds)}")
    if len(ds) > 0:
        d, i, l, s = ds[0]
        print(f"Sample 0: distorted={d.shape}, ideal={i.shape}, label={l}, snr={s}")
        print(f"Distorted range: [{d.min():.4f}, {d.max():.4f}]")
        print(f"Ideal range: [{i.min():.4f}, {i.max():.4f}]")
