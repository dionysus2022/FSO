# -*- coding: utf-8 -*-
"""
eval_awgn_robustness_cdm_stats.py

附加噪声鲁棒性测试：
在真实测试样本 rx_sc 上加入复高斯噪声，重新生成 CDM 和 blind_stats，
测试 Accuracy / Macro-F1 随 Additional SNR 的变化。

示例：
python eval_awgn_robustness_cdm_stats.py ^
  --model-dirs "D:\\...\\seed_2026\\weak_cdm_stats" "D:\\...\\seed_2026\\strong_cdm_stats" ^
  --snrs clean 30 25 20 15 10 5 0 ^
  --trials 5
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False

MOD_NAMES = ["QPSK","16QAM","32QAM","64QAM","128QAM","256QAM"]

def load_rx_sc(path: Path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    z = np.asarray(m["rx_sc"]); z = np.squeeze(z)
    if not np.iscomplexobj(z): z = z.astype(np.float32).astype(np.complex64)
    return z.astype(np.complex64)

def add_awgn(x, snr_db):
    if snr_db is None: return x
    z = x.astype(np.complex64)
    mask = np.isfinite(z.real) & np.isfinite(z.imag)
    if mask.sum() == 0: return z
    p = np.mean(np.abs(z[mask])**2)
    sigma2 = p / (10.0**(float(snr_db)/10.0))
    n = np.sqrt(sigma2/2.0) * (np.random.randn(*z.shape) + 1j*np.random.randn(*z.shape))
    return (z + n.astype(np.complex64)).astype(np.complex64)

def make_cdm(rx_sc, nbin=64, clip_val=3.0):
    z = rx_sc.reshape(-1).astype(np.complex64)
    z = z[np.isfinite(z.real) & np.isfinite(z.imag)]
    if z.size == 0: return np.zeros((1,nbin,nbin), dtype=np.float32)
    z = z - np.mean(z); z = z / np.sqrt(np.mean(np.abs(z)**2) + 1e-12)
    zr = np.clip(z.real, -clip_val, clip_val); zi = np.clip(z.imag, -clip_val, clip_val)
    edges = np.linspace(-clip_val, clip_val, nbin+1)
    H,_,_ = np.histogram2d(zi, zr, bins=[edges, edges])
    H = np.log1p(H); H = H/(np.max(H)+1e-12)
    return H[None,:,:].astype(np.float32)

def skew_np(x):
    x=np.asarray(x,dtype=np.float64); x=x[np.isfinite(x)]
    if x.size<3: return np.nan
    return np.mean(((x-np.mean(x))/(np.std(x)+1e-12))**3)

def kurt_np(x):
    x=np.asarray(x,dtype=np.float64); x=x[np.isfinite(x)]
    if x.size<4: return np.nan
    return np.mean(((x-np.mean(x))/(np.std(x)+1e-12))**4)

def make_stats(rx_sc):
    z = rx_sc.reshape(-1).astype(np.complex64)
    z = z[np.isfinite(z.real) & np.isfinite(z.imag)]
    if z.size == 0: return np.full((16,), np.nan, dtype=np.float32)
    z = z - np.mean(z); z = z/np.sqrt(np.mean(np.abs(z)**2)+1e-12)
    a=np.abs(z); i=z.real; q=z.imag; dph=np.diff(np.unwrap(np.angle(z)))
    iq_corr = 0.0 if (np.std(i)<1e-12 or np.std(q)<1e-12) else float(np.corrcoef(i,q)[0,1])
    return np.array([np.mean(a),np.std(a),skew_np(a),kurt_np(a),10*np.log10(np.max(np.abs(z)**2)/(np.mean(np.abs(z)**2)+1e-12)),np.mean(i),np.std(i),skew_np(i),kurt_np(i),np.mean(q),np.std(q),skew_np(q),kurt_np(q),np.std(dph) if dph.size>0 else np.nan,np.abs(np.mean(np.exp(1j*np.angle(z)))),iq_corr], dtype=np.float32)

class CDMBranch(nn.Module):
    def __init__(self,out_dim=64,dropout=0.125):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1,bias=False),nn.BatchNorm2d(16),nn.SiLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.SiLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.SiLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(64,96,3,padding=1,bias=False),nn.BatchNorm2d(96),nn.SiLU(inplace=True),nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),nn.Linear(96,out_dim),nn.SiLU(inplace=True),nn.Dropout(dropout))
    def forward(self,x): return self.fc(self.net(x))
class StatsBranch(nn.Module):
    def __init__(self,in_dim=16,out_dim=32,dropout=0.125):
        super().__init__(); self.net=nn.Sequential(nn.Linear(in_dim,64),nn.BatchNorm1d(64),nn.SiLU(inplace=True),nn.Dropout(dropout),nn.Linear(64,out_dim),nn.SiLU(inplace=True))
    def forward(self,x): return self.net(x)
class CDMStatsNet(nn.Module):
    def __init__(self,stats_dim=16,dropout=0.25):
        super().__init__(); self.cdm_branch=CDMBranch(64,dropout*0.5); self.stats_branch=StatsBranch(stats_dim,32,dropout*0.5)
        self.classifier=nn.Sequential(nn.Linear(96,128),nn.BatchNorm1d(128),nn.SiLU(inplace=True),nn.Dropout(dropout),nn.Linear(128,6))
    def forward(self,cdm,stats): return self.classifier(torch.cat([self.cdm_branch(cdm),self.stats_branch(stats)],dim=1))

def macro_f1(y,p,n=6):
    fs=[]
    for c in range(n):
        tp=np.sum((y==c)&(p==c)); fp=np.sum((y!=c)&(p==c)); fn=np.sum((y==c)&(p!=c))
        pr=tp/(tp+fp+1e-12); re=tp/(tp+fn+1e-12); fs.append(2*pr*re/(pr+re+1e-12))
    return float(np.mean(fs))

def load_model(model_dir, device):
    ckpt=torch.load(model_dir/'best_model.pt',map_location=device,weights_only=False)
    mean=np.asarray(ckpt['stats_mean'],dtype=np.float32).reshape(-1)
    std=np.asarray(ckpt['stats_std'],dtype=np.float32).reshape(-1); std[std<1e-6]=1.0
    dropout=ckpt.get('args',{}).get('dropout',0.25)
    model=CDMStatsNet(stats_dim=len(mean),dropout=dropout).to(device)
    model.load_state_dict(ckpt['model_state'], strict=True); model.eval()
    return model,mean,std,ckpt

def parse_snr(s):
    return None if str(s).lower() in ['clean','none','inf'] else float(s)

@torch.no_grad()
def eval_snr(model, rows, mean, std, snr_db, trials, batch_size, device):
    ys=[]; ps=[]
    for _ in range(trials):
        cdms=[]; statss=[]; yb=[]
        for _,r in rows.iterrows():
            z=add_awgn(load_rx_sc(Path(r['out_mat'])), snr_db)
            cdm=make_cdm(z); st=make_stats(z); st=np.nan_to_num(st,nan=0.0,posinf=0.0,neginf=0.0); st=(st-mean)/std; st=np.nan_to_num(st,nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32)
            cdms.append(cdm); statss.append(st); yb.append(int(r['mod_label']))
            if len(yb)>=batch_size:
                pred=model(torch.from_numpy(np.stack(cdms)).to(device), torch.from_numpy(np.stack(statss)).to(device)).argmax(1).cpu().numpy()
                ys.extend(yb); ps.extend(pred.tolist()); cdms=[]; statss=[]; yb=[]
        if yb:
            pred=model(torch.from_numpy(np.stack(cdms)).to(device), torch.from_numpy(np.stack(statss)).to(device)).argmax(1).cpu().numpy()
            ys.extend(yb); ps.extend(pred.tolist())
    y=np.asarray(ys); p=np.asarray(ps)
    return float(np.mean(y==p)), macro_f1(y,p,6)

def plot(df,path):
    if not MPL_OK: return
    x=[]
    for s in df['snr_label']:
        x.append(35 if s=='clean' else float(s))
    d=df.copy(); d['x']=x; d=d.sort_values('x')
    plt.figure(figsize=(7,4.6)); plt.plot(d['x'],d['accuracy'],marker='o',label='Accuracy'); plt.plot(d['x'],d['f1_macro'],marker='s',label='Macro-F1')
    plt.xlabel('Additional SNR (dB); clean plotted at 35 dB'); plt.ylabel('Score'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model-dirs',nargs='+',required=True); ap.add_argument('--snrs',nargs='+',default=['clean','30','25','20','15','10','5','0']); ap.add_argument('--trials',type=int,default=5); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args(); np.random.seed(args.seed); torch.manual_seed(args.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    all_rows=[]
    for md in args.model_dirs:
        model_dir=Path(md); print('\nModel:',model_dir); split=pd.read_csv(model_dir/'split.csv'); test=split[split['split'].astype(str)=='test'].copy(); print('test samples:',len(test))
        model,mean,std,ckpt=load_model(model_dir,device); rows=[]
        for s in args.snrs:
            t=time.time(); snr=parse_snr(s); acc,f1=eval_snr(model,test,mean,std,snr,args.trials,args.batch_size,device); print(f'  SNR={s:>5s} acc={acc:.4f} f1={f1:.4f} time={time.time()-t:.1f}s')
            rec={'model_dir':str(model_dir),'turb_name':ckpt.get('turb_name',model_dir.name),'snr_label':str(s).lower(),'snr_db':np.nan if snr is None else snr,'accuracy':acc,'f1_macro':f1,'trials':args.trials,'num_test_samples':len(test)}; rows.append(rec); all_rows.append(rec)
        out=pd.DataFrame(rows); out.to_csv(model_dir/'awgn_robustness.csv',index=False,encoding='utf-8-sig'); plot(out,model_dir/'awgn_robustness.png')
    pd.DataFrame(all_rows).to_csv(Path(args.model_dirs[0]).parent/'awgn_robustness_all.csv',index=False,encoding='utf-8-sig')
if __name__=='__main__': main()
