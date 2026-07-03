# -*- coding: utf-8 -*-
"""
train_cdm_stats_noiseaug.py

CDM+Stats 加噪训练版本，用于 Model-B：
训练集随机加入复高斯噪声，验证/测试保持干净。
训练完后，用 eval_awgn_robustness_cdm_stats.py 比较 Model-A vs Model-B。
"""
import argparse, json, time, random
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    import matplotlib.pyplot as plt
    MPL_OK=True
except Exception:
    MPL_OK=False
MOD_NAMES=["QPSK","16QAM","32QAM","64QAM","128QAM","256QAM"]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=True

def load_manifest(root):
    df=pd.read_csv(Path(root)/'manifest_all.csv'); df=df[df['status'].astype(str)=='ok'].copy(); df['out_mat']=df['out_mat'].astype(str)
    df=df[df['out_mat'].apply(lambda x: Path(x).exists())].copy(); df['mod_label']=df['mod_label'].astype(int); df['turb_name']=df['turb_name'].astype(str); df['rx_file']=df['rx_file'].astype(str); return df

def group_split(df,seed=2026):
    rng=np.random.default_rng(seed); rows=[]
    for y in sorted(df['mod_label'].unique()):
        d=df[df['mod_label']==y].copy(); groups=np.array(sorted(d['rx_file'].unique())); rng.shuffle(groups); n=len(groups)
        n_test=max(1,int(round(0.20*n))); n_val=max(1,int(round(0.12*n))); n_train=n-n_val-n_test
        train=set(groups[:n_train]); val=set(groups[n_train:n_train+n_val])
        for _,r in d.iterrows():
            rr=r.to_dict(); rr['split']='train' if r['rx_file'] in train else ('val' if r['rx_file'] in val else 'test'); rows.append(rr)
    return pd.DataFrame(rows)

def load_rx_sc(path):
    m=sio.loadmat(path,squeeze_me=True,struct_as_record=False); z=np.asarray(m['rx_sc']); z=np.squeeze(z)
    if not np.iscomplexobj(z): z=z.astype(np.float32).astype(np.complex64)
    return z.astype(np.complex64)

def add_awgn(x,snr_db):
    z=x.astype(np.complex64); mask=np.isfinite(z.real)&np.isfinite(z.imag)
    if mask.sum()==0: return z
    p=np.mean(np.abs(z[mask])**2); sigma2=p/(10.0**(float(snr_db)/10.0)); n=np.sqrt(sigma2/2)*(np.random.randn(*z.shape)+1j*np.random.randn(*z.shape))
    return (z+n.astype(np.complex64)).astype(np.complex64)

def make_cdm(rx_sc,nbin=64,clip_val=3.0):
    z=rx_sc.reshape(-1).astype(np.complex64); z=z[np.isfinite(z.real)&np.isfinite(z.imag)]
    if z.size==0: return np.zeros((1,nbin,nbin),dtype=np.float32)
    z=z-np.mean(z); z=z/np.sqrt(np.mean(np.abs(z)**2)+1e-12); zr=np.clip(z.real,-clip_val,clip_val); zi=np.clip(z.imag,-clip_val,clip_val)
    edges=np.linspace(-clip_val,clip_val,nbin+1); H,_,_=np.histogram2d(zi,zr,bins=[edges,edges]); H=np.log1p(H); H=H/(np.max(H)+1e-12); return H[None,:,:].astype(np.float32)

def skew(x):
    x=np.asarray(x,dtype=np.float64); x=x[np.isfinite(x)]
    return np.nan if x.size<3 else np.mean(((x-np.mean(x))/(np.std(x)+1e-12))**3)

def kurt(x):
    x=np.asarray(x,dtype=np.float64); x=x[np.isfinite(x)]
    return np.nan if x.size<4 else np.mean(((x-np.mean(x))/(np.std(x)+1e-12))**4)

def make_stats(rx_sc):
    z=rx_sc.reshape(-1).astype(np.complex64); z=z[np.isfinite(z.real)&np.isfinite(z.imag)]
    if z.size==0: return np.full((16,),np.nan,dtype=np.float32)
    z=z-np.mean(z); z=z/np.sqrt(np.mean(np.abs(z)**2)+1e-12); a=np.abs(z); i=z.real; q=z.imag; dph=np.diff(np.unwrap(np.angle(z)))
    iq=0.0 if np.std(i)<1e-12 or np.std(q)<1e-12 else float(np.corrcoef(i,q)[0,1])
    return np.array([np.mean(a),np.std(a),skew(a),kurt(a),10*np.log10(np.max(np.abs(z)**2)/(np.mean(np.abs(z)**2)+1e-12)),np.mean(i),np.std(i),skew(i),kurt(i),np.mean(q),np.std(q),skew(q),kurt(q),np.std(dph) if dph.size>0 else np.nan,np.abs(np.mean(np.exp(1j*np.angle(z)))),iq],dtype=np.float32)

def compute_norm(rows):
    X=np.stack([make_stats(load_rx_sc(Path(p))) for p in rows['out_mat'].astype(str)]).astype(np.float32)
    mean=np.nanmean(X,axis=0); std=np.nanstd(X,axis=0); mean=np.nan_to_num(mean,nan=0,posinf=0,neginf=0); std=np.nan_to_num(std,nan=1,posinf=1,neginf=1); std[std<1e-6]=1; return mean.astype(np.float32),std.astype(np.float32)

class DS(Dataset):
    def __init__(self,rows,mean,std,train=False,noise_aug=False,snr_min=10,snr_max=30,phase_aug=True):
        self.rows=rows.reset_index(drop=True); self.mean=mean; self.std=std; self.train=train; self.noise_aug=noise_aug; self.snr_min=snr_min; self.snr_max=snr_max; self.phase_aug=phase_aug
    def __len__(self): return len(self.rows)
    def __getitem__(self,idx):
        r=self.rows.iloc[idx]; z=load_rx_sc(Path(r['out_mat']))
        if self.train and self.phase_aug: z=(z*np.exp(1j*np.random.uniform(-np.pi,np.pi))).astype(np.complex64)
        if self.train and self.noise_aug: z=add_awgn(z,np.random.uniform(self.snr_min,self.snr_max))
        cdm=make_cdm(z); st=make_stats(z); st=np.nan_to_num(st,nan=0,posinf=0,neginf=0); st=(st-self.mean)/self.std; st=np.nan_to_num(st,nan=0,posinf=0,neginf=0).astype(np.float32)
        return torch.from_numpy(cdm),torch.from_numpy(st),torch.tensor(int(r['mod_label']),dtype=torch.long)

class CDMBranch(nn.Module):
    def __init__(self,out=64,drop=0.125):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1,bias=False),nn.BatchNorm2d(16),nn.SiLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.SiLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.SiLU(),nn.MaxPool2d(2),nn.Conv2d(64,96,3,padding=1,bias=False),nn.BatchNorm2d(96),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(96,out),nn.SiLU(),nn.Dropout(drop))
    def forward(self,x): return self.net(x)
class StatsBranch(nn.Module):
    def __init__(self,dim=16,out=32,drop=0.125):
        super().__init__(); self.net=nn.Sequential(nn.Linear(dim,64),nn.BatchNorm1d(64),nn.SiLU(),nn.Dropout(drop),nn.Linear(64,out),nn.SiLU())
    def forward(self,x): return self.net(x)
class Net(nn.Module):
    def __init__(self,drop=0.25):
        super().__init__(); self.cdm=CDMBranch(64,drop*0.5); self.stats=StatsBranch(16,32,drop*0.5); self.cls=nn.Sequential(nn.Linear(96,128),nn.BatchNorm1d(128),nn.SiLU(),nn.Dropout(drop),nn.Linear(128,6))
    def forward(self,cdm,st): return self.cls(torch.cat([self.cdm(cdm),self.stats(st)],1))

def macro_f1(y,p,n=6):
    fs=[]
    for c in range(n):
        tp=np.sum((y==c)&(p==c)); fp=np.sum((y!=c)&(p==c)); fn=np.sum((y==c)&(p!=c)); pr=tp/(tp+fp+1e-12); re=tp/(tp+fn+1e-12); fs.append(2*pr*re/(pr+re+1e-12))
    return float(np.mean(fs))
@torch.no_grad()
def eval_loader(model,loader,device):
    model.eval(); ys=[]; ps=[]
    for cdm,st,y in loader:
        pred=model(cdm.to(device),st.to(device)).argmax(1).cpu().numpy(); ys.extend(y.numpy().tolist()); ps.extend(pred.tolist())
    y=np.asarray(ys); p=np.asarray(ps); return float(np.mean(y==p)), macro_f1(y,p,6)

def train_one(df,turb,args,device):
    out=Path(args.out)/f'{turb}_cdm_stats_noiseaug'; out.mkdir(parents=True,exist_ok=True)
    split=group_split(df[df['turb_name']==turb].copy(),args.seed); split.to_csv(out/'split.csv',index=False,encoding='utf-8-sig')
    tr=split[split['split']=='train']; va=split[split['split']=='val']; te=split[split['split']=='test']; print(f'\n{turb}: {len(tr)}/{len(va)}/{len(te)}'); print(split.groupby(['split','mod_name']).size().unstack(fill_value=0))
    mean,std=compute_norm(tr); loaders={}
    loaders['train']=DataLoader(DS(tr,mean,std,True,True,args.aug_snr_min,args.aug_snr_max,not args.no_phase_aug),batch_size=args.batch_size,shuffle=True,num_workers=0)
    loaders['val']=DataLoader(DS(va,mean,std),batch_size=args.batch_size,shuffle=False,num_workers=0)
    loaders['test']=DataLoader(DS(te,mean,std),batch_size=args.batch_size,shuffle=False,num_workers=0)
    model=Net(args.dropout).to(device); crit=nn.CrossEntropyLoss(label_smoothing=args.label_smoothing); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.lr*0.05); scaler=torch.cuda.amp.GradScaler(enabled=(device.type=='cuda' and not args.no_amp))
    best=-1; bad=0; hist=[]; t0=time.time()
    for ep in range(1,args.epochs+1):
        model.train(); loss_sum=0; n=0
        for cdm,st,y in loaders['train']:
            cdm=cdm.to(device); st=st.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda' and not args.no_amp)):
                loss=crit(model(cdm,st),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); loss_sum+=float(loss.detach().cpu())*y.size(0); n+=y.size(0)
        sch.step(); tracc,_=eval_loader(model,loaders['train'],device); vacc,vf1=eval_loader(model,loaders['val'],device); hist.append({'epoch':ep,'loss':loss_sum/max(1,n),'train_acc':tracc,'val_acc':vacc})
        print(f'{turb} ep{ep:03d} loss={loss_sum/max(1,n):.4f} train={tracc:.4f} val={vacc:.4f}')
        if vacc>best:
            best=vacc; bad=0; torch.save({'model_state':model.state_dict(),'model_mode':'cdm_stats','turb_name':turb,'mod_names':MOD_NAMES,'stats_mean':mean,'stats_std':std,'stats_dim':16,'best_epoch':ep,'best_val_acc':best,'args':vars(args)},out/'best_model.pt')
        else:
            bad+=1
        if bad>=args.patience: print('early stop'); break
    ckpt=torch.load(out/'best_model.pt',map_location=device,weights_only=False); model.load_state_dict(ckpt['model_state']); teacc,tef1=eval_loader(model,loaders['test'],device)
    res={'turb_name':turb,'test_acc':teacc,'test_f1':tef1,'best_val_acc':float(best),'train_seconds':time.time()-t0,'history':hist,'out_dir':str(out)}
    with open(out/'metrics.json','w',encoding='utf-8') as f: json.dump(res,f,ensure_ascii=False,indent=2)
    if MPL_OK:
        plt.figure(figsize=(7,4)); plt.plot([h['epoch'] for h in hist],[h['train_acc'] for h in hist],label='train'); plt.plot([h['epoch'] for h in hist],[h['val_acc'] for h in hist],label='val'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(out/'training_curve.png',dpi=300); plt.close()
    print(f'{turb} test acc={teacc:.4f} f1={tef1:.4f}'); return res

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--out',required=True); ap.add_argument('--turbs',nargs='*',default=['weak','strong']); ap.add_argument('--epochs',type=int,default=60); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--patience',type=int,default=10); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--dropout',type=float,default=0.25); ap.add_argument('--label-smoothing',type=float,default=0.03); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--aug-snr-min',type=float,default=10); ap.add_argument('--aug-snr-max',type=float,default=30); ap.add_argument('--no-phase-aug',action='store_true'); ap.add_argument('--no-amp',action='store_true'); args=ap.parse_args()
    set_seed(args.seed); Path(args.out).mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    df=load_manifest(args.data_root); print(df.groupby(['turb_name','mod_name']).size().unstack(fill_value=0))
    results=[train_one(df,t,args,device) for t in args.turbs]
    pd.DataFrame([{'turb_name':r['turb_name'],'test_acc':r['test_acc'],'test_f1':r['test_f1'],'best_val_acc':r['best_val_acc'],'out_dir':r['out_dir']} for r in results]).to_csv(Path(args.out)/'summary_noiseaug_cdm_stats.csv',index=False,encoding='utf-8-sig')
if __name__=='__main__': main()
