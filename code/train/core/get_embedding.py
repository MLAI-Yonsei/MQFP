# core/get_embedding.py
# run command !!!! [ TODO ] bp_L2P/code/train$ PYTHONPATH=. python core/get_embedding.py [ TODO ]
# -----------------------------------------------------
# 0. 기본 import & 경로 설정
# -----------------------------------------------------
import os, sys, joblib, torch, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from scipy.io import loadmat
from tqdm.auto import tqdm

# 프로젝트 루트(code/train)를 PYTHONPATH에 포함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils  import remove_outlier, mat2df, group_annot
from models import BPTransformerRegressor

# -----------------------------------------------------
# 1. 하이퍼파라미터 / 경로
# -----------------------------------------------------
device        = "cuda:0"
root_ckpt     = "/data1/bubble3jh/bp_L2P/code/train/pretrained_models"
ds_tag        = "sensors"
backbone      = "bptransformer"
cfg           = OmegaConf.load(f"core/config/dl/{backbone}/{backbone}_{ds_tag}.yaml")
ckpt_tpl      = f"{root_ckpt}/{ds_tag}-{backbone}/fold{{}}.ckpt"

seq_len       = 256                # 시계열 길이
batch_size    = 1024        # ckpt의 test 배치 설정 그대로 사용

# -----------------------------------------------------
# 2. Custom Dataset  (signal → PPG, abp_signal → ABP)
# -----------------------------------------------------
class BPDataset(Dataset):
    def __init__(self, df, seq_len=256):
        self.df      = df.reset_index(drop=True)
        self.seq_len = seq_len

    def __len__(self): return len(self.df)

    def _fix_len(self, sig):
        sig = np.asarray(sig, dtype=np.float32)
        if len(sig) >= self.seq_len:
            sig = sig[:self.seq_len]
        else:
            sig = np.pad(sig, (0, self.seq_len-len(sig)))
        return torch.from_numpy(sig)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        x_ppg     = self._fix_len(row["signal"])        # (L,)
        y         = torch.tensor([row["SP"], row["DP"]], dtype=torch.float32)
        group     = torch.tensor(row["group"], dtype=torch.long)
        peakmask  = torch.zeros(self.seq_len, dtype=torch.bool)
        vlymask   = torch.zeros(self.seq_len, dtype=torch.bool)

        # BPTransformerRegressor 는 dict 형태의 x_ppg 를 expectation
        return (
            {"ppg": x_ppg.unsqueeze(0)},  # (1, L)
            y, group, None,
            peakmask, vlymask
        )

def bp_collate(batch):
    x_ppg, y, group, _, peakmask, vlymask = zip(*batch)
    x_ppg   = torch.stack([b["ppg"] for b in x_ppg])   # (B,1,L)
    y       = torch.stack(y)
    group   = torch.stack(group)
    peakmask= torch.stack(peakmask)
    vlymask = torch.stack(vlymask)
    return x_ppg, y, group, None, peakmask, vlymask

# -----------------------------------------------------
# 3. 데이터 전체 로드 & 전처리
# -----------------------------------------------------
if cfg.exp.subject_dict.endswith(".pkl"):
    all_split_df = joblib.load(cfg.exp.subject_dict)          # 단일 DF
else:                                                         # *_i.mat 리스트
    all_split_df = [mat2df(loadmat(f"{cfg.exp.subject_dict}_{i}.mat"))
                    for i in range(cfg.exp.N_fold)]

all_split_df = remove_outlier(all_split_df)
all_split_df = group_annot(all_split_df)
full_df      = pd.concat(all_split_df) if isinstance(all_split_df, list) else all_split_df

# -----------------------------------------------------
# 4. 고정 DataLoader (전체 데이터)
# -----------------------------------------------------
full_ds = BPDataset(full_df, seq_len=seq_len)
full_dl = DataLoader(full_ds,
                     batch_size=batch_size,
                     shuffle=False,
                     num_workers=0,
                     pin_memory=True,
                     collate_fn=bp_collate)

# -----------------------------------------------------
# 5. fold 별 모델 → 전체 데이터 임베딩
# -----------------------------------------------------
for fold_idx in range(cfg.exp.N_fold):
    ckpt_path = ckpt_tpl.format(fold_idx)
    if not os.path.exists(ckpt_path):
        print(f"[Fold {fold_idx}] ckpt 없음 → skip"); continue

    model = BPTransformerRegressor.load_from_checkpoint(f"/data1/bubble3jh/bp_L2P/code/train/pretrained_models/{ds_tag}-{backbone}/fold{fold_idx}.ckpt").to(device).eval()

    emb_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(full_dl, desc=f"Fold {fold_idx}"):
            x_ppg, y, *_ = batch
            emb = model.extract_penultimate_embedding(x_ppg.to(device))  # (B,D)
            emb_list.append(emb.cpu())
            label_list.append(y)

    fold_emb = torch.cat(emb_list)      # (N,D)
    fold_lab = torch.cat(label_list)
    out_f    = f"/data1/bubble3jh/bp_L2P/code/train/embeds_mean/bptransformer/{ds_tag}_fold{fold_idx}_ALL.pt"
    torch.save({"embeddings": fold_emb, "labels": fold_lab}, out_f)
    print(f"[Fold {fold_idx}] saved → {out_f} {tuple(fold_emb.shape)}")
