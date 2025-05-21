#!/bin/bash

# spectroresnet x ppgbp→bcg (uh341l1h)
CUDA_VISIBLE_DEVICES=0 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_ppgbp.yaml --epochs 10 --lr 1e-1 --method original --shots 10 --target bcg --transfer ppgbp --baseline ft --wd 1e-1 &

# spectroresnet x sensors→bcg (7broj02m)
CUDA_VISIBLE_DEVICES=1 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_sensors.yaml --epochs 10 --lr 1e-1 --method original --shots 10 --target bcg --transfer sensors --baseline ft --wd 1e-1 &

# spectroresnet x bcg→ppgbp (q2mwko1y)
CUDA_VISIBLE_DEVICES=2 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_bcg.yaml --epochs 10 --lr 1e-1 --method original --shots 10 --target ppgbp --transfer bcg --baseline ft --wd 1e-2 &

# spectroresnet x sensors→ppgbp (yt4j7o8e)
CUDA_VISIBLE_DEVICES=3 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_sensors.yaml --epochs 10 --lr 1e-1 --method original --shots 10 --target ppgbp --transfer sensors --baseline ft --wd 1e-2 &

# spectroresnet x bcg→sensors (lox73avg)
CUDA_VISIBLE_DEVICES=4 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_bcg.yaml --epochs 10 --lr 1e-1 --method original --shots 10 --target sensors --transfer bcg --baseline ft --wd 1e-2 &

# spectroresnet x ppgbp→sensors (biww7zzn)
CUDA_VISIBLE_DEVICES=5 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_ppgbp.yaml --epochs 10 --lr 1e-1 --method original --shots 10 --target sensors --transfer ppgbp --baseline ft --wd 1e-2 &

# spectroresnet x vital_ecg→mimic_ecg (anw3h9nn)
CUDA_VISIBLE_DEVICES=6 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_vital_ecg.yaml --epochs 10 --lr 1e-2 --method original --shots 10 --target mimic_ecg --transfer vital_ecg --baseline ft --wd 1e-1 &

# spectroresnet x mimic_ecg→vital_ecg (dn8s1zzm)
CUDA_VISIBLE_DEVICES=7 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_mimic_ecg.yaml --epochs 10 --lr 1e-1 --method original --shots 10 --target vital_ecg --transfer mimic_ecg --baseline ft --wd 1e-2 &

# mlpbp x ppgbp→bcg (s9029ynj)
CUDA_VISIBLE_DEVICES=0 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_ppgbp.yaml --epochs 10 --lr 1e-4 --method original --shots 10 --target bcg --transfer ppgbp --baseline ft --wd 1e-2 &

# mlpbp x sensors→bcg (i3q25se7)
CUDA_VISIBLE_DEVICES=1 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_sensors.yaml --epochs 10 --lr 1e-3 --method original --shots 10 --target bcg --transfer sensors --baseline ft --wd 1e-3 &

# mlpbp x bcg→ppgbp (7z9vdwat)
CUDA_VISIBLE_DEVICES=2 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_bcg.yaml --epochs 10 --lr 1e-4 --method original --shots 10 --target ppgbp --transfer bcg --baseline ft --wd 1e-2 &

# mlpbp x sensors→ppgbp (ro6rnbej)
CUDA_VISIBLE_DEVICES=3 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_sensors.yaml --epochs 10 --lr 1e-4 --method original --shots 10 --target ppgbp --transfer sensors --baseline ft --wd 1e-2 &

# mlpbp x bcg→sensors (3a3gf3sq)
CUDA_VISIBLE_DEVICES=4 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_bcg.yaml --epochs 10 --lr 1e-3 --method original --shots 10 --target sensors --transfer bcg --baseline ft --wd 1e-2 &

# mlpbp x ppgbp→sensors (zjy492r0)
CUDA_VISIBLE_DEVICES=5 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_ppgbp.yaml --epochs 10 --lr 1e-3 --method original --shots 10 --target sensors --transfer ppgbp --baseline ft --wd 1e-3 &

# mlpbp x vital_ecg→mimic_ecg (9qr1f5s0)
CUDA_VISIBLE_DEVICES=6 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_vital_ecg.yaml --epochs 10 --lr 1e-3 --method original --shots 10 --target mimic_ecg --transfer vital_ecg --baseline ft --wd 1e-1 &

# mlpbp x mimic_ecg→vital_ecg (uikg5gix)
CUDA_VISIBLE_DEVICES=7 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_mimic_ecg.yaml --epochs 10 --lr 1e-3 --method original --shots 10 --target vital_ecg --transfer mimic_ecg --baseline ft --wd 1e-2 &

# resnet1d x ppgbp→bcg (1awngvdp)
CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_ppgbp.yaml --epochs 10 --lr 1e-2 --method original --shots 10 --target bcg --transfer ppgbp --baseline ft --wd 1e-3 &

# resnet1d x sensors→bcg (hsg2g5ct)
CUDA_VISIBLE_DEVICES=1 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_sensors.yaml --epochs 10 --lr 1e-2 --method original --shots 10 --target bcg --transfer sensors --baseline ft --wd 1e-3 &

# resnet1d x bcg→ppgbp (5zhh7mw4)
CUDA_VISIBLE_DEVICES=2 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_bcg.yaml --epochs 10 --lr 1e-3 --method original --shots 10 --target ppgbp --transfer bcg --baseline ft --wd 1e-2 &

# resnet1d x sensors→ppgbp (8ipahonn)
CUDA_VISIBLE_DEVICES=3 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_sensors.yaml --epochs 10 --lr 1e-3 --method original --shots 10 --target ppgbp --transfer sensors --baseline ft --wd 1e-3 &

# resnet1d x bcg→sensors (jr9qwd1x)
CUDA_VISIBLE_DEVICES=4 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_bcg.yaml --epochs 10 --lr 1e-2 --method original --shots 10 --target sensors --transfer bcg --baseline ft --wd 1e-1 &

# resnet1d x ppgbp→sensors (z0xysnx6)
CUDA_VISIBLE_DEVICES=5 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_ppgbp.yaml --epochs 10 --lr 1e-2 --method original --shots 10 --target sensors --transfer ppgbp --baseline ft --wd 1e-3 &

# resnet1d x vital_ecg→mimic_ecg (97teik54)
CUDA_VISIBLE_DEVICES=6 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_vital_ecg.yaml --epochs 10 --lr 1e-3 --method original --shots 10 --target mimic_ecg --transfer vital_ecg --baseline ft --wd 1e-1 &

# resnet1d x mimic_ecg→vital_ecg (zd1qljvn)
CUDA_VISIBLE_DEVICES=7 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_mimic_ecg.yaml --epochs 10 --lr 1e-2 --method original --shots 10 --target vital_ecg --transfer mimic_ecg --baseline ft --wd 1e-1 &

# bptransformer x ppgbp→bcg (vcweklkc)
CUDA_VISIBLE_DEVICES=0 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_ppgbp.yaml --epochs 10 --lr 1e-4 --method original --shots 10 --target bcg --transfer ppgbp --baseline ft --wd 1e-1 &

# bptransformer x sensors→bcg (nwru8yyt)
CUDA_VISIBLE_DEVICES=1 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_sensors.yaml --epochs 10 --lr 1e-3 --method original --shots 10 --target bcg --transfer sensors --baseline ft --wd 1e-2 &

# bptransformer x bcg→ppgbp (3nk94bt4)
CUDA_VISIBLE_DEVICES=2 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_bcg.yaml --epochs 10 --lr 1e-4 --method original --shots 10 --target ppgbp --transfer bcg --baseline ft --wd 1e-3 &

# bptransformer x sensors→ppgbp (dof2a8yt)
CUDA_VISIBLE_DEVICES=3 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_sensors.yaml --epochs 10 --lr 1e-2 --method original --shots 10 --target ppgbp --transfer sensors --baseline ft --wd 1e-2 &

# bptransformer x bcg→sensors (e2cted5w)
CUDA_VISIBLE_DEVICES=4 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_bcg.yaml --epochs 10 --lr 1e-3 --method original --shots 10 --target sensors --transfer bcg --baseline ft --wd 1e-2 &

# bptransformer x ppgbp→sensors (zzdrlk1g)
CUDA_VISIBLE_DEVICES=5 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_ppgbp.yaml --epochs 10 --lr 1e-2 --method original --shots 10 --target sensors --transfer ppgbp --baseline ft --wd 1e-1 &

# bptransformer x vital_ecg→mimic_ecg (z2p61equ)
CUDA_VISIBLE_DEVICES=6 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_vital_ecg.yaml --epochs 10 --lr 1e-4 --method original --shots 10 --target mimic_ecg --transfer vital_ecg --baseline ft --wd 1e-1 &

# bptransformer x mimic_ecg→vital_ecg (r0rbqns8)
CUDA_VISIBLE_DEVICES=7 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_mimic_ecg.yaml --epochs 10 --lr 1e-4 --method original --shots 10 --target vital_ecg --transfer mimic_ecg --baseline ft --wd 1e-1 &

# bpt+ours x ppgbp→bcg (81htbmjw)
CUDA_VISIBLE_DEVICES=0 python train.py --config_file core/config/dl/bptransformer/bptransformer_ppgbp.yaml --method prompt_global --backbone bptransformer --shots 10 --transfer ppgbp --target bcg --query_dim 64 --lr 1e-3 --batch_size 6 --wd 1e-2 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 32 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 25 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x sensors→bcg (a0vo0uxe)
CUDA_VISIBLE_DEVICES=1 python train.py --config_file core/config/dl/bptransformer/bptransformer_sensors.yaml --method prompt_global --backbone bptransformer --shots 10 --transfer sensors --target bcg --query_dim 64 --lr 1e-3 --batch_size 6 --wd 1e-3 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 64 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 25 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x bcg→ppgbp (scvtj2u0)
CUDA_VISIBLE_DEVICES=2 python train.py --config_file core/config/dl/bptransformer/bptransformer_bcg.yaml --method prompt_global --backbone bptransformer --shots 10 --transfer bcg --target ppgbp --query_dim 64 --lr 1e-4 --batch_size 6 --wd 1e-3 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 64 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 50 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x sensors→ppgbp (gpozew7v)
CUDA_VISIBLE_DEVICES=3 python train.py --config_file core/config/dl/bptransformer/bptransformer_sensors.yaml --method prompt_global --backbone bptransformer --shots 10 --transfer sensors --target ppgbp --query_dim 64 --lr 1e-2 --batch_size 6 --wd 1e-3 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 64 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 25 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x bcg→sensors (7t1iz0go)
CUDA_VISIBLE_DEVICES=4 python train.py --config_file core/config/dl/bptransformer/bptransformer_bcg.yaml --method prompt_global --backbone bptransformer --shots 10 --transfer bcg --target sensors --query_dim 64 --lr 1e-2 --batch_size 6 --wd 1e-2 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 32 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 50 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x ppgbp→sensors (c02uvlty)
CUDA_VISIBLE_DEVICES=5 python train.py --config_file core/config/dl/bptransformer/bptransformer_ppgbp.yaml --method prompt_global --backbone bptransformer --shots 10 --transfer ppgbp --target sensors --query_dim 64 --lr 1e-1 --batch_size 6 --wd 1e-1 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 64 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 25 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x vital_ecg→mimic_ecg (itnarvmf)
CUDA_VISIBLE_DEVICES=6 python train.py --config_file core/config/dl/bptransformer/bptransformer_vital_ecg.yaml --method prompt_global --backbone bptransformer --shots 10 --transfer vital_ecg --target mimic_ecg --query_dim 64 --lr 1e-3 --batch_size 6 --wd 1e-3 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 32 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 25 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x mimic_ecg→vital_ecg (vcdgqu5k)
CUDA_VISIBLE_DEVICES=7 python train.py --config_file core/config/dl/bptransformer/bptransformer_mimic_ecg.yaml --method prompt_global --backbone bptransformer --shots 10 --transfer mimic_ecg --target vital_ecg --query_dim 64 --lr 1e-1 --batch_size 6 --wd 1e-1 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 32 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 50 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

