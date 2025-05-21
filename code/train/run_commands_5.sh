#!/bin/bash

# spectroresnet x ppgbp→bcg (uh5rxiuv)
CUDA_VISIBLE_DEVICES=0 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_ppgbp.yaml --epochs 10 --lr 1e-2 --method original --shots 5 --target bcg --transfer ppgbp --baseline ft --wd 1e-1 &

# spectroresnet x sensors→bcg (gg6u1pk8)
CUDA_VISIBLE_DEVICES=1 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_sensors.yaml --epochs 10 --lr 1e-2 --method original --shots 5 --target bcg --transfer sensors --baseline ft --wd 1e-1 &

# spectroresnet x bcg→ppgbp (naucf5pn)
CUDA_VISIBLE_DEVICES=2 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_bcg.yaml --epochs 10 --lr 1e-1 --method original --shots 5 --target ppgbp --transfer bcg --baseline ft --wd 1e-2 &

# spectroresnet x sensors→ppgbp (ogrg338j)
CUDA_VISIBLE_DEVICES=3 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_sensors.yaml --epochs 10 --lr 1e-1 --method original --shots 5 --target ppgbp --transfer sensors --baseline ft --wd 1e-2 &

# spectroresnet x bcg→sensors (aicdguab)
CUDA_VISIBLE_DEVICES=4 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_bcg.yaml --epochs 10 --lr 1e-1 --method original --shots 5 --target sensors --transfer bcg --baseline ft --wd 1e-1 &

# spectroresnet x ppgbp→sensors (vpgzo23v)
CUDA_VISIBLE_DEVICES=5 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_ppgbp.yaml --epochs 10 --lr 1e-1 --method original --shots 5 --target sensors --transfer ppgbp --baseline ft --wd 1e-1 &

# spectroresnet x vital_ecg→mimic_ecg (nhfee6d6)
CUDA_VISIBLE_DEVICES=6 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_vital_ecg.yaml --epochs 10 --lr 1e-2 --method original --shots 5 --target mimic_ecg --transfer vital_ecg --baseline ft --wd 1e-1 &

# spectroresnet x mimic_ecg→vital_ecg (6njr55tx)
CUDA_VISIBLE_DEVICES=7 python train.py --backbone spectroresnet --batch_size 6 --config_file core/config/dl/spectroresnet/spectroresnet_mimic_ecg.yaml --epochs 10 --lr 1e-1 --method original --shots 5 --target vital_ecg --transfer mimic_ecg --baseline ft --wd 1e-1 &

# mlpbp x ppgbp→bcg (zxywqunl)
CUDA_VISIBLE_DEVICES=0 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_ppgbp.yaml --epochs 10 --lr 1e-3 --method original --shots 5 --target bcg --transfer ppgbp --baseline ft --wd 1e-3 &

# mlpbp x sensors→bcg (jq47be1o)
CUDA_VISIBLE_DEVICES=1 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_sensors.yaml --epochs 10 --lr 1e-3 --method original --shots 5 --target bcg --transfer sensors --baseline ft --wd 1e-1 &

# mlpbp x bcg→ppgbp (5gn9yuld)
CUDA_VISIBLE_DEVICES=2 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_bcg.yaml --epochs 10 --lr 1e-4 --method original --shots 5 --target ppgbp --transfer bcg --baseline ft --wd 1e-3 &

# mlpbp x sensors→ppgbp (1uc5mm4b)
CUDA_VISIBLE_DEVICES=3 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_sensors.yaml --epochs 10 --lr 1e-4 --method original --shots 5 --target ppgbp --transfer sensors --baseline ft --wd 1e-2 &

# mlpbp x bcg→sensors (8mafbd9u)
CUDA_VISIBLE_DEVICES=4 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_bcg.yaml --epochs 10 --lr 1e-3 --method original --shots 5 --target sensors --transfer bcg --baseline ft --wd 1e-3 &

# mlpbp x ppgbp→sensors (99046yae)
CUDA_VISIBLE_DEVICES=5 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_ppgbp.yaml --epochs 10 --lr 1e-3 --method original --shots 5 --target sensors --transfer ppgbp --baseline ft --wd 1e-3 &

# mlpbp x vital_ecg→mimic_ecg (jkte6o9e)
CUDA_VISIBLE_DEVICES=6 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_vital_ecg.yaml --epochs 10 --lr 1e-4 --method original --shots 5 --target mimic_ecg --transfer vital_ecg --baseline ft --wd 1e-1 &

# mlpbp x mimic_ecg→vital_ecg (i37o0d2e)
CUDA_VISIBLE_DEVICES=7 python train.py --backbone mlpbp --batch_size 6 --config_file core/config/dl/mlpbp/mlpbp_mimic_ecg.yaml --epochs 10 --lr 1e-4 --method original --shots 5 --target vital_ecg --transfer mimic_ecg --baseline ft --wd 1e-3 &

# resnet1d x ppgbp→bcg (z8ar45wh)
CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_ppgbp.yaml --epochs 10 --lr 1e-3 --method original --shots 5 --target bcg --transfer ppgbp --baseline ft --wd 1e-1 &

# resnet1d x sensors→bcg (uys13zz6)
CUDA_VISIBLE_DEVICES=1 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_sensors.yaml --epochs 10 --lr 1e-2 --method original --shots 5 --target bcg --transfer sensors --baseline ft --wd 1e-2 &

# resnet1d x bcg→ppgbp (721ddut6)
CUDA_VISIBLE_DEVICES=2 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_bcg.yaml --epochs 10 --lr 1e-2 --method original --shots 5 --target ppgbp --transfer bcg --baseline ft --wd 1e-1 &

# resnet1d x sensors→ppgbp (puwiocm4)
CUDA_VISIBLE_DEVICES=3 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_sensors.yaml --epochs 10 --lr 1e-2 --method original --shots 5 --target ppgbp --transfer sensors --baseline ft --wd 1e-1 &

# resnet1d x bcg→sensors (2d1vpo59)
CUDA_VISIBLE_DEVICES=4 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_bcg.yaml --epochs 10 --lr 1e-2 --method original --shots 5 --target sensors --transfer bcg --baseline ft --wd 1e-1 &

# resnet1d x ppgbp→sensors (29ji834k)
CUDA_VISIBLE_DEVICES=5 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_ppgbp.yaml --epochs 10 --lr 1e-2 --method original --shots 5 --target sensors --transfer ppgbp --baseline ft --wd 1e-1 &

# resnet1d x vital_ecg→mimic_ecg (4wxuaafs)
CUDA_VISIBLE_DEVICES=6 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_vital_ecg.yaml --epochs 10 --lr 1e-2 --method original --shots 5 --target mimic_ecg --transfer vital_ecg --baseline ft --wd 1e-1 &

# resnet1d x mimic_ecg→vital_ecg (imlxavbh)
CUDA_VISIBLE_DEVICES=7 python train.py --backbone resnet1d --batch_size 6 --config_file core/config/dl/resnet/resnet_mimic_ecg.yaml --epochs 10 --lr 1e-2 --method original --shots 5 --target vital_ecg --transfer mimic_ecg --baseline ft --wd 1e-1 &

# bptransformer x ppgbp→bcg (bnrfrmcs)
CUDA_VISIBLE_DEVICES=0 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_ppgbp.yaml --epochs 10 --lr 1e-4 --method original --shots 5 --target bcg --transfer ppgbp --baseline ft --wd 1e-1 &

# bptransformer x sensors→bcg (3s47lhwz)
CUDA_VISIBLE_DEVICES=1 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_sensors.yaml --epochs 10 --lr 1e-3 --method original --shots 5 --target bcg --transfer sensors --baseline ft --wd 1e-3 &

# bptransformer x bcg→ppgbp (hq0t8zav)
CUDA_VISIBLE_DEVICES=2 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_bcg.yaml --epochs 10 --lr 1e-4 --method original --shots 5 --target ppgbp --transfer bcg --baseline ft --wd 1e-2 &

# bptransformer x sensors→ppgbp (p2d6yvs6)
CUDA_VISIBLE_DEVICES=3 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_sensors.yaml --epochs 10 --lr 1e-2 --method original --shots 5 --target ppgbp --transfer sensors --baseline ft --wd 1e-2 &

# bptransformer x bcg→sensors (iqrlv0vp)
CUDA_VISIBLE_DEVICES=4 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_bcg.yaml --epochs 10 --lr 1e-3 --method original --shots 5 --target sensors --transfer bcg --baseline ft --wd 1e-3 &

# bptransformer x ppgbp→sensors (g3r4f5g6)
CUDA_VISIBLE_DEVICES=5 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_ppgbp.yaml --epochs 10 --lr 1e-3 --method original --shots 5 --target sensors --transfer ppgbp --baseline ft --wd 1e-3 &

# bptransformer x vital_ecg→mimic_ecg (r7ozw7u7)
CUDA_VISIBLE_DEVICES=6 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_vital_ecg.yaml --epochs 10 --lr 1e-4 --method original --shots 5 --target mimic_ecg --transfer vital_ecg --baseline ft --wd 1e-2 &

# bptransformer x mimic_ecg→vital_ecg (4qtab1ve)
CUDA_VISIBLE_DEVICES=7 python train.py --backbone bptransformer --batch_size 6 --config_file core/config/dl/bptransformer/bptransformer_mimic_ecg.yaml --epochs 10 --lr 1e-4 --method original --shots 5 --target vital_ecg --transfer mimic_ecg --baseline ft --wd 1e-1 &

# bpt+ours x ppgbp→bcg (7m1ltxpa)
CUDA_VISIBLE_DEVICES=0 python train.py --config_file core/config/dl/bptransformer/bptransformer_ppgbp.yaml --method prompt_global --backbone bptransformer --shots 5 --transfer ppgbp --target bcg --query_dim 64 --lr 1e-3 --batch_size 6 --wd 1e-3 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 64 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 25 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x sensors→bcg (gxgveqrp)
CUDA_VISIBLE_DEVICES=1 python train.py --config_file core/config/dl/bptransformer/bptransformer_sensors.yaml --method prompt_global --backbone bptransformer --shots 5 --transfer sensors --target bcg --query_dim 64 --lr 1e-2 --batch_size 6 --wd 1e-3 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 32 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 50 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x bcg→ppgbp (e27vgfil)
CUDA_VISIBLE_DEVICES=2 python train.py --config_file core/config/dl/bptransformer/bptransformer_bcg.yaml --method prompt_global --backbone bptransformer --shots 5 --transfer bcg --target ppgbp --query_dim 64 --lr 1e-1 --batch_size 6 --wd 1e-1 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 64 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 50 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca &

# bpt+ours x sensors→ppgbp (152zjw6r)
CUDA_VISIBLE_DEVICES=3 python train.py --config_file core/config/dl/bptransformer/bptransformer_sensors.yaml --method prompt_global --backbone bptransformer --shots 5 --transfer sensors --target ppgbp --query_dim 64 --lr 1e-2 --batch_size 6 --wd 1e-3 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 64 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 25 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x bcg→sensors (9ptlrtzq)
CUDA_VISIBLE_DEVICES=4 python train.py --config_file core/config/dl/bptransformer/bptransformer_bcg.yaml --method prompt_global --backbone bptransformer --shots 5 --transfer bcg --target sensors --query_dim 64 --lr 1e-3 --batch_size 6 --wd 1e-3 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 32 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 25 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x ppgbp→sensors (88lvqagu)
CUDA_VISIBLE_DEVICES=5 python train.py --config_file core/config/dl/bptransformer/bptransformer_ppgbp.yaml --method prompt_global --backbone bptransformer --shots 5 --transfer ppgbp --target sensors --query_dim 64 --lr 1e-1 --batch_size 6 --wd 1e-3 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 64 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 25 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x vital_ecg→mimic_ecg (1hc9fqo7)
CUDA_VISIBLE_DEVICES=6 python train.py --config_file core/config/dl/bptransformer/bptransformer_vital_ecg.yaml --method prompt_global --backbone bptransformer --shots 5 --transfer vital_ecg --target mimic_ecg --query_dim 64 --lr 1e-1 --batch_size 6 --wd 1e-1 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 16 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 25 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca --train_head &

# bpt+ours x mimic_ecg→vital_ecg (23kphaty)
CUDA_VISIBLE_DEVICES=7 python train.py --config_file core/config/dl/bptransformer/bptransformer_mimic_ecg.yaml --method prompt_global --backbone bptransformer --shots 5 --transfer mimic_ecg --target vital_ecg --query_dim 64 --lr 1e-1 --batch_size 6 --wd 1e-1 --num_pool 4 --global_coeff 1 --qk_sim_coeff 0 --pca_dim 16 --lam 1 --prompt_weights learnable --penalty_scaler 0 --trunc_dim 25 --diff_loss_weight 1.0 --stepbystep --use_emb_diff --train_imag --add_freq --pass_pca &

