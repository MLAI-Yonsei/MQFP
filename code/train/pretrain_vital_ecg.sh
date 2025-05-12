# !/bin/bash

# Model: spectroresnet
# CUDA_VISIBLE_DEVICES=2 python train.py --target=vital_ecg --batch_size=6 --config_file=core/config/dl/spectroresnet/spectroresnet_vital_ecg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone spectroresnet &

# # Model: mlpbp
# CUDA_VISIBLE_DEVICES=3 python train.py --target=vital_ecg --batch_size=6 --config_file=core/config/dl/mlpbp/mlpbp_vital_ecg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone mlpbp &

# # Model: resnet1d
# CUDA_VISIBLE_DEVICES=4 python train.py --target=vital_ecg --batch_size=6 --config_file=core/config/dl/resnet/resnet_vital_ecg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone resnet1d &

# Model: BPTransformer
CUDA_VISIBLE_DEVICES=5 python train.py --target=vital_ecg --batch_size=6 --config_file=core/config/dl/bptransformer/bptransformer_vital_ecg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone bptransformer &

wait


echo "Done"