# Model: spectroresnet
# # for BCG
# python train.py --target=bcg --batch_size=6 --config_file=core/config/dl/spectroresnet/spectroresnet_bcg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone spectroresnet

# # for PPGBP
# python train.py --target=ppgbp --batch_size=6 --config_file=core/config/dl/spectroresnet/spectroresnet_ppgbp.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone spectroresnet

# for MIMIC ECG
CUDA_VISIBLE_DEVICES=0 python train.py --target=mimic_ecg --batch_size=6 --config_file=core/config/dl/spectroresnet/spectroresnet_mimic_ecg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone spectroresnet

# # for Sensors
# python train.py --target=sensors --batch_size=6 --config_file=core/config/dl/spectroresnet/spectroresnet_sensors.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone spectroresnet

# Model: mlpbp
# # for BCG
# python train.py --target=bcg --batch_size=6 --config_file=core/config/dl/mlpbp/mlpbp_bcg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone mlpbp

# # for PPGBP
# python train.py --target=ppgbp --batch_size=6 --config_file=core/config/dl/mlpbp/mlpbp_ppgbp.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone mlpbp

# # for Sensors
# python train.py --target=sensors --batch_size=6 --config_file=core/config/dl/mlpbp/mlpbp_sensors.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone mlpbp

# for MIMIC ECG
CUDA_VISIBLE_DEVICES=1 python train.py --target=mimic_ecg --batch_size=6 --config_file=core/config/dl/mlpbp/mlpbp_mimic_ecg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone mlpbp

# Model: resnet1d
# for MIMIC ECG
CUDA_VISIBLE_DEVICES=2 python train.py --target=mimic_ecg --batch_size=6 --config_file=core/config/dl/resnet/resnet_mimic_ecg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone resnet1d

# Model: BPTransformer
# for MIMIC ECG
CUDA_VISIBLE_DEVICES=3 python train.py --target=mimic_ecg --batch_size=6 --config_file=core/config/dl/bptransformer/bptransformer_mimic_ecg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone bptransformer