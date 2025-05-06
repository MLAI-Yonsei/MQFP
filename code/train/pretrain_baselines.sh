# for BCG
python train.py --target=bcg --batch_size=6 --config_file=core/config/dl/spectroresnet/spectroresnet_bcg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone spectroresnet

# for PPGBP
python train.py --target=ppgbp --batch_size=6 --config_file=core/config/dl/spectroresnet/spectroresnet_ppgbp.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone spectroresnet

# for Sensors
python train.py --target=sensors --batch_size=6 --config_file=core/config/dl/spectroresnet/spectroresnet_sensors.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone spectroresnet

# for BCG
python train.py --target=bcg --batch_size=6 --config_file=core/config/dl/mlpbp/mlpbp_bcg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone mlpbp

# for PPGBP
python train.py --target=ppgbp --batch_size=6 --config_file=core/config/dl/mlpbp/mlpbp_ppgbp.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone mlpbp

# for Sensors
python train.py --target=sensors --batch_size=6 --config_file=core/config/dl/mlpbp/mlpbp_sensors.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone mlpbp