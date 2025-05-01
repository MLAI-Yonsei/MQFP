# for BCG
python train.py --target=bcg --batch_size=6 --config_file=core/config/dl/bptransformer/bptransformer_bcg.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone bptransformer

# for PPGBP
python train.py --target=ppgbp --batch_size=6 --config_file=core/config/dl/bptransformer/bptransformer_ppgbp.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone bptransformer

# for Sensors
python train.py --target=sensors --batch_size=6 --config_file=core/config/dl/bptransformer/bptransformer_sensors.yaml --epochs=100 --method=original --shots=0 --ignore_wandb --save_for_pretraining --backbone bptransformer