#!/bin/bash
cd ..

LR_RANGE=(1e-1 1e-2 1e-3 1e-4)
WD_RANGE=(1e-1 1e-2 1e-3)
BACKBONES=("spectroresnet" "mlpbp" "bptransformer")
BASELINES=("lp" "ft")
TARGETS=("bcg" "ppgbp" "sensors")
TRANSFERS=("bcg" "ppgbp" "sensors")
BATCH_SIZES=(6 10)
SHOTS=(5 10)
GPU_IDS=(2 3 4 5 6 7)
EPOCHS=10
METHOD="original"

# Backbone-to-config mapping
declare -A CONFIG_MAP
CONFIG_MAP["spectroresnet"]="core/config/dl/spectroresnet/spectroresnet_bcg.yaml"
CONFIG_MAP["mlpbp"]="core/config/dl/mlpbp/mlpbp_bcg.yaml"
CONFIG_MAP["bptransformer"]="core/config/dl/bptransformer/bptransformer_bcg.yaml"

gpu_index=0

for backbone in "${BACKBONES[@]}"; do
  config_file="${CONFIG_MAP[$backbone]}"

  for baseline in "${BASELINES[@]}"; do
    for target in "${TARGETS[@]}"; do
      for transfer in "${TRANSFERS[@]}"; do
        if [ "$target" == "$transfer" ]; then
          continue
        fi
        for lr in "${LR_RANGE[@]}"; do
          for wd in "${WD_RANGE[@]}"; do
            for bs in "${BATCH_SIZES[@]}"; do
              for shot in "${SHOTS[@]}"; do
                gpu_id=${GPU_IDS[$gpu_index]}
                echo "[GPU:$gpu_id] backbone=$backbone, tgt=$target, tf=$transfer, lr=$lr, wd=$wd, bs=$bs, shot=$shot"

                CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
                  --baseline="$baseline" \
                  --backbone="$backbone" \
                  --batch_size="$bs" \
                  --config_file="$config_file" \
                  --epochs="$EPOCHS" \
                  --lr="$lr" \
                  --method="$METHOD" \
                  --shots="$shot" \
                  --target="$target" \
                  --transfer="$transfer" \
                  --wd="$wd" &

                gpu_index=$(( (gpu_index + 1) % ${#GPU_IDS[@]} ))
                sleep 0.3
              done
            done
          done
        done
      done
    done
  done
done

wait