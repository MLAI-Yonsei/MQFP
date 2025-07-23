#!/bin/bash

# BACKBONES=("bptransformer" "spectroresnet" "mlpbp")
BACKBONES=("resnet1d")
BASELINES=("ft")

GPU_IDS=(0 1 2 3 4 5 6 7)
EPOCHS=10
METHOD="original"

shot=10
TARGETS=("bcg" "ppgbp" "sensors")
TRANSFERS=("bcg" "ppgbp" "sensors")

declare -A CONFIG_BASE_PATH
CONFIG_BASE_PATH["spectroresnet"]="core/config/dl/spectroresnet"
CONFIG_BASE_PATH["mlpbp"]="core/config/dl/mlpbp"
CONFIG_BASE_PATH["bptransformer"]="core/config/dl/bptransformer"
CONFIG_BASE_PATH["resnet1d"]="core/config/dl/resnet"
job_count=0

for backbone in "${BACKBONES[@]}"; do
  config_dir="${CONFIG_BASE_PATH[$backbone]}"
  for baseline in "${BASELINES[@]}"; do
    for target in "${TARGETS[@]}"; do
      for transfer in "${TRANSFERS[@]}"; do
        if [ "$target" == "$transfer" ]; then
          continue
        fi

        if [[ "$transfer" == "bcg" && "$target" == "ppgbp" ]]; then
            lr=0.005
            wd=0.1
            bs=6
        elif [[ "$transfer" == "bcg" && "$target" == "sensors" ]]; then
            lr=0.007
            wd=0.005
            bs=5
        elif [[ "$transfer" == "ppgbp" && "$target" == "bcg" ]]; then
            lr=0.05
            wd=0.01
            bs=6
        elif [[ "$transfer" == "ppgbp" && "$target" == "sensors" ]]; then
            lr=0.01
            wd=0.02
            bs=2
        elif [[ "$transfer" == "sensors" && "$target" == "bcg" ]]; then
            lr=0.06
            wd=0.03
            bs=2
        elif [[ "$transfer" == "sensors" && "$target" == "ppgbp" ]]; then
            lr=0.01
            wd=0.09
            bs=9
        fi
        if [ "$backbone" == "resnet1d" ]; then
          config_file="core/config/dl/resnet/resnet_${transfer}.yaml"
        else
          config_file="${config_dir}/${backbone}_${transfer}.yaml"
        fi
                gpu_index=$((job_count % ${#GPU_IDS[@]}))
                gpu_id=${GPU_IDS[$gpu_index]}

                echo "[GPU:$gpu_id] backbone=$backbone, tgt=$target, tf=$transfer, lr=$lr, wd=$wd, bs=$bs, shot=$shot, baseline=$baseline"

                CMD="CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
                  --backbone $backbone \
                  --batch_size $bs \
                  --config_file $config_file \
                  --epochs $EPOCHS \
                  --lr $lr \
                  --method $METHOD \
                  --shots $shot \
                  --target $target \
                  --transfer $transfer \
                  --baseline $baseline \
                  --wd $wd"

                eval "$CMD &"

                job_count=$((job_count + 1))
                if (( job_count % ${#GPU_IDS[@]} == 0 )); then
                  wait
                fi
      done
    done
  done
done

wait