#!/bin/bash
cd ..

GPU_IDS=(0 1 2 3 4 5 6 7)
IDX=0
TRAINING_SCRIPT="train.py"

TRANSFER_RANGE=("bcg" "ppgbp" "sensors")
TARGET_RANGE=("bcg" "ppgbp" "sensors")
BACKBONE_RANGE=("bptransformer") # "resnet1d"  "spectroresnet" "mlpbp" 
SHOTS_RANGE=(5 10)

PENALTY=""
PENALTY_SCALE_RANGE=(0)
WEIGHT_PER_PROMPT_OPTIONS=("")
QK_SIM_COEFF_RANGE=(0)
GLONORM_OPTIONS=("")
SCALING_OPTIONS=("")
LAMBDA_RANGE=(1)
PCA_DIM_RANGE=(16)
PROMPT_WEIGHTS_OPTIONS=("learnable")

PASS_PCA_OPTIONS=("--pass_pca")
HEAD_OPTIONS=("" "--train_head")
IMAG_ON=("--train_imag")
LR_RANGE=(1e-1 1e-2 1e-3 1e-4)
WD_RANGE=(1e-1 1e-2 1e-3)
TRUNC_DIM=(25 50)
BATCHSIZE_RANGE=(6)
QUERY_DIM_RANGE=(64)
GLOBAL_COEFF_RANGE=(1)
POOL_RANGE=(4)
EMB_DIFF_OPTIONS=('--use_emb_diff')
DIFF_LOSS_WEIGHT_RANGE=(1.0)

METHOD_OPTIONS=("--stepbystep")
ADD_FREQ=("--add_freq")

for BACKBONE in "${BACKBONE_RANGE[@]}"; do
  for TARGET in "${TARGET_RANGE[@]}"; do
    for TRANSFER in "${TRANSFER_RANGE[@]}"; do
      for SHOTS in "${SHOTS_RANGE[@]}"; do

        if [ "$TRANSFER" == "$TARGET" ]; then
          continue
        fi

        if [ "$BACKBONE" = "resnet1d" ]; then
          CONFIG_FILE="core/config/dl/resnet/resnet_${TRANSFER}.yaml"
        elif [ "$BACKBONE" = "spectroresnet" ]; then
          CONFIG_FILE="core/config/dl/spectroresnet/spectroresnet_${TRANSFER}.yaml"
        elif [ "$BACKBONE" = "mlpbp" ]; then
          CONFIG_FILE="core/config/dl/mlpbp/mlpbp_${TRANSFER}.yaml"
        elif [ "$BACKBONE" = "bptransformer" ]; then
          CONFIG_FILE="core/config/dl/bptransformer/bptransformer_${TRANSFER}.yaml"
        fi

        for M in "${METHOD_OPTIONS[@]}"; do
          for AF in "${ADD_FREQ[@]}"; do
            for LR in "${LR_RANGE[@]}"; do
              for WD in "${WD_RANGE[@]}"; do
                for POOL in "${POOL_RANGE[@]}"; do
                  for PS in "${PENALTY_SCALE_RANGE[@]}"; do
                    for GLONORM in "${GLONORM_OPTIONS[@]}"; do
                      for GC in "${GLOBAL_COEFF_RANGE[@]}"; do
                        for QK in "${QK_SIM_COEFF_RANGE[@]}"; do
                          for PCADIM in "${PCA_DIM_RANGE[@]}"; do
                            for WPP in "${WEIGHT_PER_PROMPT_OPTIONS[@]}"; do
                              for SO in "${SCALING_OPTIONS[@]}"; do
                                for PW in "${PROMPT_WEIGHTS_OPTIONS[@]}"; do
                                  for BZ in "${BATCHSIZE_RANGE[@]}"; do
                                    for HD in "${HEAD_OPTIONS[@]}"; do
                                      for QD in "${QUERY_DIM_RANGE[@]}"; do
                                        for LAM in "${LAMBDA_RANGE[@]}"; do
                                          for TD in "${TRUNC_DIM[@]}"; do
                                            for IG in "${IMAG_ON[@]}"; do
                                              for PP in "${PASS_PCA_OPTIONS[@]}"; do
                                                for ED in "${EMB_DIFF_OPTIONS[@]}"; do
                                                  for DLW in "${DIFF_LOSS_WEIGHT_RANGE[@]}"; do

                                                    CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python $TRAINING_SCRIPT \
                                                    --config_file $CONFIG_FILE \
                                                    --method prompt_global \
                                                    --backbone $BACKBONE \
                                                    --shots $SHOTS \
                                                    --transfer $TRANSFER \
                                                    --target $TARGET \
                                                    --query_dim $QD \
                                                    --lr $LR \
                                                    --batch_size $BZ \
                                                    --wd $WD \
                                                    --num_pool $POOL \
                                                    --global_coeff $GC \
                                                    --qk_sim_coeff $QK \
                                                    --pca_dim $PCADIM \
                                                    --lam $LAM \
                                                    --prompt_weights $PW \
                                                    --penalty_scaler $PS \
                                                    --trunc_dim $TD \
                                                    --diff_loss_weight $DLW \
                                                    $M \
                                                    $ED \
                                                    $IG \
                                                    $AF \
                                                    $SO \
                                                    $GLONORM \
                                                    $PENALTY \
                                                    $WPP \
                                                    $PP \
                                                    $HD &


                                                    IDX=$(( ($IDX + 1) % ${#GPU_IDS[@]} ))
                                                    if [ $IDX -eq 0 ]; then
                                                      wait
                                                    fi

                                                  done
                                                done
                                              done
                                            done
                                          done
                                        done
                                      done
                                    done
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done

      done
    done
  done
done
