import os
import pandas as pd
import wandb
from tabulate import tabulate
from datetime import datetime, timezone

# --- 설정 --------------------------------------------------
PROJECT = "l2p_bp/mqfp_fold_new"
VALID_PAIRS = [
    ("ppgbp","bcg"), ("ppgbp","sensors"), ("bcg","ppgbp"), ("sensors","ppgbp"),
    ("bcg","sensors"), ("sensors","bcg"),
    ("mimic_ecg","vital_ecg"), ("vital_ecg","mimic_ecg")
]
FROM_DATE = datetime(2025, 5, 20, tzinfo=timezone.utc)
TRANSFER_TARGET_ORDER = [
    "ppgbp→bcg",
    "sensors→bcg",
    "bcg→ppgbp",
    "sensors→ppgbp",
    "bcg→sensors",
    "ppgbp→sensors",
    "vital_ecg→mimic_ecg",
    "mimic_ecg→vital_ecg"
]
# -----------------------------------------------------------

api = wandb.Api()

def fetch_fold_level_records():
    """각 run에서 Fold별 GAL/SPDP까지 추출"""
    or_filters = [{"$and":[{"config.transfer":t},{"config.target":u}]} 
                  for t,u in VALID_PAIRS]
    filters = {"$and": [
        {"$or": or_filters},
        {"createdAt": {"$gt": FROM_DATE.isoformat()}}
    ]}

    runs = api.runs(PROJECT, filters=filters, per_page=1000)
    fold_recs = []

    for run in runs:
        created = datetime.fromisoformat(run.created_at.replace("Z","+00:00"))
        tran, tgt = run.config.get("transfer"), run.config.get("target")
        if (tran, tgt) not in VALID_PAIRS:
            continue

        backbone = run.config.get("backbone", "")
        method = run.config.get("method", "")
        shots = run.config.get("shots", "N/A")

        # Fold0~Fold4 assumed
        for fold in range(5):
            gal_key = f"Fold{fold}/GAL"
            spdp_key = f"Fold{fold}/SPDP"
            gal_val = run.summary.get(gal_key)
            spdp_val = run.summary.get(spdp_key)

            if gal_val is not None or spdp_val is not None:
                fold_recs.append({
                    "run_id":   run.id,
                    "backbone": backbone,
                    "method":   method,
                    "transfer": tran,
                    "target":   tgt,
                    "shots":    shots,
                    "fold":     fold,
                    "GAL":      gal_val,
                    "SPDP":     spdp_val
                })

    return pd.DataFrame(fold_recs)

if __name__ == "__main__":
    df_folds = fetch_fold_level_records()
    print(df_folds.head())  # 미리보기

    csv_path = f"wandb_fold_metrics_{FROM_DATE.date()}.csv"
    df_folds.to_csv(csv_path, index=False)
    print(f"✅ Saved fold-level metrics to {csv_path}")
