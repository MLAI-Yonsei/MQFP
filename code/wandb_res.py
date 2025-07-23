import wandb
import pandas as pd
import numpy as np
import os

PROJECTS = [
    "l2p_bp/freq_prompt_sym_shot10",
    "l2p_bp/freq_prompt_sym_shot5"
]
LR_RANGE = [1e-1, 1e-2, 1e-3, 1e-4]
WD_RANGE = [1e-1, 1e-2, 1e-3]
KEYS_TO_EXTRACT = [
    "spdp", "gal", "shots", "target", "transfer", "train_head", "baseline", "created_at", "method"
]

def in_range(val, candidates, tol=1e-8):
    return any(np.isclose(val, c, atol=tol) for c in candidates)

# wandb 로그인 및 timeout 확장
api = wandb.Api(timeout=60)

all_records = []

for project in PROJECTS:
    print(f"Processing project: {project}")
    try:
        runs = api.runs(project)
    except Exception as e:
        print(f"Failed to load runs from {project}: {e}")
        continue

    for run in runs:
        config = run.config
        lr = config.get("lr")
        wd = config.get("weight_decay")

        if lr is None or wd is None:
            continue

        if in_range(lr, LR_RANGE) and in_range(wd, WD_RANGE):
            record = {}
            for key in KEYS_TO_EXTRACT:
                val = config.get(key) or run.summary.get(key) or getattr(run, key, None)
                record[key] = val
            record["created_at"] = run.created_at.isoformat()
            record["project"] = project
            all_records.append(record)

df = pd.DataFrame(all_records)
os.makedirs("wandb_exports", exist_ok=True)
df.to_csv("wandb_exports/filtered_runs.csv", index=False)

print(f"총 {len(df)}개의 run이 저장되었습니다.")
