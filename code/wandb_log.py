import wandb
import pandas as pd
import os
from tabulate import tabulate

wandb.login()
api = wandb.Api(timeout=60)

# 구성
shots_list = [5, 10]
backbones = ["mlpbp", "spectroresnet", "bptransformer"]
cache_file = "./wandb_runs_bestonly.parquet"

# 캐시 불러오기
if os.path.exists(cache_file):
    print("📦 Using cached results...")
    df = pd.read_parquet(cache_file)
else:
    print("📡 Fetching results from WandB...")
    rows = []

    for shots in shots_list:
        for backbone in backbones:
            project_name = f"l2p_bp/mqfp_{shots}shot_backbone{backbone}"
            try:
                runs = api.runs(project_name, filters={
                    "summary.spdp": {"$ne": None},
                    "summary.gal": {"$ne": None}
                })
            except Exception as e:
                print(f"❌ Failed to load project {project_name}: {e}")
                continue

            for run in runs:
                transfer = run.config.get("transfer")
                target = run.config.get("target")
                method = run.config.get("method", "original")

                if not transfer or not target:
                    continue

                row = {
                    "shots": shots,
                    "backbone": backbone,
                    "transfer": transfer,
                    "target": target,
                    "spdp": run.summary.get("spdp"),
                    "gal": run.summary.get("gal"),
                    "method": method,
                }

                rows.append(row)
                if backbone == "bptransformer" and method == "prompt_global":
                    row_ours = row.copy()
                    row_ours["backbone"] = "bptransformer+ours"
                    rows.append(row_ours)

    df = pd.DataFrame(rows)
    df.to_parquet(cache_file)
    print(f"✅ Saved to cache: {cache_file}")

# best spdp/gal filtering
def best_by_metric(df, metric):
    return (
        df.sort_values(by=metric)
        .dropna(subset=[metric])
        .groupby(["shots", "backbone", "transfer", "target"], as_index=False)
        .first()
    )

best_spdp = best_by_metric(df, "spdp")
best_gal = best_by_metric(df, "gal")

# 보기 좋게 출력
def print_filtered_table(df, value_column, shots, title):
    df_sub = df[df["shots"] == shots]
    df_pivot = df_sub.pivot(index="backbone", columns=["transfer", "target"], values=value_column)
    df_pivot["avg"] = df_pivot.min(axis=1, skipna=True)
    df_pivot = df_pivot.sort_values(by="avg").drop(columns="avg")
    order = ["mlpbp", "spectroresnet", "bptransformer", "bptransformer+ours"]
    df_pivot = df_pivot.reindex(order)
    print(f"\n📊 {title} (shots={shots}, lower is better):\n")
    print(tabulate(df_pivot.round(4), headers="keys", tablefmt="github"))

# 출력
print_filtered_table(best_spdp, "spdp", 5, "Best SPDP (SBP+DBP)")
print_filtered_table(best_spdp, "spdp", 10, "Best SPDP (SBP+DBP)")
print_filtered_table(best_gal, "gal", 5, "Best GAL (SBP+DBP GAL)")
print_filtered_table(best_gal, "gal", 10, "Best GAL (SBP+DBP GAL)")

# Best count summary
def print_best_counts(df, shots, value_column, order):
    df_sub = df[df["shots"] == shots]
    pivot = pd.pivot_table(
        df_sub,
        index="backbone",
        columns=["transfer", "target"],
        values=value_column,
        aggfunc="min"
    )
    best_per_pair = pivot.idxmin()
    counts = best_per_pair.value_counts().reindex(order, fill_value=0)
    print(f"\n🏆 Best-count summary  |  metric={value_column}, shots={shots}")
    for b in order:
        print(f"{b:18}: {counts[b]}")

order = ["mlpbp", "spectroresnet", "bptransformer", "bptransformer+ours"]

for s in shots_list:
    print_best_counts(best_spdp, s, "spdp", order)
    print_best_counts(best_gal, s, "gal", order)
