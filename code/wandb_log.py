import wandb
import pandas as pd
import os
from tabulate import tabulate

wandb.login()
api = wandb.Api(timeout=60)

# 구성
shots_list = [5, 10]
backbones = ["mlpbp", "spectroresnet", "bptransformer"]
metric_sets = {
    "mae_sum": ("sbp", "dbp"),
    "gal_sum": ("sbp_gal", "dbp_gal"),
}
cache_file = "./wandb_runs.parquet"

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
                runs = api.runs(project_name)
            except Exception as e:
                print(f"❌ Failed to load project {project_name}: {e}")
                continue

            for run in runs:
                transfer = run.config.get("transfer")
                target = run.config.get("target")

                if not transfer or not target:
                    print(f"⚠️ Missing transfer/target in run {run.name}, skipping")
                    continue

                for metric_type, (m1, m2) in metric_sets.items():
                    try:
                        val1 = run.summary.get(m1)
                        val2 = run.summary.get(m2)
                        if val1 is not None and val2 is not None:
                            total = val1 + val2
                            rows.append({
                                "shots": shots,
                                "backbone": backbone,
                                "transfer": transfer,
                                "target": target,
                                "metric_type": metric_type,
                                "avg_sum": total
                            })
                    except Exception as e:
                        print(f"⚠️ Error in run {run.name}: {e}")

    df = pd.DataFrame(rows)
    df.to_parquet(cache_file)
    print(f"✅ Saved to cache: {cache_file}")

# 보기 좋게 출력
def print_multiindex_table(df, shots, metric_type, title):
    df_sub = df[(df["shots"] == shots) & (df["metric_type"] == metric_type)]
    df_grouped = df_sub.groupby(["backbone", "transfer", "target"])["avg_sum"].min().reset_index()
    df_pivot = df_grouped.pivot(index="backbone", columns=["transfer", "target"], values="avg_sum")
    df_pivot = df_pivot.round(4)

    # ✅ 성능 낮은 순 정렬 (NaN 무시)
    df_pivot["avg"] = df_pivot.min(axis=1, skipna=True)
    df_pivot = df_pivot.sort_values(by="avg").drop(columns="avg")

    order = ["mlpbp", "spectroresnet", "bptransformer"]
    df_pivot = df_pivot.reindex(order)      # ★ 여기 한 줄
    
    print(f"\n📊 {title} (shots={shots}, lower is better):\n")
    print(tabulate(df_pivot, headers="keys", tablefmt="github"))

# 출력
print_multiindex_table(df, 5, "mae_sum", "SBP+DBP MAE Sum")
print_multiindex_table(df, 10, "mae_sum", "SBP+DBP MAE Sum")
print_multiindex_table(df, 5, "gal_sum", "SBP+DBP GAL Sum")
print_multiindex_table(df, 10, "gal_sum", "SBP+DBP GAL Sum")

def print_best_counts(df, shots, metric_type, order):
    df_sub = df[(df["shots"] == shots) & (df["metric_type"] == metric_type)]

    # (1) 각 조합의 최소값만 남긴 피벗
    pivot = pd.pivot_table(
        df_sub,
        index="backbone",
        columns=["transfer", "target"],
        values="avg_sum",
        aggfunc="min"
    )

    # (2) transfer-target 열마다 어떤 backbone이 최소인가?
    best_per_pair = pivot.idxmin()            # Series: col → backbone
    counts = best_per_pair.value_counts().reindex(order, fill_value=0)

    print(f"\n🏆 Best-count summary  |  metric={metric_type}, shots={shots}")
    for b in order:
        print(f"{b:14}: {counts[b]}")

order = ["mlpbp", "spectroresnet", "bptransformer"]

for s in shots_list:
    for m in metric_sets.keys():
        print_best_counts(df, s, m, order)
