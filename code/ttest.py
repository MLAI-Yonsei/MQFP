import pandas as pd, numpy as np
from scipy.stats import ttest_1samp, wilcoxon

ALPHA = 0.05          # ← 유의수준 여기서 조절 (0.05, 0.10, …)

df = pd.read_csv("wandb_fold_metrics_2025-05-20.csv")
df = df[df["backbone"] == "bptransformer"]

metrics = ["GAL", "SPDP"]
rows = []

for shots in sorted(df["shots"].unique()):
    d_orig   = df[(df["shots"] == shots) & (df["method"] == "original")]
    d_prompt = df[(df["shots"] == shots) & (df["method"] == "prompt_global")]

    merged = (d_orig[["transfer","target","fold"] + metrics]
              .merge(d_prompt, on=["transfer","target","fold"],
                     suffixes=("_o","_p")))

    for m in metrics:
        diff = merged[f"{m}_p"] - merged[f"{m}_o"]      # ours - orig (음수면 개선)
        if len(diff) < 5:
            continue

        # 단-측(ours < orig)
        t_stat, p_two = ttest_1samp(diff, popmean=0.0)
        p_t = p_two/2 if t_stat < 0 else 1.0            # one-tailed
        try:
            _, p_w = wilcoxon(diff, alternative="less")
        except ValueError:                              # diff가 전부 0일 때
            p_w = np.nan

        # 부트스트랩 CI
        boot = np.random.default_rng().choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
        ci_l, ci_h = np.percentile(boot, [2.5, 97.5])

        rows.append({
            "shots": shots,
            "metric": m,
            "N": len(diff),
            "mean_diff": diff.mean(),
            "t_p": p_t,
            "wilcoxon_p": p_w,
            "sig_t": p_t < ALPHA,          # ← α와 비교
            "sig_w": p_w < ALPHA,
            "CI_low": ci_l,
            "CI_high": ci_h
        })

out = pd.DataFrame(rows)
print(out.to_string(index=False, float_format="%.4f"))
