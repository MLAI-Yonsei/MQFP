#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fold-level CSV → run-level 집계 → 테이블 출력

CSV 열: run_id, backbone, method, transfer, target, shots, fold, GAL, SPDP
"""

import pandas as pd
from tabulate import tabulate

# ────────────────────────────────────────────
# 0.  파일 경로
# ────────────────────────────────────────────
CSV_FILE = "/data1/bubble3jh/bp_L2P/code/wandb_fold_metrics_2025-05-20.csv"

# ────────────────────────────────────────────
# 1.  run-level 집계
# ────────────────────────────────────────────
raw = pd.read_csv(CSV_FILE)

def _agg_run(group: pd.DataFrame) -> pd.Series:
    """각 run_id 당 평균·표준편차 + 메타데이터 한 번에 추출"""
    meta = group.iloc[0]        # backbone, method … 동일하다고 가정
    return pd.Series({
        "backbone":   meta.backbone,
        "method":     meta.method,
        "transfer":   meta.transfer,
        "target":     meta.target,
        "shots":      int(meta.shots),
        "gal_mean":   group["GAL"].mean(),
        "spdp_mean":  group["SPDP"].mean(),
        "gal_std":    group["GAL"].std(ddof=1),
        "spdp_std":   group["SPDP"].std(ddof=1),
    })

runs_df = raw.groupby("run_id").apply(_agg_run).reset_index()

# ────────────────────────────────────────────
# 2.  테이블 출력 함수
# ────────────────────────────────────────────
TRANSFER_TARGET_ORDER = [
    "ppgbp→bcg", "sensors→bcg", "bcg→ppgbp", "sensors→ppgbp",
    "bcg→sensors", "ppgbp→sensors", "vital_ecg→mimic_ecg", "mimic_ecg→vital_ecg",
]
ROW_ORDER = ["spectroresnet", "mlpbp", "resnet1d", "bptransformer", "bpt+ours"]

def show_results(*, shots: int, sort_by: str = "spdp_std", chunk_size: int = 3):
    """
    sort_by ∈ {"gal_mean","spdp_mean","gal_std","spdp_std"}
        • mean 계열, std 계열 모두 '낮을수록 좋다' 기준
    """
    assert sort_by in {"gal_mean","spdp_mean","gal_std","spdp_std"}
    other = {"gal_mean":"spdp_mean", "spdp_mean":"gal_mean",
             "gal_std":"spdp_std",   "spdp_std":"gal_std"}[sort_by]

    df = runs_df[runs_df.shots == shots].copy()

    # label 정제 ─────────────────────────────
    df["label"] = df.backbone
    df.loc[(df.backbone == "bptransformer") &
           (df.method  == "prompt_global"), "label"] = "bpt+ours"
    df["transfer_target"] = df.transfer + "→" + df.target

    # best run (낮을수록 좋음) ────────────────
    idx_best = (df.groupby(["label","transfer_target"])[sort_by]
                  .idxmin())
    best = df.loc[idx_best, ["label","transfer_target", sort_by, other]]

    # pivot (label 행, 멀티컬럼) ──────────────
    pivot = (best.set_index(["label","transfer_target"])
                  .unstack("transfer_target"))
    pivot.columns = [(tt, m) for m, tt in pivot.columns]      # (tt, metric)

    # 순서 고정
    cols = [(tt, m) for tt in TRANSFER_TARGET_ORDER for m in (sort_by, other)]
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(cols))
    pivot = pivot.reindex(ROW_ORDER)

    # 이모지 붙이기 ──────────────────────────
    for tt in TRANSFER_TARGET_ORDER:
        for m in (sort_by, other):
            col = (tt, m)
            vals = pivot[col].astype(float)
            uniq = vals.dropna().unique()
            uniq.sort()                               # 낮을수록 좋음
            best_val   = uniq[0]  if len(uniq) else None
            second_val = uniq[1]  if len(uniq) > 1 else None

            def mark(x):
                if pd.isna(x):          return "—"
                if x == best_val:       return f"{x:.2f}"
                if x == second_val:     return f"{x:.2f}"
                return f"{x:.2f}"
            pivot[col] = vals.apply(mark)

    # chunk 출력 ─────────────────────────────
    chunks = [TRANSFER_TARGET_ORDER[i:i+chunk_size]
              for i in range(0, len(TRANSFER_TARGET_ORDER), chunk_size)]
    for chunk in chunks:
        sub = pivot.loc[:, pd.IndexSlice[chunk, [sort_by, other]]]
        title = f"\n=== Shots: {shots} | best by {sort_by} | targets={chunk} ==="
        print(title)
        print(tabulate(sub, headers="keys", tablefmt="grid", stralign="center"))
        print()

# ────────────────────────────────────────────
# 3.  실행 예시
# ────────────────────────────────────────────
if __name__ == "__main__":
    show_results(shots=5, sort_by="spdp_std", chunk_size=8)
    show_results(shots=10, sort_by="spdp_std", chunk_size=8)
