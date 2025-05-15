import os
import pickle
import pandas as pd
import wandb
from tabulate import tabulate
from datetime import datetime, timezone

# --- 설정 --------------------------------------------------
PROJECT    = "l2p_bp/mqfp_new"
CACHE_FILE = "runs_records.pkl"

VALID_PAIRS = [
    ("ppgbp","bcg"), ("ppgbp","sensors"), ("bcg","ppgbp"), ("sensors","ppgbp"),
    ("bcg","sensors"), ("sensors","bcg"),
    ("mimic_ecg","vital_ecg"), ("vital_ecg","mimic_ecg")
]
# -----------------------------------------------------------

api = wandb.Api()

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return [], None
    data = pickle.load(open(CACHE_FILE, "rb"))
    last = datetime.fromisoformat(data["last_fetch"])
    return data["records"], last

def save_cache(records, last_fetch):
    data = {
        "records":    records,
        "last_fetch": last_fetch.astimezone(timezone.utc).isoformat()
    }
    pickle.dump(data, open(CACHE_FILE, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

def get_latest_run_time():
    """W&B에서 valid_pairs 전체 중 최신 한 건만 가져와서 createdAt 반환"""
    or_filters = [{"$and":[{"config.transfer":t},{"config.target":u}]} 
                  for t,u in VALID_PAIRS]
    latest = next(api.runs(
        PROJECT,
        filters={"$or": or_filters},
        order="-createdAt",
        per_page=1
    ), None)

    if latest is None:
        return None
    # run.created_at은 ISO 문자열
    iso = latest.created_at.replace("Z", "+00:00")
    return datetime.fromisoformat(iso)

def fetch_new_records(existing, last_fetch):
    """last_fetch 이후에 생성된 런만 가져와서 records에 append"""
    or_filters = [{"$and":[{"config.transfer":t},{"config.target":u}]} 
                  for t,u in VALID_PAIRS]
    # AND 조합으로 createdAt 필터까지
    filters = {"$and": [
        {"$or": or_filters},
        {"createdAt": {"$gt": last_fetch.astimezone(timezone.utc).isoformat()}}
    ]}

    runs = api.runs(PROJECT, filters=filters, per_page=500)
    new_recs = []
    max_fetch = last_fetch

    for run in runs:
        created = datetime.fromisoformat(run.created_at.replace("Z","+00:00"))
        tran, tgt = run.config.get("transfer"), run.config.get("target")
        if (tran,tgt) not in VALID_PAIRS:
            continue

        for metric in ("gal","spdp"):
            val = run.summary.get(metric)
            if val is None:
                continue
            new_recs.append({
                "run_id":     run.id,
                "created_at": created,
                "backbone":   run.config.get("backbone",""),
                "method":     run.config.get("method",""),
                "transfer":   tran,
                "target":     tgt,
                "metric":     metric,
                "value":      val,
                "shots":      run.config.get("shots","N/A"),
                "train_head": run.config.get("train_head","N/A"),
                "baseline":   run.config.get("baseline","N/A")
            })
        if created > max_fetch:
            max_fetch = created

    return existing + new_recs, max_fetch

# --- 메인 ---------------------------------------------------
records, last_fetch = load_cache()
print(f"Loaded {len(records)} records; last fetch at {last_fetch}")

latest_time = get_latest_run_time()
if latest_time is None:
    print("No runs at all in project.")
elif last_fetch and latest_time <= last_fetch:
    print("No new runs since last fetch – skipping W&B query.")
else:
    # 신규 런이 있으니 createdAt 필터로 최소한만 fetch
    records, new_fetch = fetch_new_records(records, last_fetch or datetime.min.replace(tzinfo=timezone.utc))
    print(f"Fetched {len(records) - len(records):d} new records; new last_fetch={new_fetch}")
    save_cache(records, new_fetch)

# 파일 최상단에 한 줄 추가
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

def show_results(shots, sort_by="spdp", chunk_size=3):
    """
    shots: int
    sort_by: "spdp" (낮을수록 좋음) 또는 "gal" (높을수록 좋음)
    chunk_size: 한 번에 보여줄 transfer_target 수
    """
    assert sort_by in ("spdp", "gal")

    df = pd.DataFrame(records)
    df = df[df.shots == shots]

    # label 벡터화
    mask = (df.backbone=="bptransformer") & (df.method=="prompt_global")
    df["label"] = df.backbone
    df.loc[mask, "label"] = "bpt+ours"
    df["transfer_target"] = df.transfer + "→" + df.target

    # 1) best run 인덱스 찾기
    df_metric = df[df.metric == sort_by]
    best_idx = df_metric.groupby(["label","transfer_target"])["value"].idxmin()

    best = df_metric.loc[best_idx, ["label","transfer_target","run_id","value"]].copy()
    best = best.rename(columns={"value": sort_by})

    # 2) 다른 메트릭 값 맵핑
    other = "gal" if sort_by=="spdp" else "spdp"
    df_other = df[df.metric==other][["run_id","value"]].set_index("run_id")
    best[other] = best["run_id"].map(df_other["value"])

    # 3) pivot → 멀티인덱스 컬럼: (transfer_target, metric)
    pivot = best.set_index(["label","transfer_target"])[[sort_by, other]]
    pivot = pivot.unstack("transfer_target")

    # 4) 컬럼 순서 고정: transfer_target 우선, 그 안에 [sort_by, other] metric
    metrics = [sort_by, other]
    cols = []
    for tt in TRANSFER_TARGET_ORDER:
        for m in metrics:
            cols.append((tt, m))
    # pivot.columns: (metric, transfer_target) → (transfer_target, metric)으로 변환
    pivot.columns = [(tt, m) for m, tt in pivot.columns]
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(cols), fill_value=float("nan"))

    # 5) 행 순서 고정
    row_order = ["resnet1d","spectroresnet","mlpbp","bptransformer","bpt+ours"]
    pivot = pivot.reindex(row_order)

    # 6) chunk 단위로 나눠서 출력
    chunks = [
        TRANSFER_TARGET_ORDER[i : i + chunk_size]
        for i in range(0, len(TRANSFER_TARGET_ORDER), chunk_size)
    ]

    for chunk in chunks:
        # 해당 chunk에 속한 컬럼만 골라서
        sub = pivot.loc[:, pd.IndexSlice[chunk, metrics]].copy()

        # 7) 각 컬럼별 1, 2등 이모지 추가
        for tt in chunk:
            for m in metrics:
                col = (tt, m)
                vals = sub[col].astype(float)

                # NaN 제거 후 정렬된 고유 값 추출
                sorted_vals = vals.dropna().unique()
                sorted_vals.sort()  # 오름차순 기준: 낮을수록 좋음

                best_val = sorted_vals[0] if len(sorted_vals) > 0 else None
                second_val = sorted_vals[1] if len(sorted_vals) > 1 else None

                def add_emoji(x):
                    if pd.isna(x):
                        return "nan"
                    elif x == best_val:
                        return f"{x:.2f} 🥇"
                    elif x == second_val:
                        return f"{x:.2f} 🥈"
                    else:
                        return f"{x:.2f}"

                sub[col] = vals.apply(add_emoji)
        
        # 8) 표 출력
        header = f"\n=== Shots: {shots} | best by {sort_by} | targets={chunk} ==="
        print(header)
        print(tabulate(sub, headers="keys", tablefmt="grid"))    

if __name__=="__main__":
    show_results(5, sort_by="spdp", chunk_size=2)
    show_results(10, sort_by="spdp", chunk_size=2)
