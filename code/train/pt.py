import torch

# 비교할 두 모델의 경로
path1 = "pretrained_models/bcg-bptransformer/fold0.ckpt"
path2 = "pretrained_models/ppgbp-bptransformer/fold0.ckpt"

# PyTorch Lightning 기반 .ckpt는 내부적으로 state_dict 포함
ckpt1 = torch.load(path1, map_location="cpu")["state_dict"]
ckpt2 = torch.load(path2, map_location="cpu")["state_dict"]

# 모든 key 기준으로 weight 차이 확인
diff_report = {}
for key in ckpt1.keys():
    if key not in ckpt2:
        print(f"❌ Key {key} not in both checkpoints")
        continue

    same = torch.equal(ckpt1[key], ckpt2[key])
    diff_report[key] = same

# 전체 결과 요약
all_equal = all(diff_report.values())
print("✅ All weights are identical!" if all_equal else "⚠️ Differences found.")

# 차이나는 key만 출력
if not all_equal:
    for key, same in diff_report.items():
        if not same:
            print(f"❌ Different: {key}")

is_same = ckpt1 == ckpt2
print(f"🚨 Entire state_dict identical? {is_same}")

