import wandb

# Run ID 목록
run_ids = [
    "e27vgfil", "9ptlrtzq", "23kphaty", "7m1ltxpa", "88lvqagu", "gxgveqrp", "152zjw6r", "1hc9fqo7",
    "hq0t8zav", "iqrlv0vp", "4qtab1ve", "bnrfrmcs", "g3r4f5g6", "3s47lhwz", "p2d6yvs6", "r7ozw7u7",
    "5gn9yuld", "8mafbd9u", "i37o0d2e", "zxywqunl", "99046yae", "jq47be1o", "1uc5mm4b", "jkte6o9e",
    "721ddut6", "2d1vpo59", "imlxavbh", "z8ar45wh", "29ji834k", "uys13zz6", "puwiocm4", "4wxuaafs",
    "naucf5pn", "aicdguab", "6njr55tx", "uh5rxiuv", "vpgzo23v", "gg6u1pk8", "ogrg338j", "nhfee6d6",
    "scvtj2u0", "7t1iz0go", "vcdgqu5k", "81htbmjw", "c02uvlty", "a0vo0uxe", "gpozew7v", "itnarvmf",
    "3nk94bt4", "e2cted5w", "r0rbqns8", "vcweklkc", "zzdrlk1g", "nwru8yyt", "dof2a8yt", "z2p61equ",
    "7z9vdwat", "3a3gf3sq", "uikg5gix", "s9029ynj", "zjy492r0", "i3q25se7", "ro6rnbej", "9qr1f5s0",
    "5zhh7mw4", "jr9qwd1x", "zd1qljvn", "1awngvdp", "z0xysnx6", "hsg2g5ct", "8ipahonn", "97teik54",
    "q2mwko1y", "lox73avg", "dn8s1zzm", "uh341l1h", "biww7zzn", "7broj02m", "yt4j7o8e", "anw3h9nn"
]

api = wandb.Api()
PROJECT = "l2p_bp/mqfp_new"

# with open("reproduce_runs.sh", "w") as f:
#     f.write("#!/bin/bash\n\n")

#     for run_id in run_ids:
#         try:
#             run = api.run(f"{PROJECT}/{run_id}")
#             config = run.config

#             args_str = " ".join(
#                 f"--{key} {value}" for key, value in config.items()
#                 if not key.startswith("_")
#             )

#             command = f"python train.py {args_str}"
#             f.write(f"# {run_id}\n{command}\n\n")

#         except Exception as e:
#             f.write(f"# {run_id} [ERROR] {e}\n\n")
