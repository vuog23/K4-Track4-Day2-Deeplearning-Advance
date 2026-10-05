"""One-epoch timing pilot. Uses unique Q IDs and never evaluates the test set."""
from __future__ import annotations

import pandas as pd
import argparse

try:
    from .train import Config, run
except ImportError:
    from train import Config, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prefix", default="Q")
    args = parser.parse_args()
    models = ["resnet50", "resnext50_32x4d", "deit_small_patch16_224",
              "resnet18", "mobilenetv3_large_100"]
    rows = []
    for i, backbone in enumerate(models, 1):
        cfg = Config(exp_id=f"{args.prefix}{i:02d}", backbone=backbone, seed=0, epochs=1,
                     batch_size=args.batch_size, num_workers=0, amp=True,
                     out_dir="submissions/_draft_vuong/runs_budget",
                     pred_dir="submissions/_draft_vuong/predictions_budget")
        result = run(cfg)
        rows.append(result)
        pd.DataFrame(rows).to_csv(f"submissions/_draft_vuong/budget_pilot_{args.prefix}.csv", index=False)
        print("PILOT_RESULT", result, flush=True)


if __name__ == "__main__":
    main()
