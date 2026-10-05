"""Reproducible Step 1/2 sweeps. All model selection reads validation metrics only."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .train import Config, run, REPO_ROOT
    from . import inference, dataset, model as model_lib, benchmark
except ImportError:
    from train import Config, run, REPO_ROOT
    import inference, dataset, model as model_lib, benchmark

from eval import save_predictions


BACKBONES = [
    "resnet50", "resnext50_32x4d", "deit_small_patch16_224", "efficientnet_b0",
    "mobilenetv3_large_100",
]


def _release_cuda_memory():
    """Release cached memory between independent runs (useful on Kaggle T4s)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_budget_pilot(batch_size=8, img_size=224, out_csv="results/budget_pilot.csv",
                     out_dir="runs_budget", pred_dir="predictions_budget", num_workers=2,
                     epochs=10):
    """Time one epoch of each planned backbone and estimate the full 20-run budget."""
    csv_path = Path(out_csv)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    previous = {}
    if csv_path.exists():
        try:
            previous = {str(r["exp_id"]): r for r in pd.read_csv(csv_path).to_dict("records")}
        except (pd.errors.EmptyDataError, KeyError, ValueError):
            previous = {}
    rows = []
    for i, backbone in enumerate(BACKBONES, 1):
        exp_id = f"Q{i:02d}"
        run_path = REPO_ROOT / out_dir / exp_id / "seed0"
        config_path = run_path / "config.json"
        reusable = False
        if exp_id in previous and config_path.exists():
            try:
                saved = json.loads(config_path.read_text(encoding="utf-8"))["config"]
                reusable = (
                    saved.get("backbone") == backbone and saved.get("seed") == 0
                    and saved.get("epochs") == 1 and saved.get("batch_size") == batch_size
                    and saved.get("img_size") == img_size
                    and saved.get("num_workers") == num_workers
                    and all((run_path / name).is_file() for name in
                            ("best.pt", "history.csv", "val_logits.npy", "val_labels.npy"))
                )
            except (KeyError, OSError, ValueError, TypeError):
                reusable = False
        if reusable:
            print(f"Reusing completed {exp_id} ({backbone}) from this pilot.", flush=True)
            row = previous[exp_id]
        else:
            print(f"Starting {exp_id}/{len(BACKBONES)}: {backbone} (one epoch)", flush=True)
            row = run(Config(exp_id=exp_id, backbone=backbone, seed=0,
                             batch_size=batch_size, epochs=1, img_size=img_size, amp=True,
                             out_dir=out_dir, pred_dir=pred_dir, num_workers=num_workers))
            _release_cuda_memory()
        rows.append(row)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    frame = pd.DataFrame(rows)
    mean_epoch_seconds = float(frame.seconds_per_epoch.mean())
    # 5 backbone runs + 9 single-backbone ablations + 6 final/baseline runs.
    estimated_hours = epochs * (float(frame.seconds_per_epoch.sum()) +
                                15 * mean_epoch_seconds) / 3600.0
    print(f"Rough GPU estimate for 20 runs × {epochs} epochs: {estimated_hours:.2f} h "
          f"(based on this pilot; add setup/evaluation overhead).")
    return frame


def run_backbones(out_csv="results/backbones.csv", batch_size=8, epochs=12, img_size=224,
                  out_dir="runs", pred_dir="predictions", num_workers=2):
    rows = []
    for i, backbone in enumerate(BACKBONES, 1):
        print(f"Starting B{i:02d}/{len(BACKBONES)}: {backbone}", flush=True)
        row = run(Config(exp_id=f"B{i:02d}", backbone=backbone, seed=0, batch_size=batch_size,
                         epochs=epochs, img_size=img_size, amp=True, out_dir=out_dir,
                         pred_dir=pred_dir, num_workers=num_workers))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = model_lib.build_model(backbone, pretrained=False, init="finetune").to(device)
        checkpoint_path = REPO_ROOT / out_dir / f"B{i:02d}" / "seed0" / "best.pt"
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        net.load_state_dict(checkpoint["model"])
        net = model_lib.use_data_parallel(net, device).eval()
        latency = benchmark.latency_report(net, 1, img_size, "fp32", str(device),
                                           warmup=10, iters=50)
        row.update({"latency_gpu": latency["gpu"], "latency_dtype": latency["dtype"],
                    "latency_batch": 1, "latency_batch1_p50_ms": latency["p50"],
                    "latency_batch1_p95_ms": latency["p95"],
                    "latency_batch1_p99_ms": latency["p99"]})
        rows.append(row)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        del net
        _release_cuda_memory()
    return pd.DataFrame(rows)


def run_training_ablation(backbone: str, out_csv="results/training.csv", batch_size=8,
                          epochs=12, img_size=224, out_dir="runs", pred_dir="predictions",
                          num_workers=2):
    """Baseline plus controlled one-factor tests and a combined candidate.

    T00 is the common baseline. Each T01-T07 changes one field only; the final T08
    combines the three candidate effects and is explicitly exploratory.
    """
    runs = [
        ("T00", "baseline", {}),
        ("T01", "A: initialization", {"init": "frozen"}),
        ("T02", "A: initialization", {"init": "scratch"}),
        ("T03", "B: augmentation", {"aug": "color"}),
        ("T04", "B: augmentation", {"aug": "trivial"}),
        ("T05", "C: loss", {"loss": "ls", "label_smoothing": 0.1}),
        ("T06", "C: loss", {"loss": "focal", "focal_gamma": 2.0}),
        ("T07", "D: sampler", {"sampler": "balanced"}),
    ]
    rows = []
    for exp_id, axis, changes in runs:
        print(f"Starting {exp_id}: {axis}", flush=True)
        row = run(Config(exp_id=exp_id, backbone=backbone, seed=0, batch_size=batch_size,
                         epochs=epochs, img_size=img_size, out_dir=out_dir,
                         pred_dir=pred_dir, num_workers=num_workers, **changes))
        row["changed_from_T00"] = json.dumps(changes, ensure_ascii=False, sort_keys=True)
        row["axis"] = axis
        rows.append(row)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        _release_cuda_memory()
    # Explicit interaction test: combine the validation-best augmentation and
    # loss variants with balanced sampling. T00 remains the shared baseline.
    by_id = {r["exp_id"]: r for r in rows}
    best_aug = max((by_id["T03"], by_id["T04"]), key=lambda r: r["macro_f1_val"])
    best_loss = max((by_id["T05"], by_id["T06"]), key=lambda r: r["macro_f1_val"])
    combined = {}
    combined.update(json.loads(best_aug["changed_from_T00"]))
    combined.update(json.loads(best_loss["changed_from_T00"]))
    combined["sampler"] = "balanced"
    combo = run(Config(exp_id="T08", backbone=backbone, seed=0, batch_size=batch_size,
                       epochs=epochs, img_size=img_size, out_dir=out_dir,
                       pred_dir=pred_dir, num_workers=num_workers, **combined))
    combo["changed_from_T00"] = json.dumps(combined, ensure_ascii=False, sort_keys=True)
    combo["axis"] = "B+C+D: combined interaction"
    rows.append(combo)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    _release_cuda_memory()
    return pd.DataFrame(rows)


def run_final_comparison(backbone: str, final_recipe: dict | None = None, seeds=(0, 1, 2),
                         batch_size=16, epochs=10, img_size=224,
                         out_dir="runs", pred_dir="predictions", num_workers=2,
                         labels_dir="data/labels"):
    """Step 4: retrain the selected recipe and T00, then save test predictions once per seed.

    Call only after Steps 1-3 are complete and the recipe is fixed using validation.
    This function is intentionally the only experiment helper that enables test output.
    """
    final_recipe = dict(final_recipe or {})
    if any(k in final_recipe for k in ("exp_id", "seed", "save_test_predictions")):
        raise ValueError("final_recipe may contain only Config recipe fields")
    rows = []
    Path("results").mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        print(f"Starting final F01/T00 comparison for seed {seed}", flush=True)
        final_row = run(Config(exp_id="F01", backbone=backbone, seed=int(seed),
                               batch_size=batch_size, epochs=epochs, img_size=img_size,
                               out_dir=out_dir, pred_dir=pred_dir, num_workers=num_workers,
                               save_test_predictions=True, **final_recipe))
        # Fit one scalar on this seed's validation logits, then calibrate the already
        # computed test logits. This does not make a second test-set model pass.
        final_dir = REPO_ROOT / out_dir / "F01" / f"seed{int(seed)}"
        pred_root = Path(pred_dir)
        if not pred_root.is_absolute():
            pred_root = REPO_ROOT / pred_root
        val_df, _, test_df = dataset.load_split(labels_dir, fold=0)
        val_logits = np.load(final_dir / "val_logits.npy")
        val_labels = np.load(final_dir / "val_labels.npy")
        temperature = inference.fit_temperature(val_logits, val_labels)
        val_probs = inference.apply_temperature(val_logits, temperature)
        test_logits = np.load(final_dir / "test_logits.npy")
        test_labels = np.load(final_dir / "test_labels.npy")
        uncal_path = pred_root / f"F01uncal_seed{int(seed)}_test.csv"
        shutil.copy2(pred_root / f"F01_seed{int(seed)}_test.csv", uncal_path)
        test_probs = inference.apply_temperature(test_logits, temperature)
        save_predictions(pred_root / f"F01_seed{int(seed)}_val.csv",
                         val_df.Filename.astype(str).tolist(),
                         val_labels, val_probs)
        save_predictions(pred_root / f"F01_seed{int(seed)}_test.csv",
                         test_df.Filename.astype(str).tolist(), test_labels, test_probs)
        final_row["temperature_val"] = temperature
        final_row["inference_method"] = "I00 one-view + temperature scaling fitted on val"
        final_row["recipe_config"] = json.dumps(final_recipe, ensure_ascii=False, sort_keys=True)
        rows.append({**final_row, "recipe": "selected validation recipe"})
        _release_cuda_memory()
        baseline_row = run(Config(exp_id="T00", backbone=backbone, seed=int(seed),
                                  batch_size=batch_size, epochs=epochs, img_size=img_size,
                                  out_dir=out_dir, pred_dir=pred_dir, num_workers=num_workers,
                                  save_test_predictions=True))
        baseline_row["inference_method"] = "I00 one-view (uncalibrated baseline)"
        baseline_row["recipe_config"] = "T00 default recipe"
        rows.append({**baseline_row, "recipe": "T00 baseline"})
        _release_cuda_memory()
        pd.DataFrame(rows).to_csv(Path("results") / "final_runs.csv", index=False)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("backbones", "training"))
    p.add_argument("--backbone", help="For training ablations; if omitted, choose best B val macro-F1")
    p.add_argument("--backbone-results", default="results/backbones.csv")
    p.add_argument("--out", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--pred-dir", default="predictions")
    p.add_argument("--num-workers", type=int, default=2)
    a = p.parse_args()
    if a.stage == "backbones":
        print(run_backbones(a.out or "results/backbones.csv", a.batch_size, a.epochs,
                            a.img_size, a.out_dir, a.pred_dir, a.num_workers))
    else:
        backbone = a.backbone
        if not backbone:
            frame = pd.read_csv(a.backbone_results)
            backbone = str(frame.sort_values("macro_f1_val", ascending=False).iloc[0]["backbone"])
            print("Selected by validation macro-F1:", backbone)
        print(run_training_ablation(backbone, a.out or "results/training.csv",
                                    a.batch_size, a.epochs, a.img_size, a.out_dir,
                                    a.pred_dir, a.num_workers))


if __name__ == "__main__":
    main()
