"""Validation-only inference comparison and synchronized latency measurements."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from . import dataset, model as model_lib, inference, benchmark
    from .train import run_dir
except ImportError:
    import dataset, model as model_lib, inference, benchmark
    from train import run_dir

from eval import compute_metrics


def _make_model(exp_id: str, backbone: str, device, run_root="runs"):
    ckpt_path = Path(run_root) / exp_id / "seed0" / "best.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = model_lib.build_model(backbone, pretrained=False, init="finetune")
    net.load_state_dict(checkpoint["model"])
    return model_lib.use_data_parallel(net, device).eval()


def _score(name, logits, y_true):
    probs = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1).numpy()
    metrics = compute_metrics(y_true, probs.argmax(1), probs)
    return {"exp_id": name, "macro_f1_val": metrics["macro_f1"], "top1_val": metrics["top1"],
            "ece_val": metrics["ece"], "nll_val": metrics["nll"]}, probs


def _five_crop_logits(net, loader, device, crop=224):
    view_batches, labels = [[] for _ in range(5)], []
    net.eval()
    with torch.inference_mode():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            if x.shape[-1] < crop or x.shape[-2] < crop:
                raise ValueError("Input nhỏ hơn crop")
            views = inference.views_multicrop(x, crop)
            all_logits = [net(v).float().cpu().numpy() for v in views]
            for view_id, logits in enumerate(all_logits):
                view_batches[view_id].append(logits)
            labels.append(y.numpy())
    # returns K arrays, one per view, and labels
    view_arrays = [np.concatenate(batches, axis=0) for batches in view_batches]
    return view_arrays, np.concatenate(labels)


def _measure_call(fn, device):
    def timed():
        with torch.inference_mode():
            fn()
    return benchmark.bench(timed, warmup=10, iters=50,
                           sync=torch.cuda.synchronize if device.type == "cuda" else None)


def run_inference_study(exp_id: str, backbone: str, out_csv="results/inference.csv",
                        batch_size=8, img_size=224, run_root="runs",
                        images_dir="data", labels_dir="data/labels"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_df, _ = dataset.load_split(labels_dir, fold=0)
    val_transform = dataset.build_transforms(False, img_size)
    val_loader = dataset.make_loader(val_df, images_dir, val_transform, batch_size,
                                     False, num_workers=2)
    model = _make_model(exp_id, backbone, device, run_root)
    rows = []
    probe = torch.randn(1, 3, img_size, img_size, device=device)
    latency_one = _measure_call(lambda: model(probe), device)
    latency_flip = _measure_call(lambda: (model(probe), model(torch.flip(probe, (-1,)))), device)

    # I00: existing deterministic one-view logits from the selected checkpoint.
    logits0 = np.load(Path(run_root) / exp_id / "seed0" / "val_logits.npy")
    y_true = np.load(Path(run_root) / exp_id / "seed0" / "val_labels.npy")
    row0, probs0 = _score("I00", logits0, y_true)
    row0["method"] = "one view; forward only"
    row0.update({f"p{k}": latency_one[k] for k in ("p50", "p95", "p99")})
    rows.append(row0)

    # I01/I03: horizontal flip and compare probability averaging vs logit averaging.
    _, _, flipped = inference.predict_logits(model, val_loader, device, inference.view_hflip)
    row1, probs_flip = _score("I01_hflip", np.stack([logits0, flipped]).mean(0), y_true)
    row1["method"] = "one view + horizontal flip; logit mean"
    row1.update({f"p{k}": latency_flip[k] for k in ("p50", "p95", "p99")})
    rows.append(row1)
    prob_mean = inference.aggregate_views([logits0, flipped], space="prob")
    logit_mean = inference.aggregate_views([logits0, flipped], space="logit")
    metric_prob = compute_metrics(y_true, prob_mean.argmax(1), prob_mean)
    metric_logit = compute_metrics(y_true, logit_mean.argmax(1), logit_mean)
    rows.append({"exp_id": "I03_prob_mean", "method": "probability average", "macro_f1_val": metric_prob["macro_f1"],
                 "top1_val": metric_prob["top1"], "ece_val": metric_prob["ece"], "nll_val": metric_prob["nll"],
                 **{f"p{k}": latency_flip[k] for k in ("p50", "p95", "p99")}})
    rows.append({"exp_id": "I03_logit_mean", "method": "logit average", "macro_f1_val": metric_logit["macro_f1"],
                 "top1_val": metric_logit["top1"], "ece_val": metric_logit["ece"], "nll_val": metric_logit["nll"],
                 **{f"p{k}": latency_flip[k] for k in ("p50", "p95", "p99")}})

    # I02: five-crop TTA from a 256px resized image, no random augmentation.
    from dataset import DeepWeedsDataset, IMAGENET_MEAN, IMAGENET_STD
    raw_transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor(),
                                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    raw_ds = DeepWeedsDataset(val_df, images_dir, raw_transform)
    raw_loader = DataLoader(raw_ds, batch_size=batch_size, shuffle=False, num_workers=2,
                            pin_memory=torch.cuda.is_available())
    view_logits, y_crop = _five_crop_logits(model, raw_loader, device, crop=img_size)
    crop_probs = inference.aggregate_views(view_logits, "prob")
    crop_metric = compute_metrics(y_crop, crop_probs.argmax(1), crop_probs)
    probe256 = torch.randn(1, 3, 256, 256, device=device)
    latency_crop = _measure_call(lambda: [model(v) for v in inference.views_multicrop(probe256, img_size)], device)
    rows.append({"exp_id": "I02_5crop", "method": "five crop probability mean", "macro_f1_val": crop_metric["macro_f1"],
                 "top1_val": crop_metric["top1"], "ece_val": crop_metric["ece"], "nll_val": crop_metric["nll"],
                 **{f"p{k}": latency_crop[k] for k in ("p50", "p95", "p99")}})

    # I04: check 256px single-view inference; some transformer models have fixed input sizes.
    try:
        resized = inference.views_multiscale(next(iter(val_loader))[0].to(device), [256])[0]
        with torch.inference_mode():
            model(resized)
        # Full validation pass with 256 input.
        resized_transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor(),
                                                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        resize_loader = DataLoader(DeepWeedsDataset(val_df, images_dir, resized_transform),
                                   batch_size=batch_size, shuffle=False, num_workers=2,
                                   pin_memory=torch.cuda.is_available())
        _, y_resize, logits_resize = inference.predict_logits(model, resize_loader, device)
        row_resize, _ = _score("I04_256px", logits_resize, y_resize)
        row_resize["method"] = "single view at 256px"
        latency_resize = _measure_call(lambda: model(probe256), device)
        row_resize.update({f"p{k}": latency_resize[k] for k in ("p50", "p95", "p99")})
        rows.append(row_resize)
    except (RuntimeError, ValueError) as exc:
        rows.append({"exp_id": "I04_256px", "method": f"not supported: {exc}"})

    # I07: fit one scalar T on validation only; save both calibrated and uncalibrated metrics.
    temperature = inference.fit_temperature(logits0, y_true)
    calibrated = inference.apply_temperature(logits0, temperature)
    cal_metric = compute_metrics(y_true, calibrated.argmax(1), calibrated)
    rows.append({"exp_id": "I07_temperature", "method": "temperature scaling fitted on val",
                 "temperature": temperature, "macro_f1_val": cal_metric["macro_f1"],
                 "top1_val": cal_metric["top1"], "ece_val": cal_metric["ece"], "nll_val": cal_metric["nll"],
                 **{f"p{k}": latency_one[k] for k in ("p50", "p95", "p99")}})

    # I08: report batch-1 and batch-32 FP32/AMP, then folded BN when supported.
    for dt in ("fp32", "amp"):
        for batch in (1, 32):
            try:
                result = benchmark.latency_report(model, batch, img_size, dt, str(device), warmup=10, iters=50)
                rows.append({"exp_id": f"I08_{dt}_b{batch}", "method": "latency; forward only", **result})
            except RuntimeError as exc:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                rows.append({"exp_id": f"I08_{dt}_b{batch}", "method": f"not measured: {exc}"})
    fused = inference.fuse_conv_bn(model)
    if getattr(fused, "_lab_fused_conv_bn_count", 0):
        result = benchmark.latency_report(fused, 1, img_size, "fp32", str(device), warmup=10, iters=50)
        rows.append({"exp_id": "I08_fused_bn_b1", "method": "latency after Conv-BN fusion; forward only", **result})

    baseline_latency = next((r.get("p50") for r in rows if r.get("exp_id") == "I00"), None)
    for row in rows:
        row.setdefault("gpu", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
        row.setdefault("dtype", "fp32")
        row.setdefault("batch", 1)
        row.setdefault("img_size", 256 if row.get("exp_id") == "I04_256px" else img_size)
        row.setdefault("fused_bn", row.get("exp_id") == "I08_fused_bn_b1")
        row.setdefault("k_views", 1)
        if "5crop" in str(row.get("exp_id", "")):
            row["k_views"] = 5
        elif "hflip" in str(row.get("exp_id", "")) or str(row.get("exp_id", "")).startswith("I03_"):
            row["k_views"] = 2
        if row.get("p50") is not None:
            row.setdefault("images_per_s", 1000.0 / max(float(row["p50"]), 1e-9))
            if baseline_latency:
                row.setdefault("relative_cost_vs_I00", float(row["p50"]) / float(baseline_latency))
        else:
            row.setdefault("images_per_s", np.nan)
            row.setdefault("relative_cost_vs_I00", np.nan)
    frame = pd.DataFrame(rows)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    scored = pd.DataFrame([r for r in rows if "p95" in r and "macro_f1_val" in r])
    if not scored.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(scored["p95"], scored["macro_f1_val"])
        for _, r in scored.iterrows():
            ax.annotate(r["exp_id"], (r["p95"], r["macro_f1_val"]))
        ax.set(xlabel="p95 latency (ms)", ylabel="Validation macro-F1",
               title="Accuracy-latency tradeoff")
        fig.tight_layout()
        fig.savefig("curves/inference_accuracy_latency.png", dpi=160)
        plt.close(fig)
    return frame, temperature


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-id", default="T00")
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--out", default="results/inference.csv")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--img-size", type=int, default=224)
    a = p.parse_args()
    frame, temperature = run_inference_study(a.exp_id, a.backbone, a.out, a.batch_size, a.img_size)
    print(frame.to_string(index=False))
    print(f"temperature fitted on val: {temperature:.5f}")


if __name__ == "__main__":
    main()
