"""Step 0: fold checks, EDA figures, and a small model-pipeline smoke check."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from . import dataset
except ImportError:
    import dataset


def make_eda(images_dir: str, labels_dir: str, out_dir: str, samples_per_class: int = 3) -> dict:
    train_df, val_df, test_df = dataset.load_split(labels_dir, fold=0)
    report = dataset.check_split(train_df, val_df, test_df, images_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print true counts and compare against the paper's Table 1 counts.
    full = pd.concat([train_df, val_df, test_df], ignore_index=True)
    total_counts = full.Label.astype(int).value_counts().reindex(range(9), fill_value=0)
    reference = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]
    table = pd.DataFrame({"class": dataset.CLASS_NAMES, "fold0_count": total_counts.values,
                          "paper_table1": reference})
    table.to_csv(out / "eda_class_counts.csv", index=False)
    print(table.to_string(index=False))
    print("largest/smallest class ratio:", round(total_counts.max() / total_counts.min(), 3))

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.25
    for offset, (name, frame) in enumerate((("train", train_df), ("val", val_df), ("test", test_df))):
        counts = frame.Label.astype(int).value_counts().reindex(range(9), fill_value=0)
        ax.bar(np.arange(9) + (offset - 1) * width, counts.values, width, label=name)
    ax.set_xticks(np.arange(9), dataset.CLASS_NAMES, rotation=35, ha="right")
    ax.set_ylabel("Images")
    ax.set_title("DeepWeeds fold 0 class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "eda_class_distribution.png", dpi=160)
    plt.close(fig)

    rng = np.random.default_rng(0)
    chosen = []
    for label in range(9):
        candidates = full[full.Label.astype(int) == label]
        if len(candidates) < samples_per_class:
            raise ValueError(f"Class {dataset.CLASS_NAMES[label]} has too few samples")
        idx = rng.choice(len(candidates), samples_per_class, replace=False)
        chosen.extend((label, candidates.iloc[i].Filename) for i in idx)
    fig, axes = plt.subplots(9, samples_per_class, figsize=(3 * samples_per_class, 18))
    for ax, (label, filename) in zip(axes.flat, chosen):
        with Image.open(Path(images_dir) / filename) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(dataset.CLASS_NAMES[label])
        ax.axis("off")
    fig.suptitle("Three examples per class (fold 0)")
    fig.tight_layout()
    fig.savefig(out / "eda_examples.png", dpi=150)
    plt.close(fig)

    aug_transform = dataset.build_transforms(True, img_size=224, aug="basic")
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    mean = np.asarray(dataset.IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.asarray(dataset.IMAGENET_STD).reshape(1, 1, 3)
    for label in range(9):
        filename = next(f for lab, f in chosen if lab == label)
        with Image.open(Path(images_dir) / filename) as im:
            tensor = aug_transform(im.convert("RGB"))
        rgb = (tensor.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        ax = axes.flat[label]
        ax.imshow(rgb)
        ax.set_title(dataset.CLASS_NAMES[label])
        ax.axis("off")
    fig.suptitle("Training augmentation examples (unnormalized)")
    fig.tight_layout()
    fig.savefig(out / "eda_augmented_examples.png", dpi=150)
    plt.close(fig)
    return report


def pipeline_smoke(images_dir: str, labels_dir: str, device: str | None = None) -> dict:
    """Check initial 9-way CE and whether a small batch can be memorized."""
    from model import build_model, use_data_parallel
    train_df, _, _ = dataset.load_split(labels_dir, fold=0)
    loader = dataset.make_loader(train_df.iloc[:4], images_dir,
        dataset.build_transforms(False, img_size=96), 4, False, num_workers=0)
    images, labels, _ = next(iter(loader))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = use_data_parallel(build_model("resnet18", pretrained=False, num_classes=9,
                                        init="scratch"), device)
    images, labels = images.to(device), labels.to(device)
    net.eval()
    with torch.no_grad():
        initial = float(F.cross_entropy(net(images), labels))
    if not 1.0 <= initial <= 3.5:
        raise AssertionError(f"Initial CE {initial:.3f} is far from ln(9)=2.197; inspect head/model.")
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3)
    final = initial
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(net(images), labels)
        loss.backward()
        optimizer.step()
        final = float(loss.detach())
        if final < 0.02:
            break
    if final > 0.2:
        raise AssertionError(f"Could not overfit one batch: final loss={final:.4f}")
    return {"initial_loss": initial, "final_overfit_loss": final, "device": str(device)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="data")
    p.add_argument("--labels", default="data/labels")
    p.add_argument("--out", default="eda")
    p.add_argument("--skip-smoke", action="store_true")
    args = p.parse_args()
    report = make_eda(args.images, args.labels, args.out)
    print("EDA report:", report)
    if not args.skip_smoke:
        print("Pipeline smoke:", pipeline_smoke(args.images, args.labels))


if __name__ == "__main__":
    main()
