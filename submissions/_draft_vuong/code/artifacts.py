"""Build the Step 5 workbook and report from saved experiment/eval artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _markdown_table(frame: pd.DataFrame) -> str:
    frame = frame.fillna("").astype(str)
    headers = list(frame.columns)
    if not headers:
        return "(no rows yet)"
    rows = frame.values.tolist()
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    rule = "| " + " | ".join("-" * max(width, 3) for width in widths) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
            for row in rows]
    return "\n".join([line, rule, *body])


def _save_error_montage(pred_dir: Path, eval_dir: Path, labels_dir: Path,
                        images_dir: Path) -> Path | None:
    """Create a reviewable montage of high-confidence F01 test errors, if available."""
    prediction_files = sorted(pred_dir.glob("F01_seed*_test.csv"))
    if not prediction_files:
        return None
    predictions = pd.concat([pd.read_csv(p) for p in prediction_files], ignore_index=True)
    labels = pd.read_csv(labels_dir / "labels.csv")
    names = (labels[["Label", "Species"]].drop_duplicates().sort_values("Label")
             ["Species"].astype(str).tolist())
    errors = predictions[predictions.y_true != predictions.y_pred].copy()
    if errors.empty:
        return None
    probs = errors[[f"p{i}" for i in range(len(names))]].to_numpy()
    errors["confidence"] = probs.max(axis=1)
    errors = errors.sort_values("confidence", ascending=False).drop_duplicates("Filename").head(18)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    tile_w, tile_h, cols = 240, 285, 3
    canvas = Image.new("RGB", (tile_w * cols, tile_h * ((len(errors) + cols - 1) // cols)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(errors.itertuples(index=False)):
        image_path = images_dir / str(row.Filename)
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((tile_w - 12, tile_h - 54))
        x, y = (idx % cols) * tile_w, (idx // cols) * tile_h
        canvas.paste(image, (x + (tile_w - image.width) // 2, y))
        draw.text((x + 5, y + tile_h - 48),
                  f"true: {names[int(row.y_true)]}\npred: {names[int(row.y_pred)]} ({row.confidence:.2f})",
                  fill="black")
    out = eval_dir / "F01_misclassified_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    errors.to_csv(eval_dir / "F01_misclassified.csv", index=False)
    return out


def build_results(results_dir="results", eval_dir="eval_out", out_xlsx="results.xlsx",
                  out_report="report.md", pred_dir="predictions", labels_dir="data/labels",
                  images_dir="data") -> tuple[Path, Path]:
    results_dir, eval_dir = Path(results_dir), Path(eval_dir)
    backbones = _read_csv(results_dir / "backbones.csv")
    training = _read_csv(results_dir / "training.csv")
    inference = _read_csv(results_dir / "inference.csv")
    inference = inference.rename(columns={c: f"{c}_ms" for c in ("p50", "p95", "p99")
                                         if c in inference.columns})

    final_frames, per_class_frames = [], []
    final_runs = _read_csv(results_dir / "final_runs.csv")
    for tag in ("F01", "T00"):
        per_seed = _read_csv(eval_dir / f"{tag}_per_seed.csv")
        if not per_seed.empty:
            per_seed.insert(0, "tag", tag)
            matching_runs = final_runs[final_runs.exp_id == tag] if not final_runs.empty else pd.DataFrame()
            if not matching_runs.empty:
                keep = [c for c in ("seed", "backbone", "macro_f1_val", "top1_val", "temperature_val",
                                    "inference_method", "recipe", "recipe_config", "epochs", "img_size")
                        if c in matching_runs]
                per_seed = per_seed.merge(matching_runs[keep].drop_duplicates("seed"),
                                          on="seed", how="left")
            final_frames.append(per_seed)
        per_class = _read_csv(eval_dir / f"{tag}_per_class.csv")
        if not per_class.empty:
            per_class.insert(0, "tag", tag)
            per_class_frames.append(per_class)
    final = pd.concat(final_frames, ignore_index=True) if final_frames else pd.DataFrame()
    if not final.empty:
        summary_rows = []
        for tag in ("F01", "T00"):
            summary_path = eval_dir / f"{tag}_summary.json"
            if not summary_path.exists():
                continue
            values = json.loads(summary_path.read_text(encoding="utf-8"))
            row = {"tag": tag, "file": "mean ± std", "seed": "mean ± std",
                   "n": "per-seed full test"}
            for key in ("top1", "macro_f1", "balanced_acc", "ece", "nll"):
                if key in values:
                    row[key] = f"{values[key]['mean']:.4f} ± {values[key]['std']:.4f}"
            matching = final_runs[(final_runs.exp_id == tag)] if not final_runs.empty else pd.DataFrame()
            if not matching.empty:
                row["backbone"] = str(matching.iloc[0].get("backbone", ""))
                if "macro_f1_val" in matching:
                    row["macro_f1_val"] = (f"{matching.macro_f1_val.mean():.4f} ± "
                                            f"{matching.macro_f1_val.std(ddof=1):.4f}")
                row["inference_method"] = matching.iloc[0].get("inference_method", "")
                row["recipe_config"] = matching.iloc[0].get("recipe_config", "")
            summary_rows.append(row)
        if summary_rows:
            final = pd.concat([final, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)
    per_class = pd.concat(per_class_frames, ignore_index=True) if per_class_frames else pd.DataFrame()

    latency_cols = [c for c in ("exp_id", "method", "gpu", "dtype", "batch", "img_size",
                                 "fused_bn", "p50", "p95", "p99", "images_per_s") if c in inference.columns]
    latency = inference[latency_cols].copy() if latency_cols else pd.DataFrame()
    latency = latency.rename(columns={c: f"{c}_ms" for c in ("p50", "p95", "p99") if c in latency})
    summary_rows = []
    if not backbones.empty:
        summary_rows.append(backbones.sort_values("macro_f1_val", ascending=False).head(10)
                            .assign(section="Backbones"))
    if not training.empty:
        summary_rows.append(training.sort_values("macro_f1_val", ascending=False).head(10)
                            .assign(section="Training"))
    if not inference.empty and "macro_f1_val" in inference:
        summary_rows.append(inference.sort_values("macro_f1_val", ascending=False).head(10)
                            .assign(section="Inference"))
    summary = pd.concat(summary_rows, ignore_index=True, sort=False) if summary_rows else pd.DataFrame()

    out_xlsx = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for name, frame in (("Backbones", backbones), ("Training", training),
                            ("Inference", inference), ("Final", final),
                            ("PerClass", per_class), ("Latency", latency),
                            ("Summary", summary)):
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "A2"
            if ws.max_row > 1 and ws.max_column > 0:
                ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                width = min(max(max(len(str(c.value or "")) for c in col) + 2, 10), 42)
                ws.column_dimensions[col[0].column_letter].width = width

    report = ["# DeepWeeds Lab Day 2 report", "",
              "> Generated from this notebook's saved logs and predictions. Replace the bracketed "
              "analysis prompts after reviewing the measured results; do not add unsupported claims.", "",
              "## 1. Summary", "",
              "- Task: classify the nine DeepWeeds classes using author-provided fold 0.",
              "- Selection: backbone, training recipe, inference method, and temperature are selected/fitted on validation only.",
              "- Final: F01 is the validation-selected recipe with one-view inference and temperature scaling fitted on validation; T00/I00 is the uncalibrated baseline.",
              "- Final test numbers below are generated from the saved once-per-seed predictions. Add the selected configuration and its mean ± sample std after results are available.", "",
              "## 2. Data and setup", "",
              "DeepWeeds contains 17,509 RGB images (256×256) across eight weed species and Negative. Fold-0 CSVs are used unchanged; train/val/test remain disjoint and validation/test are never included in training.", ""]
    split_path = Path("eda/split_check.json")
    counts_path = Path("eda/eda_class_counts.csv")
    smoke_path = Path("eda/pipeline_smoke.json")
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        report += [f"Split sizes: `{split.get('n', {})}`; overlap counts: `{split.get('overlap', {})}`; union: `{split.get('union', 'unknown')}`.",
                   f"Per-class counts by split: `{split.get('per_class', {})}`.", ""]
    if counts_path.exists():
        counts = pd.read_csv(counts_path)
        report += ["Fold-0 class-count comparison with the paper's Table 1:", "", _markdown_table(counts), ""]
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        report += [f"Pipeline smoke check: initial 9-way CE `{smoke.get('initial_loss')}`, "
                   f"one-batch overfit loss `{smoke.get('final_overfit_loss')}`, device `{smoke.get('device')}`.", ""]
    report += ["EDA figures: `eda/eda_class_distribution.png`, `eda/eda_examples.png`, and `eda/eda_augmented_examples.png`.",
               "[Add observations about class imbalance, visual ambiguity, and augmentation artifacts after inspecting those figures.]", "",
               "## 3. Backbone comparison", ""]
    if not backbones.empty:
        top = backbones.sort_values("macro_f1_val", ascending=False)
        report.append(_markdown_table(top[[c for c in ("exp_id", "backbone", "macro_f1_val", "top1_val",
                                                       "params_m", "gmac", "seconds_per_epoch",
                                                       "latency_batch1_p95_ms") if c in top.columns]]))
        report.append("")
        row = top.iloc[0]
        report.append(f"Highest validation macro-F1 in this table: **{row['backbone']}** "
                      f"(macro-F1 {row['macro_f1_val']:.4f}). Compare parameter count, GMAC, and latency before choosing the final model.")
    else:
        report.append("Backbone results are not available yet.")
    report += ["", "## 4. Training recipe comparison", ""]
    if not training.empty:
        top = training.sort_values("macro_f1_val", ascending=False)
        report.append(_markdown_table(top[[c for c in ("exp_id", "backbone", "changed_from_T00", "macro_f1_val",
                                                       "top1_val", "f1_chinee_apple_val",
                                                       "f1_snake_weed_val") if c in top.columns]]))
        report.append("")
        report.append(f"Highest one-seed validation recipe: **{top.iloc[0]['exp_id']}**. "
                      "The table records one changed factor per ablation; T08 is the explicitly combined interaction test.")
        report.append("[Discuss per-axis Δ versus T00 and whether each effect is larger than seed noise; these screening runs use one seed, so treat small differences as inconclusive.]")
    else:
        report.append("Training ablation results are not available yet.")
    report += ["", "## 5. Inference and latency", ""]
    if not inference.empty:
        cols = [c for c in ("exp_id", "method", "macro_f1_val", "top1_val", "ece_val", "p50", "p95", "p99")
                if c in inference.columns]
        report.append(_markdown_table(inference[cols]))
        report.append("")
        report.append("Temperature is fitted by minimizing validation NLL. F01 test probabilities are temperature-scaled from saved test logits; the accompanying `F01uncal_*_test.csv` files preserve the same test pass before calibration. Latency reports use warmup and synchronized timing where CUDA is available.")
        report.append("[Compare macro-F1/ECE against p95 latency. State which methods suit offline processing and which fit a real-time budget; calibration does not change argmax accuracy.]")
    else:
        report.append("Inference results are not available yet.")
    report += ["", "## 6. Final configuration and test evaluation", ""]
    summaries = []
    for tag in ("F01", "T00"):
        path = eval_dir / f"{tag}_summary.json"
        if path.exists():
            summaries.append((tag, json.loads(path.read_text(encoding="utf-8"))))
    if summaries:
        for tag, values in summaries:
            report.append(f"### {tag}")
            report.append("")
            report.append(f"Seeds: {values.get('seeds', [])}")
            report.append("")
            report.append("| Metric | Mean | Std |\n|---|---:|---:|")
            for key in ("top1", "macro_f1", "balanced_acc", "ece", "nll"):
                if key in values:
                    report.append(f"| {key} | {values[key]['mean']:.4f} | {values[key]['std']:.4f} |")
            report.append("")
    else:
        report.append("Final test evaluation has not been run or exported.")
    report += ["The final model is retrained from scratch with seeds 0, 1, and 2 using the selected training recipe; T00 is retrained with matching seeds. Test is evaluated once per run after the choices are frozen. Prediction CSVs are the source of truth for all reported metrics.", "",
               "### Per-class performance", ""]
    if not per_class.empty:
        report.append(_markdown_table(per_class[per_class.tag == "F01"]))
        report.append("")
    report += ["### Confusion matrix and error review", ""]
    confusion_path = eval_dir / "F01_confusion_sum.csv"
    if confusion_path.exists():
        cm = pd.read_csv(confusion_path, index_col=0)
        if not cm.empty:
            names = [str(x).removeprefix("pred_") for x in cm.columns]
            cm.columns = names
            cm.index = [str(x).removeprefix("true_") for x in cm.index]
            errors = []
            for i, true_name in enumerate(cm.index):
                for j, predicted_name in enumerate(cm.columns):
                    if i != j and int(cm.iloc[i, j]) > 0:
                        errors.append((int(cm.iloc[i, j]), true_name, predicted_name))
            errors.sort(reverse=True)
            report.append("Most frequent off-diagonal confusions across final seeds (rows=true, columns=predicted):")
            report.append("")
            report.append("| True class | Predicted class | Count |\n|---|---|---:|")
            report.extend(f"| {truth} | {pred} | {count} |" for count, truth, pred in errors[:10])
            report.append("")
    else:
        report.append("Run eval.py score to export the final confusion matrix and inspect the largest class confusions.")
        report.append("")
    montage = _save_error_montage(Path(pred_dir), Path(eval_dir), Path(labels_dir), Path(images_dir))
    if montage:
        report += [f"High-confidence test errors for visual review: `{montage.as_posix()}`; row-level file: `{(Path(eval_dir) / 'F01_misclassified.csv').as_posix()}`.",
                   "[Inspect the montage, especially Chinee apple ↔ Snake weed, and write a brief evidence-based explanation of recurring visual confusions.]", ""]
    report += ["## 7. Conclusions and deployment recommendation", "",
               "[Name the validation-selected backbone, training factors with the largest measured contribution, and the inference choice. Compare F01 against T00 using the same seeds and the larger observed std; say ‘not distinguishable’ when Δ is within noise. For the 30–100 ms/ frame robot budget, cite a measured batch-1 p95 and corresponding macro-F1, or state that no tested configuration met it.]", "",
               "## 8. Limitations and next steps", "",
               "Fold 0 is a random image split rather than a location-held-out split. Results from one fold "
               "and three final seeds may not transfer to new field sites. The paper's benchmark used a different training recipe and many more epochs, so its accuracy is contextual rather than a direct target. Any reduced epochs, image resolution, or batch size should be documented. The Summary sheet retains the validation ranking; test metrics are reported separately.",
               "[List any experiments skipped or failed because of the Kaggle time/GPU budget, distribution-shift risks, and the next experiment you would run.]", "",
               "## 9. Appendix: reproducibility", "",
               "- Notebook: attach the Kaggle notebook URL in the submission README after publishing it.",
               "- Split: author-provided fold 0; no seed-dependent split changes.",
               "- Main run settings: see each `runs/<exp_id>/seed<k>/config.json` and `history.csv`.",
               "- Outputs: `results.xlsx`, `curves/`, `predictions/`, and `eval_out/`.",
               "- This report contains no hard-coded experimental metric; all numeric result tables are read from the saved CSV/JSON outputs.", ""]
    out_report = Path(out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(report), encoding="utf-8")
    return out_xlsx, out_report
