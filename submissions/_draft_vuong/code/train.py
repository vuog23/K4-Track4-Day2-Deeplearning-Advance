"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

PSEUDO-CODE: chỉ có khung (cấu hình và quy ước đặt tên file); bạn tự hoàn thiện mọi hàm có
`raise NotImplementedError` và các bước TODO trong `run()`. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import dataclasses
import json
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

try:
    from . import dataset, model as model_lib, losses
except ImportError:
    import dataset
    import model as model_lib
    import losses

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "eval.py").exists()),
                 Path.cwd())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from eval import compute_metrics, save_predictions

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    TODO: random, numpy, torch (CPU và CUDA); cân nhắc cudnn.deterministic/benchmark và
    seed cho worker của DataLoader. Ghi lại trong báo cáo mức độ tái lập bạn đạt được.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups). TODO."""
    model = model_lib.unwrap_model(model)
    return torch.optim.AdamW(model_lib.param_groups(model, cfg.lr_backbone, cfg.lr_head,
                                                   cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55). TODO.

    Cập nhật theo bước (iteration) hoặc theo epoch đều được; ghi rõ bạn chọn gì.
    Gợi ý kiểm tra: vẽ đường LR theo bước để thấy đúng hình warmup + cosine.
    """
    total_steps = max(int(cfg.epochs * steps_per_epoch), 1)
    warmup_steps = min(int(cfg.warmup_epochs * steps_per_epoch), total_steps - 1)
    def scale(step):
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        remain = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / remain, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    TODO:
      - __init__(model, decay): sao chép trọng số
      - update(model): sau mỗi bước tối ưu
      - copy_to(model) hoặc dùng bản sao riêng để đánh giá bằng trọng số EMA
      - lưu ý BatchNorm: buffer (running_mean/var) cũng phải được xử lý hợp lý
    """

    def __init__(self, model, decay: float):
        import copy
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model) -> None:
        with torch.no_grad():
            src = dict(model.named_parameters())
            for name, target in self.module.named_parameters():
                target.lerp_(src[name].detach(), 1.0 - self.decay)
            src_buffers = dict(model.named_buffers())
            for name, target in self.module.named_buffers():
                source = src_buffers[name].detach()
                if target.is_floating_point():
                    target.lerp_(source, 1.0 - self.decay)
                else:
                    target.copy_(source)

    def copy_to(self, model):
        model.load_state_dict(self.module.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}.

    TODO:
      - model.train() (nếu init == "frozen": giữ phần backbone ở eval, xem model.freeze_backbone)
      - nếu cfg.mix: mix_batch rồi mixed_loss (losses.py)
      - AMP (autocast + GradScaler), clip gradient nếu cần, optimizer.step(), scheduler.step()
      - nếu có EMA: ema.update(model)
    """
    model.train()
    base_model = model_lib.unwrap_model(model)
    if getattr(base_model, "_lab_frozen", False):
        for module in base_model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total, seen = 0.0, 0
    scaler_enabled = bool(cfg.amp and device.type == "cuda")
    scaler = scaler or torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    total_batches = len(loader)
    progress_every = max(total_batches // 20, 1)
    for step, (images, targets, _) in enumerate(loader, start=1):
        if step == 1 or step % progress_every == 0 or step == total_batches:
            print(f"{cfg.exp_id} training batch {step}/{total_batches}", flush=True)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        mixed_targets = targets
        if cfg.mix:
            images, mixed_targets = losses.mix_batch(images, targets, cfg.mix_alpha, cfg.mix)
        with torch.cuda.amp.autocast(enabled=scaler_enabled):
            logits = model(images)
            loss = losses.mixed_loss(criterion, logits, mixed_targets)
        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # GradScaler skips optimizer.step() when it finds inf/NaN gradients.
        # Keep the LR schedule in lockstep with successful optimizer updates.
        optimizer_updated = (not scaler_enabled) or scaler.get_scale() >= scale_before
        if scheduler is not None and optimizer_updated:
            scheduler.step()
        if ema is not None and optimizer_updated:
            ema.update(model_lib.unwrap_model(model))
        total += float(loss.detach()) * len(images)
        seen += len(images)
    return {"train_loss": total / max(seen, 1),
            "lr": float(optimizer.param_groups[0]["lr"])}


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.

    TODO: model.eval(), torch.inference_mode(), gom kết quả. Softmax khi cần xác suất.
    """
    model.eval()
    filenames, ys, logits_all, loss_sum, n = [], [], [], 0.0, 0
    with torch.inference_mode():
        for images, targets, names in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            filenames.extend(list(names))
            ys.append(targets.cpu().numpy())
            logits_all.append(logits.float().cpu().numpy())
            loss_sum += float(loss) * len(images)
            n += len(images)
    return filenames, np.concatenate(ys), np.concatenate(logits_all), loss_sum / max(n, 1)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    TODO: tối thiểu loss train/val và macro-F1 val theo epoch; có tiêu đề, nhãn trục, chú thích;
    khuyến khích thêm LR theo bước. Lưu bằng matplotlib với dpi đủ nét để đọc số.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="val")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Cross entropy")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_macro_f1"] for r in history], label="val macro-F1")
    axes[1].plot(epochs, [r["val_top1"] for r in history], label="val top-1")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation metrics", ylim=(0, 1))
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Write checkpoints atomically so a Kaggle session stop cannot truncate them."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    TODO theo thứ tự:
      1. set_seed; tạo thư mục run_dir(cfg); ghi config.json (dataclasses.asdict(cfg))
      2. dataset.load_split + dataset.check_split (dừng nếu vi phạm S1-S6)
      3. dựng train/val loader (test loader chỉ tạo khi cfg.save_test_predictions)
      4. model.build_model, criterion (losses.build_criterion), optimizer, scheduler, scaler, EMA
      5. với mỗi epoch: train_one_epoch -> evaluate(val) -> ghi history (loss, macro-F1 val, lr...)
         và lưu checkpoint tốt nhất theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn)
      6. cuối: nạp checkpoint tốt nhất, lưu val logits và eval.save_predictions(pred_path(cfg, "val"), ...)
      7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): đánh giá test đúng MỘT lần,
         lưu logits và eval.save_predictions(pred_path(cfg, "test"), ...)
      8. ghi history.csv, plot_curves(...), trả về dict tóm tắt
         (best_epoch, macro-F1 val, thời gian train mỗi epoch, số tham số, GMAC)
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    if cfg.fold != 0:
        raise ValueError("Bài lab bắt buộc dùng fold 0; không đổi split qua seed")
    set_seed(cfg.seed)
    run_path = REPO_ROOT / run_dir(cfg)
    run_path.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(cfg.pred_dir)
    if not pred_dir.is_absolute(): pred_dir = REPO_ROOT / pred_dir
    pred_dir.mkdir(parents=True, exist_ok=True)
    (run_path / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2), encoding="utf-8")
    labels_dir, images_dir = Path(cfg.labels_dir), Path(cfg.images_dir)
    if not labels_dir.is_absolute(): labels_dir = REPO_ROOT / labels_dir
    if not images_dir.is_absolute(): images_dir = REPO_ROOT / images_dir
    train_df, val_df, test_df = dataset.load_split(labels_dir, cfg.fold)
    dataset.check_split(train_df, val_df, test_df, images_dir)
    train_loader = dataset.make_loader(train_df, images_dir,
        dataset.build_transforms(True, cfg.img_size, cfg.aug), cfg.batch_size, True,
        cfg.sampler, cfg.num_workers)
    val_loader = dataset.make_loader(val_df, images_dir,
        dataset.build_transforms(False, cfg.img_size), cfg.batch_size, False, None, cfg.num_workers)
    test_loader = None
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(test_df, images_dir,
            dataset.build_transforms(False, cfg.img_size), cfg.batch_size, False, None, cfg.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_net = model_lib.build_model(cfg.backbone, pretrained=True, drop_rate=cfg.drop_rate,
                                     init=cfg.init).to(device)
    net = model_lib.use_data_parallel(base_net, device)
    try:
        import timm
        timm_version = timm.__version__
    except ImportError:
        timm_version = "unavailable"
    env_record = {"config": dataclasses.asdict(cfg), "torch": torch.__version__, "timm": timm_version,
                  "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                  "data_parallel": isinstance(net, nn.DataParallel),
                  "parallel_devices": (list(range(torch.cuda.device_count()))
                                       if isinstance(net, nn.DataParallel) else []),
                  "pretrained_cfg": getattr(base_net, "_lab_pretrained_cfg", {})}
    (run_path / "config.json").write_text(json.dumps(env_record, indent=2, default=str), encoding="utf-8")
    criterion = losses.build_criterion(cfg.loss,
        smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        weight=(losses.class_weights(np.bincount(train_df.Label.astype(int), minlength=9),
                    (0.0 if cfg.class_weight_beta is None else cfg.class_weight_beta)).to(device)
                 if cfg.loss == "ce_weighted" else None))
    optimizer = build_optimizer(base_net, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    ema = EMA(base_net, cfg.ema_decay) if cfg.ema_decay is not None else None
    best_f1, best_epoch, history = -1.0, None, []
    epoch_seconds = []
    best_ckpt = run_path / "best.pt"
    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        train_stats = train_one_epoch(net, train_loader, criterion, optimizer, scheduler,
                                      scaler, cfg, device, ema)
        eval_model = ema.module if ema is not None else net
        _, y_val, val_logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
        metrics = compute_metrics(y_val, val_probs.argmax(1), val_probs)
        record = {"epoch": epoch, **train_stats, "val_loss": val_loss,
                  "val_macro_f1": float(metrics["macro_f1"]), "val_top1": float(metrics["top1"]),
                  "val_ece": float(metrics["ece"])}
        history.append(record)
        epoch_seconds.append(time.perf_counter() - start)
        print(f"{cfg.exp_id} epoch {epoch}/{cfg.epochs}: train_loss={record['train_loss']:.4f} "
              f"val_loss={val_loss:.4f} val_macro_f1={record['val_macro_f1']:.4f}")
        if metrics["macro_f1"] > best_f1:  # strict > preserves the earlier epoch on ties
            best_f1, best_epoch = float(metrics["macro_f1"]), epoch
            _atomic_torch_save({"model": model_lib.unwrap_model(eval_model).state_dict(), "epoch": epoch,
                                "macro_f1": best_f1, "config": dataclasses.asdict(cfg)}, best_ckpt)
        latest_payload = {"model": model_lib.unwrap_model(net).state_dict(), "epoch": epoch,
                          "best_epoch": best_epoch, "best_macro_f1": best_f1,
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "scaler": scaler.state_dict(), "ema": ema.module.state_dict() if ema else None,
                          "config": dataclasses.asdict(cfg)}
        _atomic_torch_save(latest_payload, run_path / "latest.pt")
        pd.DataFrame(history).to_csv(run_path / "history.csv", index=False)
        plot_curves(history, REPO_ROOT / "curves" / f"{cfg.exp_id}_seed{cfg.seed}.png",
                    f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}")
    try:
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_ckpt, map_location=device)
    base_net.load_state_dict(checkpoint["model"])
    net.eval()
    _, y_val, val_logits, _ = evaluate(net, val_loader, criterion, device)
    np.save(run_path / "val_logits.npy", val_logits)
    np.save(run_path / "val_labels.npy", y_val)
    val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
    best_metrics = compute_metrics(y_val, val_probs.argmax(1), val_probs)
    save_predictions(pred_dir / pred_path(cfg, "val").name, val_df.Filename.astype(str).tolist(), y_val, val_probs)
    if test_loader is not None:
        _, y_test, test_logits, _ = evaluate(net, test_loader, criterion, device)
        np.save(run_path / "test_logits.npy", test_logits)
        np.save(run_path / "test_labels.npy", y_test)
        test_probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
        save_predictions(pred_dir / pred_path(cfg, "test").name, test_df.Filename.astype(str).tolist(), y_test, test_probs)
    pd.DataFrame(history).to_csv(run_path / "history.csv", index=False)
    plot_curves(history, REPO_ROOT / "curves" / f"{cfg.exp_id}_seed{cfg.seed}.png",
                f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}")
    return {"exp_id": cfg.exp_id, "backbone": cfg.backbone, "seed": cfg.seed, "best_epoch": best_epoch,
            "macro_f1_val": float(best_metrics["macro_f1"]), "top1_val": float(best_metrics["top1"]),
            "f1_chinee_apple_val": float(best_metrics["f1"][0]),
            "f1_snake_weed_val": float(best_metrics["f1"][7]),
            "params_m": model_lib.count_params(base_net), "gmac": model_lib.count_gmacs(base_net, cfg.img_size),
            "seconds_per_epoch": float(np.mean(epoch_seconds)),
            "pretrained_cfg": getattr(base_net, "_lab_pretrained_cfg", {}),
            "data_parallel": isinstance(net, nn.DataParallel),
            "num_devices": torch.cuda.device_count() if device.type == "cuda" else 0}


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config.

    TODO: tách key/value, báo lỗi rõ nếu key không có trong Config, ép int/float/bool/None theo kiểu field.
    """
    int_fields = {"seed", "fold", "img_size", "epochs", "batch_size", "num_workers"}
    float_fields = {"drop_rate", "mix_alpha", "label_smoothing", "focal_gamma", "lr_backbone",
                    "lr_head", "weight_decay", "warmup_epochs"}
    optional_floats = {"class_weight_beta", "ema_decay"}
    bool_fields = {"amp", "save_test_predictions"}
    string_fields = {f.name for f in dataclasses.fields(Config)} - int_fields - float_fields - optional_floats - bool_fields
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Cần KEY=VALUE, nhận {pair!r}")
        key, raw = pair.split("=", 1)
        if key not in {f.name for f in dataclasses.fields(Config)}:
            raise ValueError(f"Config không có trường {key!r}")
        low = raw.strip().lower()
        if low in {"none", "null"}:
            value = None
        elif key in int_fields:
            value = int(raw)
        elif key in float_fields | optional_floats:
            value = float(raw)
        elif key in bool_fields:
            if low not in {"true", "false", "1", "0"}:
                raise ValueError(f"Boolean không hợp lệ: {raw}")
            value = low in {"true", "1"}
        else:
            value = raw
        result[key] = value
    return result


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

    TODO: argparse nhận `--set KEY=VALUE ...`, dựng Config qua parse_overrides, gọi run(cfg), in kết quả.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(run(cfg))


if __name__ == "__main__":
    main()
