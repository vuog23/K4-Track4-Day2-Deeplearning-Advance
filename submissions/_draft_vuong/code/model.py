"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import warnings
from collections import Counter
import torch
import torch.nn as nn
import timm

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
    "regnety_004": "regnety_004",                # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ

    TODO:
      - timm.create_model(name, pretrained=..., num_classes=num_classes, drop_rate=...)
        (timm tự thay head mới; head khởi tạo ngẫu nhiên)
      - nếu init == "frozen": gọi freeze_backbone(model)
      - ghi lại tên tag trọng số thực sự được tải (model.pretrained_cfg)
    """
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError(f"init không hỗ trợ: {init}")
    use_pretrained = bool(pretrained and init != "scratch")
    create_kwargs = {}
    if use_pretrained:
        # Kaggle's unauthenticated HF Hub downloads can stall on large sharded
        # checkpoints. Prefer timm's official direct URL when the model has one.
        try:
            pretrained_cfg = timm.get_pretrained_cfg(name)
        except (AttributeError, KeyError, TypeError):
            pretrained_cfg = None
        if pretrained_cfg is not None:
            url = (pretrained_cfg.get("url", "") if isinstance(pretrained_cfg, dict)
                   else getattr(pretrained_cfg, "url", ""))
            if url:
                create_kwargs["pretrained_cfg_overlay"] = {"hf_hub_id": ""}
    model = timm.create_model(name, pretrained=use_pretrained, num_classes=num_classes,
                              drop_rate=drop_rate, **create_kwargs)
    if init == "frozen":
        freeze_backbone(model)
    model._lab_backbone_name = name
    model._lab_pretrained_cfg = dict(getattr(model, "pretrained_cfg", {}) or {})
    return model


def unwrap_model(model):
    """Return the underlying module for DataParallel or a plain model otherwise."""
    return model.module if isinstance(model, nn.DataParallel) else model


def use_data_parallel(model, device):
    """Use PyTorch DataParallel for Kaggle CUDA runs; leave CPU runs unwrapped."""
    model = model.to(device)
    if torch.device(device).type == "cuda":
        return nn.DataParallel(model)
    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    """
    model = unwrap_model(model)
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if classifier is None:
        raise ValueError("model không có classifier head nhận diện được")
    head_ids = {id(p) for p in classifier.parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in head_ids)
    model._lab_frozen = True


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay

    TODO:
      - bỏ qua tham số requires_grad == False
      - trả về list[dict] dạng {"params": [...], "lr": ..., "weight_decay": ...}
      - (trục E) mở rộng: LR theo tầng nếu bạn muốn thử
    """
    model = unwrap_model(model)
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    head_ids = {id(p) for p in classifier.parameters()} if classifier is not None else set()
    groups = {"backbone_decay": [], "backbone_nodecay": [], "head": []}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            groups["head"].append(p)
        elif p.ndim <= 1:
            groups["backbone_nodecay"].append(p)
        else:
            groups["backbone_decay"].append(p)
    out = []
    for key, lr, wd in (("backbone_decay", lr_backbone, weight_decay),
                        ("backbone_nodecay", lr_backbone, 0.0),
                        ("head", lr_head, weight_decay)):
        if groups[key]:
            out.append({"params": groups[key], "lr": lr, "weight_decay": wd})
    if not out:
        raise ValueError("Không có tham số nào được tối ưu")
    return out


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng. TODO."""
    model = unwrap_model(model)
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x).

    TODO: dùng thư viện đếm (fvcore, ptflops, thop...) hoặc tự đếm bằng hook.
    Ghi rõ công cụ đã dùng; số có thể lệch vài phần trăm giữa các công cụ.
    """
    model = unwrap_model(model)
    try:
        from fvcore.nn import FlopCountAnalysis
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        def count_sdpa(inputs, outputs):
            shape = inputs[0].type().sizes()
            if shape is None or len(shape) != 4:
                return Counter()
            batch, heads, tokens, head_dim = shape
            # QK^T and attention*V: two token-by-token matrix products.
            return Counter({"scaled_dot_product_attention": 2 * batch * heads * tokens * tokens * head_dim})
        with torch.inference_mode():
            analysis = FlopCountAnalysis(model, dummy).set_op_handle(
                "aten::scaled_dot_product_attention", count_sdpa)
            macs = analysis.total()
        model.train(was_training)
        return float(macs / 1e9)
    except ImportError:
        warnings.warn("fvcore chưa có; dùng hook Conv2d/Linear (bỏ sót MAC của attention).", RuntimeWarning)
    counts = [0]
    hooks = []
    def conv_hook(layer, inputs, output):
        out = output
        kh, kw = layer.kernel_size
        counts[0] += out.numel() * (layer.in_channels // layer.groups) * kh * kw
    def linear_hook(layer, inputs, output):
        counts[0] += output.numel() * layer.in_features
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d): hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear): hooks.append(mod.register_forward_hook(linear_hook))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, img_size, img_size, device=device))
    finally:
        for h in hooks: h.remove()
        model.train(was_training)
    return float(counts[0] / 1e9)
