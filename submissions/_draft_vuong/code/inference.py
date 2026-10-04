"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    TODO: model.eval(), torch.inference_mode(), (tuỳ chọn) autocast. Trả về numpy.
    """
    model.eval()
    filenames, labels, outputs = [], [], []
    device = torch.device(device)
    with torch.inference_mode():
        for images, target, names in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)
            logits = model(images)
            filenames.extend(list(names))
            labels.append(target.numpy())
            outputs.append(logits.float().cpu().numpy())
    return filenames, np.concatenate(labels), np.concatenate(outputs)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W). TODO: dùng torch.flip trên chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch. TODO."""
    h, w = x.shape[-2:]
    if crop > h or crop > w or crop <= 0:
        raise ValueError("crop phải nằm trong rentang 1..min(H,W)")
    positions = [(0, 0), (0, w - crop), (h - crop, 0),
                 (h - crop, w - crop), ((h - crop) // 2, (w - crop) // 2)]
    return [x[:, :, top:top + crop, left:left + crop] for top, left in positions]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch. TODO.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ). Ghi rõ giới hạn bạn gặp.
    """
    out = []
    for size in sizes:
        size = int(size)
        out.append(F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False,
                                 antialias=True))
    return out


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Slide chưa kết luận cách nào luôn tốt hơn: chọn một và ghi rõ, hoặc so sánh cả hai (I03).
    TODO: trả về xác suất (N, 9) đã chuẩn hoá.
    """
    if not logits_per_view:
        raise ValueError("Cần ít nhất một view")
    if space not in {"prob", "logit"}:
        raise ValueError("space phải là prob hoặc logit")
    tensors = [torch.as_tensor(x, dtype=torch.float64) for x in logits_per_view]
    if any(t.shape != tensors[0].shape for t in tensors):
        raise ValueError("Tất cả logits phải cùng dạng (N,K)")
    if space == "prob":
        probs = [torch.softmax(t, dim=1) for t in tensors]
        result = torch.stack(probs).mean(0)
    else:
        result = torch.softmax(torch.stack(tensors).mean(0), dim=1)
    return result.numpy()


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed). TODO.

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    if not list_of_probs:
        raise ValueError("Cần ít nhất một mảng xác suất")
    arrays = [np.asarray(p, dtype=np.float64) for p in list_of_probs]
    if any(p.shape != arrays[0].shape for p in arrays):
        raise ValueError("Các xác suất ensemble phải cùng dạng")
    out = np.mean(arrays, axis=0)
    out /= out.sum(axis=1, keepdims=True).clip(1e-15)
    return out


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    TODO: tối ưu hoá một tham số (LBFGS trên log T, hoặc tìm lưới thô rồi tinh).
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    logits = torch.as_tensor(val_logits, dtype=torch.float64).cpu()
    labels = torch.as_tensor(val_labels, dtype=torch.long).cpu()
    if logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError("logits cần dạng (N,K), labels cần dạng (N,)")
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp().clamp(1e-4, 1e4), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(1e-4, 1e4))


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T). TODO."""
    if T <= 0 or not np.isfinite(T):
        raise ValueError("T phải hữu hạn và dương")
    return torch.softmax(torch.as_tensor(logits, dtype=torch.float64) / float(T), dim=1).numpy()


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    TODO:
      - model.eval() trước
      - với từng cặp (Conv2d, BatchNorm2d) liền kề: tạo conv mới (có bias) và thay BN bằng Identity
      - kiểm tra: đầu ra trước/sau gộp lệch nhau cỡ 1e-5 trở xuống (in ra sai số lớn nhất)
    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm), mục này không áp dụng; ghi rõ.
    """
    model.eval()
    fused = copy.deepcopy(model).eval()
    fused_count = 0
    def fuse_parent(parent):
        nonlocal fused_count
        names = list(parent._modules)
        i = 0
        while i + 1 < len(names):
            a, b = names[i], names[i + 1]
            conv, bn = parent._modules[a], parent._modules[b]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d) and not bn.training:
                new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                    conv.stride, conv.padding, conv.dilation, conv.groups, bias=True,
                    padding_mode=conv.padding_mode, device=conv.weight.device, dtype=conv.weight.dtype)
                w = conv.weight.detach()
                bias = conv.bias.detach() if conv.bias is not None else torch.zeros(
                    conv.out_channels, device=w.device, dtype=w.dtype)
                gamma = bn.weight.detach() if bn.affine else torch.ones_like(bn.running_var)
                beta = bn.bias.detach() if bn.affine else torch.zeros_like(bn.running_var)
                scale = gamma / torch.sqrt(bn.running_var.detach() + bn.eps)
                with torch.no_grad():
                    new.weight.copy_(w * scale.reshape(-1, 1, 1, 1))
                    new.bias.copy_(beta + (bias - bn.running_mean.detach()) * scale)
                parent._modules[a] = new
                parent._modules[b] = nn.Identity()
                fused_count += 1
            i += 1
        for child in parent.children():
            if not isinstance(child, (nn.Conv2d, nn.BatchNorm2d)):
                fuse_parent(child)
    fuse_parent(fused)
    if fused_count:
        # Structural conversion preserves outputs; report a numerical check on a deterministic probe.
        first = next(model.parameters())
        side = 224
        probe = torch.zeros(1, 3, side, side, device=first.device, dtype=first.dtype)
        with torch.inference_mode():
            before, after = model(probe), fused(probe)
        max_error = float((before - after).abs().max())
        print(f"Fused {fused_count} Conv-BN pairs; max output error={max_error:.3g}")
        if max_error > 1e-5:
            raise AssertionError(f"Conv-BN fusion changed outputs by {max_error:.3g} (>1e-5)")
    else:
        print("Không tìm thấy cặp Conv2d-BatchNorm2d liền kề; không thay đổi model.")
    fused._lab_fused_conv_bn_count = fused_count
    return fused
