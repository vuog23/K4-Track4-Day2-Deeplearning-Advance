"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations

import time
import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    TODO:
      - chạy warmup lần đầu rồi bỏ
      - với mỗi lần đo: sync(); t0 = time.perf_counter(); fn(); sync(); lấy hiệu * 1000
      - trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}
    Gợi ý: dùng numpy.percentile hoặc torch.quantile.
    """
    if warmup < 0 or iters < 1:
        raise ValueError("warmup >= 0 và iters >= 1")
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()
    samples = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
            "p99": float(np.percentile(samples, 99)),
            "mean": float(np.mean(samples)), "n": int(iters)}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    TODO:
      - model.eval(), torch.inference_mode()
      - dtype: "fp32" | "amp" (autocast) | "fp16" (model.half())
      - gọi bench(...) với sync phù hợp; lấy tên GPU bằng torch.cuda.get_device_name
      - Nhớ: ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định
    """
    device_obj = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(device_obj).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device_obj)
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype ต้องเป็น fp32, amp hoặc fp16")
    original_dtype = next(model.parameters()).dtype
    if dtype == "fp16":
        model.half()
        x = x.half()
    def forward():
        with torch.inference_mode():
            if dtype == "amp" and device_obj.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(x)
            else:
                model(x)
    sync = torch.cuda.synchronize if device_obj.type == "cuda" else None
    results = bench(forward, warmup=warmup, iters=iters, sync=sync)
    gpu_name = torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else "CPU"
    if dtype == "fp16":
        model.float() if original_dtype == torch.float32 else model.to(dtype=original_dtype)
    return {"gpu": gpu_name, "dtype": dtype, "batch": batch_size, "img_size": img_size,
            **results, "images_per_s": float(batch_size / (results["p50"] / 1000.0)),
            "torch": torch.__version__}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy (slide trang 63). TODO: đo thật, so với K * p50."""
    warmup = kw.pop("warmup", 10)
    iters = kw.pop("iters", 100)
    device = torch.device(kw.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    batch = int(kw.get("batch_size", 1))
    size = int(kw.get("img_size", 224))
    dtype = kw.get("dtype", "fp32")
    x = torch.randn(batch, 3, size, size, device=device)
    if dtype == "fp16":
        model.half()
        x = x.half()
    model.to(device).eval()
    def forward_views():
        with torch.inference_mode():
            for _ in range(k_views):
                if dtype == "amp" and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        model(x)
                else:
                    model(x)
    result = bench(forward_views, warmup=warmup, iters=iters,
                   sync=torch.cuda.synchronize if device.type == "cuda" else None)
    return {"k_views": k_views, "batch": batch, "img_size": size, "dtype": dtype,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", **result}
