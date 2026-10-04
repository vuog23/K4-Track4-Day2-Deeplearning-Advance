"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import random
import torch
import warnings
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.

    TODO:
      - đọc ba file CSV bằng pandas
      - trả về (train_df, val_df, test_df)
    """
    labels_dir = Path(labels_dir)
    frames = [pd.read_csv(labels_dir / f"{name}_subset{fold}.csv")
              for name in ("train", "val", "test")]
    for name, frame in zip(("train", "val", "test"), frames):
        # Author-provided split CSVs contain Filename/Label only; Species lives in labels.csv.
        missing = {"Filename", "Label"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} CSV thiếu cột {sorted(missing)}")
        if frame["Filename"].isna().any() or frame["Label"].isna().any():
            raise ValueError(f"{name} CSV chứa Filename/Label rỗng")
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    TODO kiểm tra, mỗi ý lỗi thì `assert` / raise để dừng ngay:
      1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. hợp ba tập phải bằng đúng 17.509 ảnh
      4. mọi Filename đều tồn tại trong `images_dir`
    Trả về dict, ví dụ {"n": {...}, "per_class": {...}, "overlap": {...}} để dán vào báo cáo.
    """
    frames = {"train": train_df, "val": val_df, "test": test_df}
    expected = set(range(NUM_CLASSES))
    names = {k: set(v["Filename"].astype(str)) for k, v in frames.items()}
    overlap = {"train_val": sorted(names["train"] & names["val"]),
               "train_test": sorted(names["train"] & names["test"]),
               "val_test": sorted(names["val"] & names["test"])}
    if any(overlap.values()):
        raise ValueError(f"Split có ảnh trùng: { {k: len(v) for k, v in overlap.items()} }")
    union = set.union(*names.values())
    if len(union) != 17509:
        raise ValueError(f"Hợp 3 split phải có 17,509 ảnh; có {len(union)}")
    per_class = {}
    counts = {}
    for split, frame in frames.items():
        labels = pd.to_numeric(frame["Label"], errors="coerce")
        if labels.isna().any() or not labels.isin(expected).all():
            raise ValueError(f"{split} có Label ngoài khoảng 0..8")
        per_class[split] = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(NUM_CLASSES)}
        counts[split] = len(frame)
    root = Path(images_dir)
    missing_files = [f for values in names.values() for f in values if not (root / f).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Thiếu {len(missing_files)} ảnh trong {root}; ví dụ: {missing_files[:5]}")
    report = {"n": counts, "per_class": per_class,
              "overlap": {k: len(v) for k, v in overlap.items()}, "union": len(union)}
    proportions = {k: v / sum(counts.values()) for k, v in counts.items()}
    report["proportions"] = proportions
    expected_props = {"train": 0.60, "val": 0.20, "test": 0.20}
    for split, expected in expected_props.items():
        if abs(proportions[split] - expected) > 0.01:
            warnings.warn(f"{split} chiếm {proportions[split]:.2%}, lệch hơn 1 điểm % so với {expected:.0%}; "
                          "hãy báo giảng viên trước khi train.", RuntimeWarning)
    print("Split sizes:", counts)
    print("Per-class counts:", per_class)
    print("Overlap counts:", report["overlap"], "union:", len(union))
    print("Split proportions:", {k: f"{v:.2%}" for k, v in proportions.items()})
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị.

    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) (hoặc giữ nguyên 256; ghi rõ bạn chọn gì)
              + ToTensor + Normalize. KHÔNG augmentation ngẫu nhiên khi đánh giá.

    TODO: dùng torchvision.transforms (hoặc v2). Lưu ý: lật dọc có hợp lệ với ảnh cỏ dại không?
    """
    if img_size < 32:
        raise ValueError("img_size quá nhỏ")
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        ops = [transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BILINEAR),
               transforms.RandomHorizontalFlip()]
        if aug == "color":
            ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
        elif aug == "trivial":
            ops.append(transforms.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR))
        elif aug == "randaug":
            ops.append(transforms.RandAugment(interpolation=InterpolationMode.BILINEAR))
        elif aug != "basic":
            raise ValueError(f"augmentation không hỗ trợ: {aug}")
        ops.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(ops)
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size), transforms.ToTensor(), normalize])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) phải trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.

    TODO:
      - __init__(self, df, images_dir, transform): giữ df, mở ảnh bằng PIL, chuyển sang RGB
      - __len__
      - __getitem__ -> (tensor, int(label), filename)
      - (tuỳ chọn) nạp trước ảnh vào RAM nếu bị nghẽn đọc đĩa trên Colab
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        with Image.open(self.images_dir / filename) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["Label"]), filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader.

    TODO:
      - train=True: shuffle (hoặc dùng sampler); train=False: không shuffle, giữ thứ tự df
        (thứ tự phải ổn định để ghép logit với Filename)
      - sampler=None | "balanced": "balanced" dùng WeightedRandomSampler với trọng số
        1/(số ảnh của lớp) (trục D của GUIDE.md mục 3)
      - drop_last=True khi train nếu batch cuối quá nhỏ làm BatchNorm không ổn định
      - pin_memory=True, num_workers hợp lý; seed cho worker (worker_init_fn) để tái lập
    """
    ds = DeepWeedsDataset(df, images_dir, transform)
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed() % (2**32))
    chosen_sampler = None
    shuffle = bool(train)
    if sampler is not None:
        if sampler != "balanced":
            raise ValueError(f"sampler không hỗ trợ: {sampler}")
        y = df["Label"].astype(int).to_numpy()
        class_counts = np.bincount(y, minlength=NUM_CLASSES)
        weights = 1.0 / np.maximum(class_counts[y], 1)
        chosen_sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                               num_samples=len(y), replacement=True,
                                               generator=generator)
        shuffle = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=chosen_sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=bool(train and len(ds) >= batch_size),
                      worker_init_fn=_seed_worker, generator=generator,
                      persistent_workers=num_workers > 0)
