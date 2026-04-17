from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from .landmarks import create_hands_detector, extract_from_path


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _iter_labeled_images(dataset_root: Path) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    for cls_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        label = cls_dir.name
        for img_path in cls_dir.rglob("*"):
            if img_path.suffix.lower() in VALID_EXTS:
                items.append((img_path, label))
    return items


def build_landmarks_csv(
    dataset_root: Path,
    output_csv: Path,
    max_per_class: Optional[int] = None,
    min_quality: float = 0.03,
    normalize_mode: str = "rotation",
) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    skipped_no_hand = 0
    skipped_quality = 0

    rows = []
    hands = create_hands_detector(static_image_mode=True)

    for path, label in _iter_labeled_images(dataset_root):
        if max_per_class is not None and counts[label] >= max_per_class:
            continue

        result = extract_from_path(str(path), hands, normalize_mode=normalize_mode)
        if result.keypoints is None:
            skipped_no_hand += 1
            continue

        if result.quality_score < min_quality:
            skipped_quality += 1
            continue

        row = {f"kp_{i}": float(v) for i, v in enumerate(result.keypoints)}
        row["label"] = label
        rows.append(row)
        counts[label] += 1

    hands.close()

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return {
        "num_rows": int(len(df)),
        "num_classes": int(df["label"].nunique()) if len(df) else 0,
        "skipped_no_hand": skipped_no_hand,
        "skipped_quality": skipped_quality,
    }


def load_landmarks_for_training(csv_path: Path, test_size: float, random_state: int):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    y_cat = to_categorical(y_encoded)

    return train_test_split(
        X,
        y_cat,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    ), encoder


def load_cnn_images(
    dataset_root: Path,
    target_size: Tuple[int, int] = (64, 64),
    max_per_class: Optional[int] = None,
):
    counts: Dict[str, int] = defaultdict(int)
    images: List[np.ndarray] = []
    labels: List[str] = []

    for path, label in _iter_labeled_images(dataset_root):
        if max_per_class is not None and counts[label] >= max_per_class:
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        images.append(img.astype(np.float32) / 255.0)
        labels.append(label)
        counts[label] += 1

    X = np.array(images, dtype=np.float32)
    y_raw = np.array(labels)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    y_cat = to_categorical(y_encoded)

    return X, y_cat, encoder
