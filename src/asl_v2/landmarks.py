from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


EXPECTED_KP_SIZE = 42


@dataclass
class ExtractionResult:
    keypoints: Optional[np.ndarray]
    quality_score: float
    is_stable: bool


def create_hands_detector(static_image_mode: bool = True) -> mp.solutions.hands.Hands:
    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _normalize_landmarks(landmarks: Iterable, mode: str = "wrist") -> np.ndarray:
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)

    if mode == "wrist":
        pts = pts - pts[0]
    elif mode == "palm_scale":
        pts = pts - pts[0]
        palm_size = np.linalg.norm(pts[9] - pts[0]) + 1e-6
        pts = pts / palm_size
    elif mode == "rotation":
        pts = pts - pts[0]
        axis = pts[9] - pts[0]
        angle = np.arctan2(axis[1], axis[0])
        c, s = np.cos(-angle), np.sin(-angle)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = pts @ rot.T
        palm_size = np.linalg.norm(pts[9] - pts[0]) + 1e-6
        pts = pts / palm_size
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return pts.reshape(-1)


def _quality_score(landmarks: np.ndarray) -> float:
    xs = landmarks[0::2]
    ys = landmarks[1::2]
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    area = width * height
    return float(np.clip(area * 10.0, 0.0, 1.0))


def extract_from_bgr(
    frame_bgr: np.ndarray,
    hands: mp.solutions.hands.Hands,
    normalize_mode: str = "rotation",
    previous_kp: Optional[np.ndarray] = None,
    stability_threshold: float = 0.2,
) -> ExtractionResult:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return ExtractionResult(keypoints=None, quality_score=0.0, is_stable=False)

    kp = _normalize_landmarks(results.multi_hand_landmarks[0].landmark, mode=normalize_mode)
    if kp.size != EXPECTED_KP_SIZE:
        return ExtractionResult(keypoints=None, quality_score=0.0, is_stable=False)

    score = _quality_score(kp)
    is_stable = True
    if previous_kp is not None:
        drift = float(np.linalg.norm(kp - previous_kp) / np.sqrt(EXPECTED_KP_SIZE))
        is_stable = drift <= stability_threshold

    return ExtractionResult(keypoints=kp, quality_score=score, is_stable=is_stable)


def extract_from_path(
    image_path: str,
    hands: mp.solutions.hands.Hands,
    normalize_mode: str = "rotation",
) -> ExtractionResult:
    image = cv2.imread(image_path)
    if image is None:
        return ExtractionResult(keypoints=None, quality_score=0.0, is_stable=False)
    return extract_from_bgr(image, hands, normalize_mode=normalize_mode)
