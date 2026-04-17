from collections import deque
from typing import Deque, List, Optional

import numpy as np


class TemporalDecoder:
    """Smooth per-frame probabilities before producing stable letters."""

    def __init__(
        self,
        window_size: int = 7,
        ema_alpha: float = 0.4,
        min_confidence: float = 0.55,
        min_run_length: int = 2,
    ) -> None:
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.min_confidence = min_confidence
        self.min_run_length = min_run_length
        self._history: Deque[np.ndarray] = deque(maxlen=window_size)
        self._ema_probs: Optional[np.ndarray] = None
        self._last_letter: Optional[str] = None
        self._run_count = 0

    def update(self, probs: np.ndarray, labels: np.ndarray) -> Optional[str]:
        probs = np.asarray(probs, dtype=np.float32)
        self._history.append(probs)

        mean_probs = np.mean(np.stack(self._history, axis=0), axis=0)
        if self._ema_probs is None:
            self._ema_probs = mean_probs
        else:
            self._ema_probs = self.ema_alpha * mean_probs + (1 - self.ema_alpha) * self._ema_probs

        top_idx = int(np.argmax(self._ema_probs))
        top_score = float(self._ema_probs[top_idx])
        candidate = str(labels[top_idx])

        if top_score < self.min_confidence:
            self._run_count = 0
            return None

        if candidate == self._last_letter:
            self._run_count += 1
        else:
            self._last_letter = candidate
            self._run_count = 1

        if self._run_count >= self.min_run_length:
            return candidate
        return None


def deduplicate_sequence(sequence: List[str]) -> List[str]:
    result: List[str] = []
    for item in sequence:
        if not result or result[-1] != item:
            result.append(item)
    return result
