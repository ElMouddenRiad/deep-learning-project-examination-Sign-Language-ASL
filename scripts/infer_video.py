import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from asl_v2.landmarks import create_hands_detector, extract_from_bgr
from asl_v2.temporal import TemporalDecoder, deduplicate_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stabilized ASL video inference.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--mode", choices=["cnn", "mlp_landmarks"], default="cnn")
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--ema-alpha", type=float, default=0.4)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    classes = np.load(args.classes, allow_pickle=True)

    decoder = TemporalDecoder(
        window_size=args.window_size,
        ema_alpha=args.ema_alpha,
        min_confidence=args.min_confidence,
        min_run_length=2,
    )

    hands = create_hands_detector(static_image_mode=True) if args.mode == "mlp_landmarks" else None
    previous_kp = None

    cap = cv2.VideoCapture(str(args.video))
    frame_idx = 0
    letters = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_skip != 0:
            frame_idx += 1
            continue

        if args.mode == "cnn":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (64, 64)).astype(np.float32) / 255.0
            probs = model.predict(np.expand_dims(frame_resized, axis=0), verbose=0)[0]
        else:
            result = extract_from_bgr(
                frame_bgr=frame,
                hands=hands,
                normalize_mode="rotation",
                previous_kp=previous_kp,
                stability_threshold=0.2,
            )
            if result.keypoints is None or not result.is_stable or result.quality_score < 0.03:
                frame_idx += 1
                continue
            previous_kp = result.keypoints
            probs = model.predict(np.expand_dims(result.keypoints, axis=0), verbose=0)[0]

        stable_letter = decoder.update(probs=probs, labels=classes)
        if stable_letter is not None:
            letters.append(stable_letter)
            print(f"[frame {frame_idx}] stable={stable_letter}")

        frame_idx += 1

    cap.release()
    if hands is not None:
        hands.close()

    compact = deduplicate_sequence(letters)
    print("Raw letters:", " ".join(letters))
    print("Deduplicated:", "".join(compact))


if __name__ == "__main__":
    main()
