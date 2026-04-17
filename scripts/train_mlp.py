import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from asl_v2.config import TrainConfig
from asl_v2.data import build_landmarks_csv, load_landmarks_for_training
from asl_v2.evaluation import compute_metrics, save_confusion_matrix, save_metrics_json
from asl_v2.models import build_mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP on ASL landmarks.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/mlp"))
    parser.add_argument("--landmarks-csv", type=Path, default=Path("artifacts/landmarks/keypoints_dataset_v2.csv"))
    parser.add_argument("--max-per-class", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)

    stats = build_landmarks_csv(
        dataset_root=args.dataset_root,
        output_csv=args.landmarks_csv,
        max_per_class=args.max_per_class,
        min_quality=0.03,
        normalize_mode="rotation",
    )
    print("Landmarks extraction:", stats)

    (X_train, X_test, y_train, y_test), encoder = load_landmarks_for_training(
        args.landmarks_csv,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_state,
    )

    model = build_mlp(num_features=X_train.shape[1], num_classes=y_train.shape[1])

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "mlp_model.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=train_cfg.patience, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    metrics = compute_metrics(y_true, y_pred, encoder.classes_)
    save_metrics_json(metrics, args.artifacts_dir / "metrics.json")
    save_confusion_matrix(y_true, y_pred, encoder.classes_, args.artifacts_dir / "confusion_matrix.png", "MLP Confusion Matrix")

    np.save(args.artifacts_dir / "classes.npy", encoder.classes_)

    print(classification_report(y_true, y_pred, target_names=encoder.classes_, zero_division=0))
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
