import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from asl_v2.config import TrainConfig
from asl_v2.data import load_cnn_images
from asl_v2.evaluation import compute_metrics, save_confusion_matrix, save_metrics_json
from asl_v2.models import build_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN on ASL images.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/cnn"))
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument("--trainable-backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)

    X, y, encoder = load_cnn_images(
        dataset_root=args.dataset_root,
        target_size=(64, 64),
        max_per_class=args.max_per_class,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_state,
        stratify=np.argmax(y, axis=1),
    )

    model = build_cnn(
        num_classes=y.shape[1],
        input_shape=(64, 64, 3),
        use_pretrained=args.use_pretrained,
        trainable_backbone=args.trainable_backbone,
    )

    datagen = ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.1,
        horizontal_flip=False,
    )

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "cnn_model.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=train_cfg.patience, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=train_cfg.batch_size),
        validation_data=(X_test, y_test),
        epochs=train_cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    metrics = compute_metrics(y_true, y_pred, encoder.classes_)
    save_metrics_json(metrics, args.artifacts_dir / "metrics.json")
    save_confusion_matrix(y_true, y_pred, encoder.classes_, args.artifacts_dir / "confusion_matrix.png", "CNN Confusion Matrix")

    np.save(args.artifacts_dir / "classes.npy", encoder.classes_)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
