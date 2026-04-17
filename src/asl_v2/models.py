from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_mlp(num_features: int, num_classes: int) -> models.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(num_features,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(
    num_classes: int,
    input_shape=(64, 64, 3),
    use_pretrained: bool = False,
    trainable_backbone: bool = False,
) -> models.Model:
    if use_pretrained:
        base_model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Resizing(96, 96),
                layers.Lambda(lambda x: x * 255.0),
            ]
        )

        mobile = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(96, 96, 3),
        )
        mobile.trainable = trainable_backbone

        model = models.Sequential(
            [
                base_model,
                mobile,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    else:
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
