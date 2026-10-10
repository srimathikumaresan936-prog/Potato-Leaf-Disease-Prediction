import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def make_balanced_split(dataset_dir: Path, output_dir: Path, images_per_class: int):
    """Create balanced train/validation/test folders from class subfolders."""
    random.seed(SEED)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    split_names = ("train", "validation", "test")
    for split in split_names:
        for class_name in CLASS_NAMES:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        source = dataset_dir / class_name
        if not source.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {source}. "
                "The dataset directory must contain all three potato class folders."
            )

        images = [
            p for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if len(images) < images_per_class:
            raise ValueError(
                f"{class_name} has only {len(images)} images; "
                f"need at least {images_per_class}."
            )

        selected = random.sample(images, images_per_class)
        random.shuffle(selected)
        train_end = int(0.70 * len(selected))
        val_end = int(0.85 * len(selected))
        splits = {
            "train": selected[:train_end],
            "validation": selected[train_end:val_end],
            "test": selected[val_end:],
        }

        for split, files in splits.items():
            for image_path in files:
                shutil.copy2(image_path, output_dir / split / class_name / image_path.name)

    print(f"Balanced data split created at: {output_dir}")


def build_model():
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    return models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Train a CNN to classify potato leaf images."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the folder containing the three Potato___* class folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/split",
        help="Folder where balanced train/validation/test splits are created.",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=152,
        help="Number of images sampled per class (default: 152, matching the notebook).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    dataset_dir = Path(args.dataset_dir)
    split_dir = Path(args.output_dir)
    make_balanced_split(dataset_dir, split_dir, args.images_per_class)

    train_data = tf.keras.utils.image_dataset_from_directory(
        split_dir / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_data = tf.keras.utils.image_dataset_from_directory(
        split_dir / "validation",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_data = tf.keras.utils.image_dataset_from_directory(
        split_dir / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Class order:", train_data.class_names)
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
    )
    test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")

    y_true, y_pred = [], []
    for images, labels in test_data:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())

    print(classification_report(y_true, y_pred, target_names=test_data.class_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / "potato_leaf_model.keras")

    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "epochs": args.epochs,
        "images_per_class": args.images_per_class,
        "class_names": test_data.class_names,
        "training_accuracy": [float(x) for x in history.history["accuracy"]],
        "validation_accuracy": [float(x) for x in history.history["val_accuracy"]],
    }
    Path("training_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved model to models/potato_leaf_model.keras")


if __name__ == "__main__":
    main()
