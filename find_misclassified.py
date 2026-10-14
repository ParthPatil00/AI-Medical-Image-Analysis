import csv
from pathlib import Path

import numpy as np
import tensorflow as tf


DATA_DIR = Path("data/chest_xray/test")
MODEL_PATH = Path("models/chest_xray_cnn.keras")
OUTPUT_FILE = Path("outputs/misclassified_images.csv")
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    # Use the same dataset-loading method as evaluate.py
    dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    class_names = dataset.class_names

    if class_names != ["NORMAL", "PNEUMONIA"]:
        raise ValueError(f"Unexpected class order: {class_names}")

    # Normalize pixel values exactly as evaluate.py does
    dataset = dataset.map(
        lambda images, labels: (
            tf.cast(images, tf.float32) / 255.0,
            labels,
        )
    )

    # Collect actual labels and model predictions
    actual_labels = np.concatenate(
        [labels.numpy().ravel() for _, labels in dataset]
    ).astype(int)

    probabilities = model.predict(dataset, verbose=1).ravel()
    predicted_labels = (probabilities >= 0.5).astype(int)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_count = len(actual_labels)
    misclassified_count = 0

    # Match predictions to the dataset's file order
    image_paths = []

    for class_name in class_names:
        folder = DATA_DIR / class_name
        image_paths.extend(
            sorted(
                path
                for path in folder.iterdir()
                if path.suffix.lower()
                in [".jpg", ".jpeg", ".png", ".bmp"]
            )
        )

    if len(image_paths) != total_count:
        raise ValueError(
            f"Found {len(image_paths)} image paths but "
            f"the dataset contains {total_count} images."
        )

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "actual_class",
            "predicted_class",
            "pneumonia_probability",
            "image_path",
        ])

        for index in range(total_count):
            actual_class = class_names[actual_labels[index]]
            predicted_class = class_names[predicted_labels[index]]
            probability = float(probabilities[index])

            if actual_labels[index] != predicted_labels[index]:
                misclassified_count += 1

                writer.writerow([
                    actual_class,
                    predicted_class,
                    round(probability, 4),
                    str(image_paths[index]),
                ])

                print(
                    f"Actual: {actual_class} | "
                    f"Predicted: {predicted_class} | "
                    f"Pneumonia probability: {probability:.4f} | "
                    f"Image: {image_paths[index].name}"
                )

    print(f"\nTotal test images checked: {total_count}")
    print(f"Incorrect predictions: {misclassified_count}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()