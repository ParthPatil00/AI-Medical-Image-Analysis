import csv
from pathlib import Path

import numpy as np
import tensorflow as tf


# Load the trained model
model = tf.keras.models.load_model(
    "models/chest_xray_cnn.keras"
)

# Folder containing the test images
test_dir = Path("data/chest_xray/test")

# File where incorrect predictions will be recorded
output_file = Path("outputs/misclassified_images.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)

misclassified_count = 0
total_count = 0

with output_file.open("w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow([
        "actual_class",
        "predicted_class",
        "pneumonia_probability",
        "image_path",
    ])

    for actual_class in ["NORMAL", "PNEUMONIA"]:
        folder = test_dir / actual_class

        image_files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        )

        for image_path in image_files:
            image = tf.keras.utils.load_img(
                image_path,
                target_size=(180, 180),
                color_mode="rgb",
                interpolation="bilinear",
            )

            image = tf.keras.utils.img_to_array(image).astype("float32") / 255.0
            image = np.expand_dims(image, axis=0)

            probability = float(
                model.predict(image, verbose=0)[0][0]
            )

            predicted_class = (
                "PNEUMONIA" if probability >= 0.5 else "NORMAL"
            )

            total_count += 1

            if predicted_class != actual_class:
                misclassified_count += 1

                writer.writerow([
                    actual_class,
                    predicted_class,
                    round(probability, 4),
                    str(image_path),
                ])

                print(
                    f"Actual: {actual_class} | "
                    f"Predicted: {predicted_class} | "
                    f"Pneumonia probability: {probability:.4f} | "
                    f"Image: {image_path.name}"
                )

print(f"\nTotal test images checked: {total_count}")
print(f"Incorrect predictions: {misclassified_count}")
print(f"Results saved to: {output_file}")