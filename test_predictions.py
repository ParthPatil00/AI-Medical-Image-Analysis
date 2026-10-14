import tensorflow as tf
from pathlib import Path
from src.preprocessing import load_image_for_prediction

model = tf.keras.models.load_model(
    "models/chest_xray_cnn.keras"
)

for label in ["NORMAL", "PNEUMONIA"]:
    folder = Path("data/chest_xray/test") / label

    files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    )[:10]

    print(f"\nActual class: {label}")

    for image_path in files:
        image = load_image_for_prediction(image_path)

        probability = float(
            model.predict(image, verbose=0)[0][0]
        )

        prediction = (
            "PNEUMONIA" if probability >= 0.5 else "NORMAL"
        )

        print(
            f"{image_path.name}: "
            f"Predicted={prediction}, "
            f"Pneumonia probability={probability:.4f}"
        )