"""Evaluate the trained CNN on the held-out test split."""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = Path("data/chest_xray/test")
MODEL_PATH = Path("models/chest_xray_cnn.keras")
OUTPUT_DIR = Path("outputs")
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model missing. Run `python train.py` first.")
    if not DATA_DIR.exists():
        raise FileNotFoundError("Test dataset missing. Follow data/README_dataset.txt first.")

    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=False
    )
    class_names = ds.class_names
    if class_names != ["NORMAL", "PNEUMONIA"]:
        raise ValueError(f"Expected NORMAL and PNEUMONIA folders; found {class_names}")
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    model = tf.keras.models.load_model(MODEL_PATH)
    y_true = np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int)
    probabilities = model.predict(ds, verbose=1).ravel()
    y_pred = (probabilities >= 0.5).astype(int)

    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=class_names, output_dict=True,
        zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    disp.plot()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=class_names, zero_division=0))
    print(f"Outputs saved to {OUTPUT_DIR}/")
    print("Reminder: test metrics do not establish clinical safety or diagnostic validity.")

if __name__ == "__main__":
    main()
