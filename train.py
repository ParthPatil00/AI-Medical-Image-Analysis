"""Train a basic CNN on a folder-based chest X-ray dataset."""
from pathlib import Path
import json
import tensorflow as tf
import matplotlib.pyplot as plt

DATA_DIR = Path("data/chest_xray")
MODEL_PATH = Path("models/chest_xray_cnn.keras")
OUTPUT_DIR = Path("outputs")
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 123

def main():
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    for path in (train_dir, val_dir):
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {path}. Follow data/README_dataset.txt first."
            )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", seed=SEED
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", seed=SEED, shuffle=False
    )
    class_names = train_ds.class_names
    if class_names != ["NORMAL", "PNEUMONIA"]:
        raise ValueError(
            f"Expected class folders NORMAL and PNEUMONIA in alphabetical order; found {class_names}"
        )

    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]
    class_weights = {0: 1.95, 1: 0.67}

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in values] for k, values in history.history.items()}, f, indent=2)

    plt.figure()
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=150)
    plt.close()
    print(f"Model saved to: {MODEL_PATH}")
    print("Reminder: educational prototype only; not for clinical use.")

if __name__ == "__main__":
    main()
