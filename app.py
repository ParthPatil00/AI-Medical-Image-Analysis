"""Local Flask demo for an educational chest X-ray classifier."""
from pathlib import Path
import os
import tempfile
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from src.preprocessing import load_image_for_prediction, CLASS_NAMES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "chest_xray_cnn.keras"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
app = Flask(__name__, template_folder="app/templates")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
_model = None

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run `python train.py` after setting up the dataset.")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or not uploaded.filename:
            error = "Choose an image file first."
        elif not allowed_file(uploaded.filename):
            error = "Please upload a PNG, JPG, JPEG, or BMP image."
        else:
            safe_name = secure_filename(uploaded.filename)
            suffix = Path(safe_name).suffix.lower()
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    uploaded.save(tmp.name)
                    temp_path = Path(tmp.name)
                try:
                    image_array = load_image_for_prediction(temp_path)
                    probability = float(get_model().predict(image_array, verbose=0)[0][0])
                finally:
                    temp_path.unlink(missing_ok=True)
                predicted_index = 1 if probability >= 0.5 else 0
                confidence = probability if predicted_index == 1 else 1 - probability
                result = {
                    "label": CLASS_NAMES[predicted_index],
                    "confidence": round(confidence * 100, 1),
                    "note": "Model output only—not a medical diagnosis."
                }
            except Exception as exc:
                error = f"Could not process image: {exc}"
    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    # Local-only development server; do not expose publicly.
    app.run(host="127.0.0.1", port=5000, debug=False)
