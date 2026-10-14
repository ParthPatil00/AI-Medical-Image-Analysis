# AI-Powered Medical Image Analysis System
**Beginner-friendly educational project:** chest X-ray image classification into `NORMAL` and `PNEUMONIA`.

> **Medical safety:** This is an educational demonstration, not a medical device. It is not validated for clinical use and must not be used to diagnose, rule out, or treat disease. Predictions may be wrong. A qualified healthcare professional must interpret medical images.

## What's included
- `train.py` — trains a small convolutional neural network (CNN) using an image-folder dataset.
- `evaluate.py` — evaluates a saved model on a separate test dataset.
- `app.py` — local Flask web app for uploading an image and viewing a model prediction.
- `src/preprocessing.py` — shared image loading/preprocessing helpers.
- `requirements.txt` — Python packages.
- `data/README_dataset.txt` — expected dataset layout and download notes.
- `models/` — place for the trained model.
- `outputs/` — evaluation outputs.

## 1. Requirements
- Windows 10/11
- Python **3.11.x** (recommended for compatibility)
- VS Code (optional but helpful)

You do not need to uninstall other Python versions.

## 2. Set up on Windows
Open Command Prompt and run:

```cmd
cd "%USERPROFILE%"
mkdir AI-Medical-Image-Analysis
cd AI-Medical-Image-Analysis
```

If you downloaded and extracted this ZIP, instead `cd` into the extracted project folder.

Create and activate a virtual environment:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is not recognized, install Python 3.11 64-bit and retry.

## 3. Dataset
The ZIP does **not** contain X-ray images or a trained model. Obtain a dataset you are authorized to use. One commonly used educational dataset is the Chest X-Ray Images (Pneumonia) dataset hosted on Kaggle; check its current terms, access requirements, and intended use before downloading.

Arrange files like this (folder names must match exactly):

```text
data/
  chest_xray/
    train/
      NORMAL/
      PNEUMONIA/
    val/
      NORMAL/
      PNEUMONIA/
    test/
      NORMAL/
      PNEUMONIA/
```

Put image files inside each class folder. Keep patient privacy in mind and do not upload identifiable medical data to public services.

## 4. Train
With the environment activated and the dataset arranged as above:

```cmd
python train.py
```

The trained model is saved to `models/chest_xray_cnn.keras`.

## 5. Evaluate
```cmd
python evaluate.py
```
Metrics and a confusion-matrix image are written to `outputs/`.

## 6. Run the local web app
```cmd
python app.py
```
Then open `http://127.0.0.1:5000` in your browser. The app expects `models/chest_xray_cnn.keras` to exist, so train the model first.

## Project structure
```text
AI_Medical_Image_Analysis/
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   └── preprocessing.py
├── app/
│   └── templates/
│       └── index.html
├── data/
│   └── README_dataset.txt
├── models/
│   └── .gitkeep
└── outputs/
    └── .gitkeep
```

## Troubleshooting
- Confirm `python --version` shows Python 3.11.x while `.venv` is activated.
- If installation fails, copy the full error message before retrying.
- The first TensorFlow installation can take a few minutes and needs internet access.
- If you get `Model not found`, train the model before launching the web app.
