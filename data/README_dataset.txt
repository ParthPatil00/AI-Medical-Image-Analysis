DATASET SETUP
============

The ZIP intentionally does not bundle a medical-image dataset. Obtain a dataset you are authorized to use and review its license/terms and intended use. Do not use this educational project for patient care.

Expected folders:
data/chest_xray/train/NORMAL/
data/chest_xray/train/PNEUMONIA/
data/chest_xray/val/NORMAL/
data/chest_xray/val/PNEUMONIA/
data/chest_xray/test/NORMAL/
data/chest_xray/test/PNEUMONIA/

Place the corresponding image files in each class folder. Folder names must be exactly NORMAL and PNEUMONIA. Keep train, validation, and test sets separate. Avoid placing images from the same patient across different splits when patient identifiers and dataset documentation allow you to check this.
