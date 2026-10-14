# AI-Powered Medical Image Analysis System

## Project Documentation, Architecture, Implementation Plan, and Proof Checklist

> **Project status:** Educational prototype for binary chest X-ray image
> classification (`NORMAL` vs `PNEUMONIA`).
>
> **Medical safety notice:** This project is for learning and
> demonstration only. It is not a medical device and has not been
> validated for clinical use. Do not use it to diagnose, rule out, or
> treat disease. Model predictions can be wrong and must not replace
> interpretation by a qualified healthcare professional.

------------------------------------------------------------------------

## 1. Project Overview

### 1.1 In simple terms

This project uses a convolutional neural network (CNN) to examine a
chest X-ray image and predict which of two dataset labels it most
resembles:

-   `NORMAL`
-   `PNEUMONIA`

A user can train the model, evaluate it on a separate test dataset, and
run a local Flask web app to upload an image and see the model's
prediction.

The model learns patterns from labeled training images. It does not
understand a scan the way a radiologist does, and a prediction is not a
diagnosis.

### 1.2 Technical overview

The project uses Python and TensorFlow/Keras to train a CNN on images
organized in class folders. Images are resized to **180 × 180 pixels**,
represented as **RGB**, and normalized to floating-point values in the
range **0--1**. The trained model is saved as a Keras `.keras` file. A
separate evaluation script calculates classification metrics and
produces a confusion matrix. A Flask application loads the saved model
and exposes a local browser interface for image uploads.

### 1.3 Problem and educational relevance

Medical-image classification is an example of computer vision: software
learns visual patterns from labeled images and applies those learned
patterns to new images. Studying this workflow introduces practical
topics used in machine-learning projects, including data organization,
preprocessing, model training, evaluation, error analysis, and a
user-facing application.

This repository demonstrates the **workflow**, not clinical readiness. A
useful portfolio project should report limitations and errors honestly,
not just the best-looking metric.

### 1.4 Intended workflow

1.  Place an appropriately licensed chest X-ray dataset in the expected
    folder structure.
2.  Create and activate the Python virtual environment.
3.  Install the packages listed in `requirements.txt`.
4.  Run `train.py` to train the CNN and save the model.
5.  Run `evaluate.py` to evaluate the saved model on the test split.
6.  Review the classification report and confusion matrix.
7.  Run `app.py` and open the local web page.
8.  Upload an image and inspect the predicted class and displayed
    confidence.
9.  Save evidence, document the measured results, and push appropriate
    source files and documentation to GitHub.

------------------------------------------------------------------------

## 2. Technology-Stack Options

These options describe possible approaches to this type of project. They
are alternatives for planning; this repository implements **Option A**,
not all three.

  --------------------------------------------------------------------------------------------
  Option              Typical tools       Advantages     Trade-offs         Best suited for
  ------------------- ------------------- -------------- ------------------ ------------------
  **A ---             Python,             Direct         A small CNN may    Learning the
  Beginner-friendly   TensorFlow/Keras, a learning path; generalize poorly; end-to-end
  (implemented        small CNN, Flask    training and   training and       machine-learning
  here)**                                 inference are  evaluation require workflow
                                          easy to        care               
                                          follow; local                     
                                          web interface                     

  **B ---             Python,             Can reuse      More design        A follow-up
  Intermediate**      TensorFlow/Keras,   learned image  choices;           experiment
                      transfer learning   features;      pretrained-model   comparing models
                      with a pretrained   useful for     licensing and      
                      vision model, Flask comparing      input requirements 
                                          approaches     need checking;     
                                                         results still      
                                                         require rigorous   
                                                         validation         

  **C --- Advanced**  Python, PyTorch or  More           More               A larger research
                      TensorFlow,         reproducible   infrastructure,    or engineering
                      experiment          engineering    security, privacy, project
                      tracking, an API    and deployment validation, and    
                      service, automated  options        maintenance work   
                      tests,                                                
                      containerization                                      
                      and monitored                                         
                      deployment                                            
  --------------------------------------------------------------------------------------------

**Implemented choice:** Option A. The current project uses a custom CNN
with convolution and pooling layers, a dense layer, dropout, and a
sigmoid output. It does not currently implement transfer learning, a
cloud deployment, or a production medical workflow.

------------------------------------------------------------------------

## 3. Architecture

### 3.1 Text architecture diagram

``` text
                 Labeled chest X-ray dataset
                              |
                              v
       data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}
                              |
                 +------------+------------+
                 |                         |
                 v                         v
       Training data loader        Validation data loader
                 |                         |
                 +------------+------------+
                              |
                              v
                   Image preprocessing
                 Resize to 180 x 180
                  RGB; scale to 0–1
                              |
                              v
                      CNN training
             Convolution + pooling layers
                  Dense layer + dropout
                    Sigmoid output
                              |
                              v
              models/chest_xray_cnn.keras
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
             evaluate.py                app.py
                  |                       |
                  v                       v
       Test metrics and confusion   Local Flask page
              matrix                Image upload
                                          |
                                          v
                                Preprocess uploaded image
                                          |
                                          v
                                 Load model and predict
                                          |
                                          v
                             NORMAL or PNEUMONIA result
```

### 3.2 Main modules

  -----------------------------------------------------------------------
  File or folder                      Responsibility
  ----------------------------------- -----------------------------------
  `train.py`                          Loads folder-based image datasets,
                                      checks class order, trains the CNN,
                                      and saves the model and training
                                      history/artifacts as implemented

  `evaluate.py`                       Loads the saved model and evaluates
                                      it against the separate test split;
                                      writes evaluation artifacts to
                                      `outputs/`

  `src/preprocessing.py`              Defines shared image size and
                                      preprocessing helpers

  `app.py`                            Runs the Flask application, loads
                                      the saved model, accepts an
                                      uploaded image, preprocesses it,
                                      and returns a prediction

  `app/templates/index.html`          HTML template for the web interface

  `data/README_dataset.txt`           Explains the expected dataset
                                      layout and dataset notes

  `models/`                           Location for the locally generated
                                      trained model

  `outputs/`                          Location for locally generated
                                      metrics and plots

  `evidence/`                         Selected screenshots and results
                                      intended to document the project
  -----------------------------------------------------------------------

### 3.3 Data flow

**Training flow:** labeled training images → image loader and
preprocessing → CNN optimization → validation monitoring → saved Keras
model.

**Evaluation flow:** test images → the same compatible preprocessing →
saved model predictions → classification report and confusion matrix.

**Web-app flow:** uploaded image → validation and preprocessing in the
app → saved model inference → predicted label and displayed score.

Preprocessing used for inference must remain compatible with
preprocessing used during training. Class-label order must also be
consistent: the project expects `NORMAL` as class index 0 and
`PNEUMONIA` as class index 1.

------------------------------------------------------------------------

## 4. Dataset and Preprocessing

### 4.1 Expected folder layout

``` text
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

The directory names and class names must match the code. Use only a
dataset you are authorized to access and use. Check its current terms,
attribution requirements, and intended use. Do not publish identifiable
patient information or private medical images.

### 4.2 Preprocessing in this implementation

-   Resize images to **180 × 180** pixels.
-   Use **RGB** image representation.
-   Convert pixel values to floating-point values and scale by **1/255**
    to the range **0--1**.
-   Preserve consistent class ordering across training, evaluation, and
    web inference.

Do not describe this implementation as grayscale 256 × 256
preprocessing; that is not the configuration used by this project.

### 4.3 Data-split caution

The current dataset counts observed during training were:

  Split          NORMAL   PNEUMONIA   Total
  ------------ -------- ----------- -------
  Training        1,341       3,875   5,216
  Validation          8           8      16
  Test              234         390     624

The validation split is very small (only 16 images). A perfect
validation score on such a small split is not reliable evidence of
generalization. Dataset versions and split counts can differ, so
re-check the output if the dataset is replaced or reorganized.

------------------------------------------------------------------------

## 5. Implementation Plan: Ten Phases

Use this plan as a reproducible project checklist. The phases describe
the workflow and expected checks; they do not claim that every possible
enhancement has already been implemented.

### Phase 1 --- Define the goal and safety boundary

-   **What:** Define the two target labels and describe the system as an
    educational classifier.
-   **Why:** A clear task prevents the project from making unsupported
    medical claims.
-   **Output:** Project scope and medical-safety notice.
-   **Common mistake:** Calling a model prediction a diagnosis.
-   **Verify:** README and web app identify the project as educational
    and not for clinical use.

### Phase 2 --- Prepare the development environment

-   **What:** Use Python 3.11, create `.venv`, activate it, and install
    `requirements.txt`.
-   **Why:** A virtual environment isolates project dependencies.
-   **Output:** Working Python environment.
-   **Common mistake:** Installing packages into a different Python
    interpreter.
-   **Verify:** With the environment activated, `python --version`
    reports Python 3.11.x and package imports succeed.

### Phase 3 --- Organize and inspect the dataset

-   **What:** Put images into the expected train/validation/test and
    class folders.
-   **Why:** Folder-based loaders infer labels from directory names.
-   **Output:** Dataset structure recognized by the loader.
-   **Common mistake:** Misspelling a class folder or mixing test images
    into training.
-   **Verify:** Run the training loader and confirm class names and
    image counts.

### Phase 4 --- Implement and verify preprocessing

-   **What:** Resize images to 180 × 180, use RGB, and scale pixel
    values to 0--1.
-   **Why:** The model needs consistent input dimensions and numerical
    ranges.
-   **Output:** Images in the input format expected by the CNN.
-   **Common mistake:** Applying a different size, color mode, or
    scaling rule in the app.
-   **Verify:** Inspect a sample preprocessed image and compare training
    and inference preprocessing.

### Phase 5 --- Build the CNN

-   **What:** Use the implemented convolution/pooling blocks, dense
    layer, dropout, and sigmoid output.
-   **Why:** Convolutional filters learn spatial patterns in images.
-   **Output:** Compiled binary-classification model.
-   **Common mistake:** Mismatching the output layer, labels, and loss
    function.
-   **Verify:** Print the model summary and confirm that the output
    represents one binary prediction.

### Phase 6 --- Train and monitor the model

-   **What:** Run `python train.py`; monitor training and validation
    metrics. The current implementation uses early stopping and class
    weights.
-   **Why:** Monitoring can help detect overfitting and address class
    imbalance during training.
-   **Output:** Saved model and training-history artifacts.
-   **Common mistake:** Treating a high training score as proof of test
    performance.
-   **Verify:** Confirm that training completes, the expected model file
    exists, and training history is saved where configured.

### Phase 7 --- Evaluate on held-out test data

-   **What:** Run `python evaluate.py` and inspect per-class precision,
    recall, F1-score, accuracy, and the confusion matrix.
-   **Why:** Aggregate accuracy can hide important class-specific
    errors.
-   **Output:** Evaluation report and confusion-matrix image in
    `outputs/`.
-   **Common mistake:** Reporting only accuracy or tuning repeatedly
    against the test set.
-   **Verify:** Check the test support counts and confirm that the
    confusion-matrix total equals the number of evaluated images.

### Phase 8 --- Perform error analysis

-   **What:** Review false positives and false negatives using the
    available error-analysis output and carefully selected examples.
-   **Why:** Mistakes show where a model may be unreliable.
-   **Output:** Error-analysis notes and, where available, a
    misclassified-images CSV.
-   **Common mistake:** Assuming every wrong prediction is a simple
    coding bug; some errors arise from data, label noise, or model
    limitations.
-   **Verify:** Explain the meanings of the four confusion-matrix cells
    and relate them to the observed results. Keep medical images private
    unless publication is authorized.

### Phase 9 --- Integrate the local web application

-   **What:** Run `python app.py` and open `http://127.0.0.1:5000`.
-   **Why:** A web interface demonstrates a basic path from user input
    to model inference.
-   **Output:** A locally running Flask app that displays a prediction.
-   **Common mistake:** Starting the app before the model exists, or
    using incompatible preprocessing.
-   **Verify:** Upload a permitted test image, check the returned label,
    and confirm that invalid uploads are handled safely.

### Phase 10 --- Document, capture evidence, and publish

-   **What:** Update README and this documentation, capture evidence,
    check `.gitignore`, commit, and push source/documentation.
-   **Why:** Reproducibility and honest reporting are important
    portfolio skills.
-   **Output:** A readable GitHub repository with source, instructions,
    evidence, and limitations.
-   **Common mistake:** Uploading the virtual environment, large
    dataset, model weights, or private medical data.
-   **Verify:** Open the GitHub repository in a browser, follow setup
    instructions, and confirm that evidence links work.

------------------------------------------------------------------------

## 6. Running the Project on Windows

Run commands from the project root.

### 6.1 Create and activate an environment

If you are starting from a fresh project checkout, open Command Prompt:

``` cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `.venv` already exists and is working, do not create another
environment unnecessarily.

### 6.2 Add the dataset

Place an authorized dataset under `data/chest_xray/` using the layout in
Section 4. If the dataset is not included in the repository, follow the
dataset provider's current download and licensing instructions.

### 6.3 Train

``` cmd
python train.py
```

Expected model path:

``` text
models/chest_xray_cnn.keras
```

Training time depends on the computer and available hardware.

### 6.4 Evaluate

``` cmd
python evaluate.py
```

Review the generated evaluation outputs in `outputs/`. Do not assume
locally generated files are tracked by Git; the repository's
`.gitignore` intentionally ignores generated output files.

### 6.5 Start the web app

``` cmd
python app.py
```

Open this address in your browser:

``` text
http://127.0.0.1:5000
```

Stop the development server with `Ctrl+C` in the terminal. This is a
local development app, not a public production deployment.

### 6.6 macOS/Linux notes

The exact setup commands differ by shell and operating system. A typical
virtual-environment activation command is:

``` bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then use `python train.py`, `python evaluate.py`, and `python app.py`
from the project root. TensorFlow compatibility and installation
availability depend on the operating system, Python version, and
hardware. If installation fails, use the official package guidance for
the relevant platform rather than assuming every platform has identical
support.

------------------------------------------------------------------------

## 7. Observed Results and Interpretation

The following values reflect one recorded run of this project. Re-run
evaluation before using them to describe a newly trained model or a
changed dataset.

### 7.1 Recorded test result

-   **Test images:** 624
-   **Correct predictions:** 484
-   **Incorrect predictions:** 140
-   **Test accuracy:** 77.56% (484 / 624)

Recorded classification report:

  Class                    Precision   Recall   F1-score   Support
  ---------------------- ----------- -------- ---------- ---------
  NORMAL                        0.96     0.42       0.58       234
  PNEUMONIA                     0.74     0.99       0.85       390
  **Overall accuracy**                          **0.78**   **624**
  Macro average                 0.85     0.70       0.71       624
  Weighted average              0.82     0.78       0.75       624

Recorded confusion matrix (rows are actual labels; columns are predicted
labels):

  Actual  Predicted     NORMAL   PNEUMONIA
  ------------------- -------- -----------
  NORMAL                    98         136
  PNEUMONIA                  4         386

### 7.2 What these results mean

-   The model correctly classified 98 of 234 NORMAL images in this run.
-   It incorrectly labeled 136 NORMAL images as PNEUMONIA.
-   It correctly classified 386 of 390 PNEUMONIA images in this run.
-   It incorrectly labeled 4 PNEUMONIA images as NORMAL.
-   The low NORMAL recall (0.42) is a major limitation and should not be
    hidden behind the overall accuracy.
-   The validation split had only 16 images. A 100% validation result
    from that split is too small to establish reliable generalization.
-   The displayed confidence score in the web app is not a guarantee of
    correctness and should not be interpreted as a calibrated
    probability of disease.

These are dataset-specific experimental results, not estimates of
real-world clinical performance. They do not establish suitability for
screening, diagnosis, triage, or treatment.

------------------------------------------------------------------------

## 8. Evidence and Screenshot Checklist

Store evidence in the matching `evidence/` subfolders. Use genuine
outputs from your own run; do not fabricate screenshots or alter metrics
to make results look better.

  -----------------------------------------------------------------------------------------------------
  Evidence                Suggested path                                        Purpose
  ----------------------- ----------------------------------------------------- -----------------------
  Example NORMAL image    `evidence/01_sample_images/sample_normal.jpeg`        Shows an example input;
                                                                                ensure use and
                                                                                publication are
                                                                                authorized

  Example PNEUMONIA image `evidence/01_sample_images/sample_pneumonia.jpeg`     Shows a second example
                                                                                input; ensure use and
                                                                                publication are
                                                                                authorized

  Preprocessing code or   `evidence/02_preprocessing/preprocessing_code.png`    Documents the input
  output                                                                        preparation

  Training accuracy curve `evidence/03_training/accuracy_curve.png`             Shows
                                                                                training/validation
                                                                                behavior, if generated
                                                                                by the project

  Training history        `evidence/03_training/training_history.json`          Preserves recorded
                                                                                training metrics

  Classification report   `evidence/04_evaluation/classification_report.json`   Preserves per-class
                                                                                metrics

  Confusion matrix        `evidence/04_evaluation/confusion_matrix.png`         Makes the error
                                                                                distribution visible

  Evaluation report       `evidence/04_evaluation/evaluation_report.png`        Shows evaluation output
  screenshot                                                                    in a readable form

  Misclassified-image     `evidence/04_evaluation/misclassified_images.csv`     Supports error
  list                                                                          analysis; check that it
                                                                                contains no sensitive
                                                                                information

  NORMAL prediction       `evidence/05_predictions/normal_prediction.png`       Demonstrates the UI
  screenshot                                                                    response for one
                                                                                example

  PNEUMONIA prediction    `evidence/05_predictions/pneumonia_prediction.png`    Demonstrates the UI
  screenshot                                                                    response for one
                                                                                example

  Web app homepage        `evidence/05_predictions/web_app_homepage.png`        Shows the local
                                                                                interface

  GitHub repository       `evidence/06_github/github_repository.png`            Shows repository
  screenshot                                                                    presentation, if saved
  -----------------------------------------------------------------------------------------------------

A screenshot of a single prediction proves only that the interface
produced that output for that input. It does not prove that the model is
accurate or clinically valid.

### Screenshot hygiene

-   Capture actual outputs from your own run.
-   Keep text and metric values readable.
-   Avoid showing usernames, private file paths, tokens, personal
    information, or identifiable patient data unnecessarily.
-   Do not upload full medical datasets or model artifacts unless you
    have confirmed licensing, privacy, size, and repository
    requirements.
-   Use relative links in the README so evidence remains navigable on
    GitHub.

------------------------------------------------------------------------

## 9. Seven-Day Proof and Portfolio Plan

This is a suggested schedule for documenting the work. If the core
project is already complete, use the plan to audit and fill gaps rather
than repeating completed work.

### Day 1 --- Environment and repository

-   Check Python version and package installation.
-   Confirm the repository contains source files, `requirements.txt`,
    README, and `.gitignore`.
-   Save a setup/structure screenshot if useful.
-   Suggested commit: `docs: document environment and project setup`

### Day 2 --- Dataset and labels

-   Confirm the authorized dataset's folder structure and class names.
-   Record dataset source, access terms, and split counts.
-   Do not commit the dataset itself.
-   Suggested commit: `docs: document dataset layout and provenance`

### Day 3 --- Preprocessing

-   Verify 180 × 180 RGB input and scaling to 0--1.
-   Capture a genuine preprocessing code/output screenshot.
-   Confirm app preprocessing matches training preprocessing.
-   Suggested commit: `docs: add preprocessing evidence`

### Day 4 --- Model architecture

-   Run or inspect the model summary.
-   Document the CNN layers and binary output in plain language.
-   Avoid claiming transfer learning or other techniques not
    implemented.
-   Suggested commit: `docs: describe CNN architecture`

### Day 5 --- Training

-   Train the model if necessary and record the actual run's
    logs/history.
-   Review training versus validation behavior.
-   Keep the model file out of Git if that remains the repository
    policy.
-   Suggested commit: `docs: record training evidence`

### Day 6 --- Evaluation and error analysis

-   Run `python evaluate.py`.
-   Record test-set size, class-wise metrics, and confusion matrix.
-   Discuss false positives and false negatives, especially the recorded
    low NORMAL recall.
-   Suggested commit: `docs: add evaluation and limitations`

### Day 7 --- App and final repository review

-   Run the local app and test an authorized image upload.
-   Capture a genuine UI screenshot.
-   Open every README evidence link on GitHub.
-   Check `git status` and ensure datasets, `.venv`, model weights, and
    private data are not staged.
-   Suggested commit: `docs: finalize project evidence and instructions`

These commit messages are examples. Use them only for changes actually
made, and do not create empty commits just to follow a schedule.

------------------------------------------------------------------------

## 10. GitHub and Version-Control Checklist

### 10.1 Keep the repository safe and reproducible

The current `.gitignore` is intended to exclude items such as:

``` gitignore
.venv/
__pycache__/
*.py[cod]
models/*.keras
outputs/*
!outputs/.gitkeep
data/chest_xray/
.DS_Store
```

Review the actual `.gitignore` before relying on this example. If a
large or sensitive file was already tracked, adding it to `.gitignore`
alone does not remove it from Git history.

### 10.2 Review changes before committing

From the project root:

``` cmd
git status
git diff
```

Stage only the intended documentation file and other files you
deliberately changed. For example, after creating this document:

``` cmd
git add docs/PROJECT_DOCUMENTATION.md
git status
git commit -m "docs: add project architecture and implementation plan"
git push
```

If the repository has no `docs` folder yet, create it first. If Git
reports that there is nothing to commit, inspect `git status` rather
than making an empty commit.

### 10.3 Final GitHub review

-   README clearly explains the project, setup, execution, measured
    results, and limitations.
-   Documentation and relative evidence links work in the GitHub browser
    view.
-   `requirements.txt` is present.
-   No `.venv`, dataset archive, model weights, credentials, or private
    medical images were accidentally committed.
-   Repository description and topics accurately describe an educational
    chest X-ray classification project.
-   Commit history reflects real work.
-   The project does not claim clinical validation, production
    readiness, or performance not supported by recorded experiments.

------------------------------------------------------------------------

## 11. Troubleshooting

  -----------------------------------------------------------------------
  Problem                             Checks
  ----------------------------------- -----------------------------------
  `python` or `pip` is not found      Activate `.venv`; check
                                      `python --version`; use
                                      `python -m pip`

  Package installation fails          Save the full error; verify
                                      Python/platform compatibility and
                                      `requirements.txt`

  Dataset directory not found         Run commands from the project root
                                      and check `data/chest_xray/`

  Class names or counts are wrong     Check exact folder names: `NORMAL`
                                      and `PNEUMONIA`

  Model file not found by the app     Run training successfully and
                                      confirm
                                      `models/chest_xray_cnn.keras`
                                      exists

  Input shape mismatch                Confirm inference uses 180 × 180
                                      RGB input compatible with training

  App does not open                   Check terminal output and whether
                                      the local server is running at
                                      `127.0.0.1:5000`

  Evidence link fails on GitHub       Check capitalization, filename,
                                      relative path, and whether the file
                                      was committed and pushed

  Git is trying to upload large files Stop before committing; inspect
                                      `git status` and `.gitignore`;
                                      confirm no data or model file is
                                      staged
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 12. Limitations and Responsible Use

1.  **Educational only:** no clinical validation or regulatory approval
    is claimed.
2.  **Limited task:** the model distinguishes two dataset labels; it
    does not perform general radiology interpretation.
3.  **Dataset dependence:** performance can change with dataset source,
    patient population, image acquisition, labels, and preprocessing.
4.  **Small validation set:** the recorded validation split contains
    only 16 images, which is inadequate for a strong generalization
    claim.
5.  **Class-specific errors:** the recorded test run has NORMAL recall
    of 0.42 and 136 NORMAL images misclassified as PNEUMONIA.
6.  **Confidence limitations:** a displayed score is not automatically a
    calibrated probability and does not establish certainty.
7.  **Privacy and licensing:** medical images may be sensitive; check
    dataset terms and avoid publishing identifiable data.
8.  **No deployment claim:** Flask is run locally for demonstration.
    Public deployment would require additional security, privacy,
    reliability, and governance work.

Any research extension should use appropriate datasets and validation
design, avoid patient-level leakage, document the data provenance,
report per-class metrics and uncertainty, and obtain qualified clinical
and ethical oversight where relevant. These extensions are
recommendations, not current project capabilities.

------------------------------------------------------------------------

## 13. Learning Outcomes

By completing and documenting this project, a learner can demonstrate
practice with:

-   Python project structure and virtual environments.
-   Image-folder datasets and label ordering.
-   Image resizing, color representation, and pixel normalization.
-   CNN-based binary classification using TensorFlow/Keras.
-   Training monitoring, early stopping, and class weighting.
-   Test-set evaluation using precision, recall, F1-score, and a
    confusion matrix.
-   Error analysis and honest reporting of limitations.
-   Connecting model inference to a local Flask interface.
-   Git, GitHub, README writing, and evidence organization.
-   Responsible handling of medical-image data and avoiding unsupported
    diagnostic claims.

------------------------------------------------------------------------

## 14. Final Completion Checklist

-   [ ] The project scope and medical-safety notice are clear.
-   [ ] The selected technology stack matches the code actually present.
-   [ ] The dataset layout and provenance are documented.
-   [ ] Preprocessing is documented accurately as 180 × 180 RGB, scaled
    to 0--1.
-   [ ] The architecture and data flow are explained.
-   [ ] Training and evaluation commands work from the project root.
-   [ ] Test metrics and the confusion matrix are reported honestly.
-   [ ] The small validation split and low NORMAL recall are disclosed.
-   [ ] The local Flask app runs and accepts a suitable image.
-   [ ] Evidence images and links are genuine and readable.
-   [ ] README and documentation match the repository's current
    filenames.
-   [ ] `.gitignore` excludes large/generated/private files as intended.
-   [ ] No unauthorized dataset or identifiable medical information is
    published.
-   [ ] GitHub instructions and relative links have been checked in a
    browser.
-   [ ] No clinical accuracy, diagnosis, or production-readiness claims
    are made.

**End of documentation.**
