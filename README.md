::: {align="center"}
# 🧠 Brain Tumor MRI Classification

### An AI-powered research project for multi-class brain MRI image classification

**Deep Learning · TensorFlow/Keras · EfficientNetB0 · Streamlit**

A clean, interactive web application that classifies uploaded brain MRI
images into four dataset categories using transfer learning.
:::

------------------------------------------------------------------------

## ✨ Project Overview

Brain Tumor MRI Classification is an educational deep-learning project
exploring how convolutional neural networks and transfer learning can
classify MRI images. It compares a custom CNN with an
ImageNet-pretrained EfficientNetB0 model and provides a Streamlit
interface for experimenting with predictions.

> **Medical disclaimer:** This is an educational research prototype, not
> a medical device. Predictions may be wrong and must not be used to
> diagnose, rule out, or make treatment decisions about a medical
> condition. Consult a qualified medical professional.

## 🎯 Project Goals

-   Inspect and organize a labelled brain MRI dataset.
-   Resize images to a consistent input size and apply training-time
    augmentation.
-   Build and evaluate a custom CNN baseline.
-   Train and evaluate an EfficientNetB0 transfer-learning model.
-   Compare accuracy, precision, recall, F1-score, loss, and confusion
    matrices.
-   Explore predictions through a Streamlit web interface.

## 🧩 Supported Classes

  Dataset label   Display name
  --------------- --------------
  `glioma`        Glioma
  `meningioma`    Meningioma
  `no_tumor`      No Tumor
  `pituitary`     Pituitary

These are dataset categories, not confirmed medical findings.

## 🏆 Current Results

Results below are from the test-set evaluation output produced during
this project.

  Metric                 Custom CNN   EfficientNetB0
  -------------------- ------------ ----------------
  Test accuracy              67.89%       **87.40%**
  Weighted precision         58.98%       **87.38%**
  Weighted recall            67.89%       **87.40%**
  Weighted F1-score          59.29%       **86.97%**
  Test loss                  0.7323       **0.3383**

EfficientNetB0 improved test accuracy by **19.51 percentage points**
compared with the Custom CNN in these runs.

These results apply only to this project's test set of 246 images. They
do not establish clinical accuracy, and performance on external images
may differ.

## 🖥️ Application Features

-   Upload a JPG, JPEG, or PNG image.
-   Preview the uploaded image.
-   View the top predicted dataset class.
-   View model output scores for all four classes.
-   Switch between light and dark themes.
-   Clear the current image and try another.
-   Read a visible medical-use disclaimer.

The displayed confidence is a model output score, not a calibrated
probability that a medical diagnosis is correct.

## 🧰 Technology Stack

  Technology             Purpose
  ---------------------- ---------------------------------------
  Python                 Programming language
  TensorFlow / Keras     Model training and inference
  EfficientNetB0         Pretrained transfer-learning backbone
  NumPy                  Numerical operations and image arrays
  Pillow                 Image loading and conversion
  scikit-learn           Classification metrics
  Matplotlib / Seaborn   Evaluation visualizations
  Pandas                 Model comparison CSV
  Streamlit              Interactive web application

## 📁 Project Structure

``` text
Brain Tumor MRI Image Classification/
├── app/
│   └── app.py
├── Datasets/
│   ├── train/train/       
│   ├── valid/valid/       
│   └── test/test/         
├── models/
│   ├── custom_cnn.keras
│   ├── efficientnetb0.keras
│   └── efficientnetb0_fixed.keras
├── notebooks/
│   └── Brain_Tumor_MRI_Analysis.ipynb
├── results/
│   ├── confusion_matrices/
│   ├── metrics/
│   │   └── model_comparison.csv
│   └── plots/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── custom_cnn.py
│   ├── train.py
│   ├── evaluate.py
│   ├── transfer_learning.py
│   ├── evaluate_transfer.py
│   ├── model_comparison.py
│   └── plot_existing_results.py
├── requirements.txt
└── README.md
```

Dataset files and trained model weights may be stored locally and may
not be included in a source-code repository.

## ⚙️ Setup and Installation

### 1. Open the project

If you already have the project locally, open its root folder in VS
Code. To clone a repository:

``` bash
git clone <YOUR_REPOSITORY_URL>
cd "Brain Tumor MRI Image Classification"
```

Replace `<YOUR_REPOSITORY_URL>` with your actual repository URL.

### 2. Create a virtual environment (recommended)

Windows PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, select the `.venv` interpreter in VS Code and
invoke it directly. Do not bypass device security policies.

### 3. Install dependencies

``` powershell
python -m pip install -r requirements.txt
```

If Windows Application Control blocks installation, use an approved
Python environment or contact the device administrator.

### 4. Check required files

The Streamlit app expects this model checkpoint:

``` text
models/efficientnetb0_fixed.keras
```

The dataset should contain `train`, `valid`, and `test` splits, each
with the four class folders.

## 🚀 Run the Web App

From the project root:

``` powershell
python -m streamlit run app/app.py
```

Streamlit normally opens at `http://localhost:8501`. Upload an MRI image
and select **Classify MRI** to view the top predicted class and the
model's output scores.

## 🧪 Train and Evaluate Models

Run commands from the project root. Training can take considerable time
on a CPU.

### Train the Custom CNN

``` powershell
python src/train.py
```

Best checkpoint: `models/custom_cnn.keras`

### Evaluate the Custom CNN

``` powershell
python src/evaluate.py
```

### Train EfficientNetB0

``` powershell
python src/transfer_learning.py
```

Checkpoint: `models/efficientnetb0_fixed.keras`

### Evaluate EfficientNetB0

``` powershell
python src/evaluate_transfer.py
```

### Generate model comparison CSV

``` powershell
python src/model_comparison.py
```

Output: `results/metrics/model_comparison.csv`

### Generate graphs from previously recorded epoch values

``` powershell
python src/plot_existing_results.py
```

This script reconstructs graphs from epoch values entered into the
script. It is not a substitute for saving training history directly
during training.

## 🔬 Workflow

1.  **Dataset inspection** --- inspect folder structure, classes, image
    dimensions, and class counts.
2.  **Preprocessing** --- resize images to 224 × 224 pixels and augment
    training images.
3.  **Baseline model** --- train a custom CNN using convolution,
    pooling, batch normalization, and dropout.
4.  **Transfer learning** --- use pretrained EfficientNetB0 with a
    four-class classification head.
5.  **Fine-tuning** --- unfreeze selected pretrained layers and continue
    training with a smaller learning rate.
6.  **Evaluation** --- calculate test accuracy, precision, recall,
    F1-score, loss, and confusion matrix.
7.  **Comparison** --- compare both models on the same test split.
8.  **Demonstration** --- load the saved EfficientNetB0 checkpoint in
    Streamlit.

## 📊 Evaluation Artifacts

-   `results/confusion_matrices/` --- class-by-class prediction counts.
-   `results/metrics/model_comparison.csv` --- summary metrics for both
    models.
-   `results/plots/` --- training and validation accuracy/loss graphs.

## 🛠️ Troubleshooting

**Model file not found** - Confirm `models/efficientnetb0_fixed.keras`
exists. - Run Streamlit from the project root.

**TensorFlow reports no GPU support on native Windows** - TensorFlow may
run on CPU, so training can be slower. This message alone does not mean
training failed.

**`pip.exe` is blocked** - Try
`python -m pip install -r requirements.txt`. If Windows Application
Control also blocks it, use an approved environment or ask the
administrator.

**Training-history JSON files are missing** - Earlier runs occurred
before history saving was added. The graph-reconstruction script uses
previously recorded epoch values.

**Predictions look unexpected** - Confirm the intended checkpoint is
loaded, the class order matches the training folder mapping, and
preprocessing matches the successful training/evaluation pipeline.

## 🔐 Data, Ethics, and Limitations

-   Use only data you are authorized to access and share.
-   Do not upload identifiable or sensitive medical images to a public
    deployment without appropriate consent, safeguards, and review.
-   Dataset labels and model predictions can be incorrect.
-   A high score on one test set does not prove generalization to other
    hospitals, scanners, populations, or image formats.
-   This project is for learning and research, not clinical
    decision-making.

## 🗺️ Future Improvements

-   Save training histories automatically for every experiment.
-   Add automated tests for preprocessing, model loading, and prediction
    output.
-   Evaluate per-class metrics across repeated runs and external
    datasets.
-   Add versioned experiment configurations and reproducibility
    metadata.
-   Improve accessibility and explain model limitations.
-   Consider calibration and robust external validation before making
    claims beyond this dataset.

## 👩‍💻 Project Details

**Project:** Brain Tumor MRI Classification\
**Focus:** Deep Learning · Computer Vision · Transfer Learning · Model
Evaluation · Streamlit

------------------------------------------------------------------------

::: {align="center"}
**Built for learning, experimentation, and responsible AI.** 🧠
:::
