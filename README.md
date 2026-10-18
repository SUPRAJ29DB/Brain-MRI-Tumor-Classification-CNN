# Brain MRI Tumor Classification Using CNN

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SUPRAJ29DB/Brain-MRI-Tumor-Classification-CNN/blob/main/Brain-MRI-Tumor-Classification-CNN.ipynb)

A TensorFlow/Keras convolutional neural network that classifies brain MRI images into four dataset categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

This repository is an educational deep-learning project developed and trained in Google Colab. It demonstrates dataset preparation, CNN model construction, image classification, model evaluation, model saving, and single-image prediction.

> **Medical disclaimer:** This project is for educational and research purposes only. It is not a medical device, diagnostic system, or substitute for professional medical advice. Model predictions must not be used to diagnose, confirm, or rule out a brain tumor or to guide patient treatment.

## Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Contents](#repository-contents)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Initial Results](#initial-results)
- [Getting Started](#getting-started)
- [Prediction](#prediction)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Responsible Use](#responsible-use)
- [Author](#author)

## Project Overview

Brain MRI images can contain substantial visual variation in anatomy, acquisition conditions, and image quality. This project explores whether a convolutional neural network can learn patterns from labeled MRI images and assign each image to one of four categories in the source dataset.

The workflow in the notebook includes:

1. Installing the required Python packages.
2. Downloading the dataset from Kaggle.
3. Inspecting the dataset folders and class distribution.
4. Resizing images to `128 × 128` pixels.
5. Training a custom CNN using TensorFlow/Keras.
6. Evaluating predictions with accuracy, precision, recall, F1-score, and a confusion matrix.
7. Saving the trained model in Keras format.
8. Uploading an image and generating a predicted dataset category.

## Features

- Four-class MRI image classification.
- Custom CNN implemented with TensorFlow/Keras.
- Automatic image resizing and pixel-value rescaling.
- Adam optimization with sparse categorical cross-entropy loss.
- Early stopping with best-weight restoration.
- Classification report and confusion-matrix visualization.
- Single-image prediction through the Colab upload widget.
- Saved trained model included in the repository.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | Model development and training |
| NumPy | Array and numerical operations |
| Pillow | Image loading and conversion |
| Matplotlib | Plotting and visualizations |
| scikit-learn | Classification metrics |
| Seaborn | Confusion-matrix visualization |
| Kaggle API | Dataset download |
| Google Colab | Notebook execution and GPU training |

## Repository Contents

```text
Brain-MRI-Tumor-Classification-CNN/
├── Brain-MRI-Tumor-Classification-CNN.ipynb  # Complete training and prediction workflow
├── brain_mri_cnn.keras                         # Saved TensorFlow/Keras model
└── README.md                                   # Project documentation
```

The notebook is the primary source of the implementation. It includes the data-download, preprocessing, training, evaluation, model-export, and prediction steps.

## Dataset

This project uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) published on Kaggle by Masoud Nickparvar.

The dataset contains four directory-based classes:

| Class | Meaning in the source dataset |
|---|---|
| `glioma` | MRI images labeled as glioma |
| `meningioma` | MRI images labeled as meningioma |
| `notumor` | MRI images labeled as no tumor |
| `pituitary` | MRI images labeled as pituitary tumor |

The expected directory structure is:

```text
brain-tumor-mri-dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

In the recorded notebook run, the dataset contained 5,600 training images and 1,600 testing images, with 1,400 training images and 400 testing images in each class. Counts may differ if the dataset is updated or downloaded from another source.

## Model Architecture

```text
Input image: 128 × 128 × 3
          │
          ▼
Rescaling: pixel values / 255
          │
          ▼
Conv2D: 32 filters → MaxPooling2D
          │
          ▼
Conv2D: 64 filters → MaxPooling2D
          │
          ▼
Conv2D: 128 filters → MaxPooling2D
          │
          ▼
Conv2D: 256 filters → MaxPooling2D
          │
          ▼
GlobalAveragePooling2D
          │
          ▼
Dense: 128 units → Dropout: 0.4
          │
          ▼
Softmax output: 4 classes
```

### Training configuration

| Parameter | Value |
|---|---|
| Input size | `128 × 128 × 3` |
| Batch size | `32` |
| Maximum epochs | `15` |
| Optimizer | Adam |
| Loss | Sparse categorical cross-entropy |
| Output activation | Softmax |
| Early stopping | Patience of 3 epochs, restoring best weights |
| Hardware used | Google Colab GPU runtime |

## Initial Results

The recorded notebook run reported the following results:

| Metric | Score |
|---|---:|
| Accuracy | 76.88% |
| Macro precision | 0.78 |
| Macro recall | 0.77 |
| Macro F1-score | 0.75 |

Per-class results from that run were:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `glioma` | 0.82 | 0.64 | 0.72 | 400 |
| `meningioma` | 0.73 | 0.47 | 0.58 | 400 |
| `notumor` | 0.68 | 1.00 | 0.81 | 400 |
| `pituitary` | 0.87 | 0.96 | 0.92 | 400 |

### Evaluation note

In the current notebook, the `Testing` directory is passed to Keras as `validation_data` during training and is also used for the later metrics calculation. Consequently, the reported score should be treated as an initial experiment rather than an independent, leakage-free test benchmark.

For a more reliable evaluation, create a validation split from the training directory, use that split for early stopping, and reserve the `Testing` directory exclusively for final evaluation.

Results can also vary because of random initialization, package versions, hardware, preprocessing details, and dataset changes.

## Getting Started

### Option 1: Google Colab

Google Colab is the recommended way to run this project because the notebook was developed for a Colab GPU runtime.

1. Open the notebook using the **Open in Google Colab** button at the top of this README.
2. Create a Kaggle API token from your Kaggle account settings.
3. Run the notebook cells in order.
4. Enter the Kaggle token only when prompted by the notebook.
5. Allow the notebook to download the dataset into the temporary Colab runtime.
6. Train and evaluate the model.
7. Use the final upload cell to test an individual image.

The notebook downloads the dataset under:

```text
/content/brain_mri
```

The trained model is saved during the notebook run as:

```text
/content/brain_mri_cnn.keras
```

Do not commit Kaggle API tokens, `kaggle.json`, credentials, or other secrets to the repository.

### Option 2: Run the notebook locally

Clone the repository:

```bash
git clone https://github.com/SUPRAJ29DB/Brain-MRI-Tumor-Classification-CNN.git
cd Brain-MRI-Tumor-Classification-CNN
```

Create and activate a virtual environment if desired:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the main dependencies:

```bash
pip install tensorflow numpy pillow matplotlib scikit-learn seaborn kaggle
```

Then open and run:

```text
Brain-MRI-Tumor-Classification-CNN.ipynb
```

The notebook currently uses Colab-specific functionality for Kaggle authentication, file uploads, and downloads. Local execution may therefore require adapting those cells and providing local dataset paths.

## Pretrained Model

The repository includes `brain_mri_cnn.keras`, a saved TensorFlow/Keras model from the project.

The model expects:

- RGB images.
- Images resized to `128 × 128` pixels.
- Four output classes in the dataset order:

```text
['glioma', 'meningioma', 'notumor', 'pituitary']
```

The model output is a score generated by the classifier. It is not a calibrated probability of disease and should not be interpreted as clinical certainty.

## Prediction

The final notebook cell accepts an uploaded image, converts it to RGB, resizes it to `128 × 128`, and returns the class with the highest model score.

Illustrative output:

```text
Predicted class: notumor
Model score: 92.14%
```

The displayed score represents the model's softmax output for the selected class. It does not indicate diagnostic confidence, disease probability, or medical reliability.

## Limitations

- This is an educational proof-of-concept, not a clinically validated system.
- The current evaluation procedure reuses the testing directory during training validation.
- Performance may not generalize to images from different hospitals, scanners, protocols, populations, or image sources.
- Dataset labels may not represent independent clinical verification.
- The model can produce incorrect predictions, including high-scoring incorrect predictions.
- The dataset may contain hidden similarities, duplicates, acquisition-specific patterns, or other forms of leakage.
- No clinical utility, calibration, fairness, robustness, or external validation has been established.
- The model should not be used for patient care or medical decision-making.

## Future Improvements

- Create a dedicated validation split from the training data.
- Use the testing directory only for final evaluation.
- Add reproducible random seeds and record package versions.
- Add data augmentation and compare its effect on generalization.
- Compare the custom CNN with transfer-learning models such as MobileNetV2 or EfficientNet.
- Evaluate on an independent external dataset.
- Investigate duplicate images, class balance, and possible dataset leakage.
- Add calibration and uncertainty analysis.
- Generate Grad-CAM visualizations for interpretability research.
- Add automated tests and a reproducible training script.
- Build an optional Streamlit demonstration with prominent safety warnings.

## Responsible Use

This repository demonstrates image-classification techniques for learning and experimentation. It must not be used to:

- Diagnose or screen patients.
- Confirm or rule out a tumor.
- Recommend treatment.
- Replace radiologists, clinicians, or other qualified healthcare professionals.
- Make decisions about a person's health or care.

MRI scans should be interpreted by qualified healthcare professionals using appropriate clinical context and validated diagnostic procedures.

## Author

**Suprakash Ghosh**  
B.Tech — Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning

GitHub: [@SUPRAJ29DB](https://github.com/SUPRAJ29DB)

If you found this project useful for learning, consider starring the repository.
