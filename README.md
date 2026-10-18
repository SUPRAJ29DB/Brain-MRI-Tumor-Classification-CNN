# Brain MRI Tumor Classification Using CNN

A Deep Learning mini project that uses a Convolutional Neural Network (CNN) to classify brain MRI images into four categories: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**.

The project is implemented in Python using TensorFlow/Keras and developed in Google Colab.

## Overview

Brain MRI scans are widely used in medical imaging to examine brain structures and abnormalities. This project explores how deep learning can classify MRI images into predefined categories using image classification techniques.

The CNN learns visual features from labeled MRI images and predicts the corresponding dataset category for a given input image.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and must not be used to diagnose, confirm, or rule out a brain tumor.

## Features

- Classifies MRI images into four dataset categories.
- Uses a CNN built with TensorFlow/Keras.
- Automatically resizes and normalizes input images.
- Uses convolutional, pooling, and dense layers for classification.
- Evaluates performance using accuracy, precision, recall, and F1-score.
- Generates a confusion matrix and classification report.
- Supports predictions on individual MRI images.
- Developed and trained using Google Colab.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | CNN development and training |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| scikit-learn | Model evaluation |
| Seaborn | Confusion matrix visualization |
| Google Colab | Development and training environment |

## Dataset

This project uses the Brain Tumor MRI Dataset available on Kaggle.

**Dataset link:**  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset commonly contains four classes:

| Class | Description |
|---|---|
| `glioma` | MRI images labeled as glioma |
| `meningioma` | MRI images labeled as meningioma |
| `notumor` | MRI images labeled as no tumor |
| `pituitary` | MRI images labeled as pituitary tumor |

The dataset contains separate training and testing directories. Image counts may vary depending on the dataset version.

### Dataset Structure

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

Download the dataset from Kaggle and make it available in your Google Colab environment before training.

## Project Architecture

```text
Brain MRI Image
      |
      v
Image Preprocessing
(Resize to 128 x 128)
      |
      v
Pixel Normalization
      |
      v
Convolutional Layer
      |
      v
Max Pooling
      |
      v
Convolutional Layer
      |
      v
Max Pooling
      |
      v
Convolutional Layer
      |
      v
Max Pooling
      |
      v
Global Average Pooling
      |
      v
Dense Layer + Dropout
      |
      v
Softmax Output Layer
      |
      v
Predicted Dataset Category
```

## CNN Model

The model consists of the following layers:

1. **Input Layer:** Accepts RGB images of size `128 × 128 × 3`.
2. **Rescaling Layer:** Normalizes pixel values to the range 0–1.
3. **Convolutional Layers:** Extract image features such as edges, textures, and shapes.
4. **Max Pooling Layers:** Reduce spatial dimensions.
5. **Global Average Pooling:** Summarizes learned feature maps.
6. **Dense Layer:** Learns classification patterns.
7. **Dropout Layer:** Helps reduce overfitting.
8. **Output Layer:** Produces scores for the four dataset classes.

The model uses the Adam optimizer and sparse categorical cross-entropy loss.

## Getting Started

### Option 1: Run in Google Colab

Google Colab is the recommended environment for this project.

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new Python notebook.
3. Download the dataset from Kaggle.
4. Upload or download the dataset into the Colab runtime.
5. Execute the preprocessing, model-building, training, and evaluation cells in order.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Brain-MRI-Tumor-Classification-CNN.git
```

Navigate to the project directory:

```bash
cd Brain-MRI-Tumor-Classification-CNN
```

Create and activate a virtual environment if desired.

Install the dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

Place the dataset in the appropriate directory, then run your training script or notebook.

> Update the commands to match the actual filenames and folder structure in your repository.

## Training

The model is trained using labeled MRI images.

Example training configuration:

| Parameter | Value |
|---|---|
| Image size | 128 × 128 |
| Batch size | 32 |
| Maximum epochs | 15 |
| Optimizer | Adam |
| Loss function | Sparse categorical cross-entropy |
| Output activation | Softmax |
| Early stopping | Used in the initial experiment |

Training accuracy and validation accuracy can be monitored during training. Early stopping may terminate training before the maximum number of epochs when validation loss stops improving.

## Evaluation Metrics

The project can be evaluated using:

- **Accuracy:** Overall classification performance.
- **Precision:** Proportion of predicted class assignments that are correct.
- **Recall:** Proportion of actual class examples identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Displays correct and incorrect predictions for each class.

Actual performance should be measured on a separate held-out test set. Record the final test metrics after completing the evaluation; do not assume a particular accuracy before running the experiment.

## Prediction

The trained model accepts an MRI image and returns scores for the four classes.

Example output:

```text
Predicted class: meningioma
Model score: [value returned by the model]
```

This is an illustrative example, not a guaranteed prediction or a verified medical conclusion. The model's output score is not a calibrated probability of disease.

## Future Improvements

- Add data augmentation to improve generalization.
- Use a separate validation set during model training.
- Compare a custom CNN with transfer-learning models such as MobileNetV2.
- Add a Streamlit interface for image upload and prediction.
- Visualize learned features using Grad-CAM.
- Evaluate performance on independent datasets.
- Investigate class imbalance and possible dataset leakage.

## Limitations

- Performance depends on the quality and distribution of the training dataset.
- The model may misclassify images from different sources or imaging protocols.
- Dataset labels do not necessarily represent independent clinical verification.
- A high prediction score does not establish diagnostic reliability.
- Clinical use would require extensive external validation and appropriate medical oversight.

## Repository Structure

A suggested structure is:

```text
Brain-MRI-Tumor-Classification-CNN/
├── README.md
├── notebooks/
│   └── brain_mri_cnn.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── .gitignore
└── LICENSE
```

Adjust this structure to match the files you actually commit. Do not upload large datasets or generated model files unless necessary and appropriate for the repository.

## Responsible Use

This repository demonstrates deep learning and image classification concepts. It is not intended for clinical decision-making, patient treatment, or medical diagnosis.

MRI interpretation should be performed by qualified healthcare professionals.

## Author

**Suprakash Ghosh**

B.Tech — Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning

GitHub: https://github.com/SUPRAJ29DB

If you find this project useful for learning, consider starring the repository.
