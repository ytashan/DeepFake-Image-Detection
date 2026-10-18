# AI-Powered Deepfake Image Detection

An AI-based image classification project that uses **EfficientNetB0 and Transfer Learning** to distinguish real images from AI-generated or deepfake images. The project is implemented in Python using TensorFlow and Keras, with support for model fine-tuning, performance evaluation, and individual or batch image prediction.

## Overview

The rapid development of generative AI has made synthetic images increasingly realistic. This project explores deep learning techniques for identifying potentially AI-generated or manipulated images through binary image classification.

The model is trained using the CIFAKE dataset and can also be evaluated on a separate Deepfake Image Detection dataset to examine its performance on different image sources.

## Key Features

- **EfficientNetB0 Architecture:** Uses an ImageNet-pretrained convolutional neural network for image feature extraction.
- **Transfer Learning:** Freezes the pretrained backbone during the initial training phase.
- **Fine-Tuning:** Unfreezes the last 30 layers of the backbone for additional training.
- **Data Augmentation:** Applies rotation, shifting, shearing, zooming, and horizontal flipping.
- **Binary Classification:** Predicts whether an image belongs to the real or fake class.
- **Model Checkpointing:** Saves the best model according to validation AUC.
- **Performance Evaluation:** Calculates accuracy, loss, AUC, classification report, confusion matrix, and ROC curve.
- **Single-Image Prediction:** Classifies an individual image and displays its predicted label and confidence.
- **Batch Prediction:** Processes multiple images from a directory.

## Model Architecture

The project uses EfficientNetB0 as the pretrained feature extractor, followed by a custom classification head.

**Architecture:**

```text
Input Image (224 × 224 × 3)
          |
          v
EfficientNetB0 (ImageNet Weights)
          |
          v
Global Average Pooling
          |
          v
Dropout (0.3)
          |
          v
Dense Layer (256 Units, ReLU)
          |
          v
Dropout (0.3)
          |
          v
Dense Layer (1 Unit, Sigmoid)
          |
          v
Real / Fake Classification
```

The sigmoid output represents the model's score for the positive class. Its interpretation as a fake-image probability depends on the dataset's class-to-index mapping.

## Datasets

This project uses two datasets configured in the notebook:

### 1. CIFAKE: Real and AI-Generated Synthetic Images

Used for model training, validation, and primary testing.

- Training data: CIFAKE training directory
- Validation: 15% of the training directory
- Testing: Separate CIFAKE test directory

Dataset: [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

### 2. Deepfake Image Detection

Used for optional evaluation on a separate image dataset and for sample fake-image predictions.

Dataset: [Deepfake Image Detection](https://www.kaggle.com/datasets/saurabhbagchi/deepfake-image-detection)

**Important:** The two datasets may contain different image-generation methods and distributions. Performance on one dataset does not guarantee the same performance on another.

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Kaggle Notebooks

## Project Configuration

| Parameter | Value |
|---|---|
| Input image size | 224 × 224 |
| Batch size | 32 |
| Initial training epochs | Up to 15 |
| Fine-tuning epochs | Up to 8 |
| Initial learning rate | 0.0001 |
| Fine-tuning learning rate | 0.00001 |
| Optimizer | Adam |
| Loss function | Binary cross-entropy |
| Main metrics | Accuracy and AUC |
| Validation split | 15% |
| Random seed | 42 |
| Prediction threshold | 0.5 |

The training duration may be shorter because of early stopping.

## Installation

Install the required Python packages:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

For GPU acceleration, configure a compatible TensorFlow and CUDA environment for your operating system and hardware.

## How to Run

### Step 1: Open the Notebook

Upload `deepfake-image.ipynb` to Kaggle Notebooks or open it in a compatible Jupyter environment.

### Step 2: Configure the Datasets

The notebook currently uses Kaggle-specific dataset paths. Update the path variables if your directory structure differs.

```python
BASE = "/kaggle/input/datasets"
```

Ensure that the training and test directories contain the appropriate class subdirectories.

### Step 3: Train the Model

Run the cells in order to initialize the data generators, build the EfficientNetB0 model, compile it, and start training.

### Step 4: Fine-Tune the Model

The notebook subsequently unfreezes the last 30 layers of EfficientNetB0 and continues training with a lower learning rate.

### Step 5: Evaluate the Model

The evaluation cells generate:

- Test accuracy and loss
- ROC-AUC score
- Classification report
- Confusion matrix
- ROC curve
- Training and validation history plots

### Step 6: Predict an Image

After loading the saved model, use the prediction helper:

```python
label, confidence, probability = predict_image(
    "/path/to/your/image.jpg"
)

print("Prediction:", label)
print("Confidence:", confidence)
print("Fake score:", probability)
```

Replace the example path with the location of your image.

The notebook also includes `predict_folder()` for batch prediction on supported image formats.

## Model Saving

The best checkpoint is saved to:

```text
/kaggle/working/models/best_deepfake_detector.keras
```

This path is specific to the Kaggle environment. Download the checkpoint or change the path if you want to use the model outside Kaggle.

## Evaluation Metrics

The notebook uses the following metrics to assess model performance:

- **Accuracy:** Overall proportion of correctly classified images.
- **AUC:** Measures discrimination between the two classes across classification thresholds.
- **Precision:** Measures how often predictions for a class are correct.
- **Recall:** Measures how many actual examples of a class are identified.
- **F1-score:** Combines precision and recall.
- **Confusion Matrix:** Summarizes correct and incorrect predictions.
- **ROC Curve:** Illustrates the relationship between true-positive and false-positive rates.

Actual performance values should be taken from the completed notebook execution.

## Limitations

- The classifier's generalization depends on the datasets used during training and evaluation.
- Images with unseen generation techniques or substantial post-processing may be misclassified.
- A high confidence score does not guarantee that a prediction is correct.
- The model is an experimental detection tool and should not be treated as definitive proof of image authenticity.
- The notebook's standalone prediction helper scales images to the range [0, 1], while training uses EfficientNetB0's `preprocess_input`. For consistent inference, the preprocessing used during prediction should match the model's training pipeline.
- Class labels and the meaning of the sigmoid output should be verified against the actual directory-to-class mapping.

## Future Improvements

- Evaluate on additional unseen AI-generated image datasets.
- Apply cross-dataset validation and robustness testing.
- Investigate image compression and resizing effects.
- Add a web interface using Flask or Streamlit.
- Support image uploads and batch inference through a user-friendly interface.
- Explore explainability techniques such as Grad-CAM.
- Report per-class precision, recall, F1-score, and calibration results.

## Repository Structure

```text
AI-Deepfake-Image-Detection/
├── README.md
├── deepfake-image.ipynb
├── models/
│   └── best_deepfake_detector.keras
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── training_history.png
```

The `models/` and `results/` directories are suggested for organizing exported artifacts. They are not automatically created in this exact repository structure by the notebook.

## Disclaimer

This project is intended for educational and research purposes. Its predictions should be interpreted cautiously and should not be used as the sole basis for determining whether an image is authentic or manipulated.

## Author

**Md. Osiur Rahman Rahif**

GitHub: [Your GitHub Profile](https://github.com/)

## License

Choose an appropriate open-source license before distributing the project. Also review the licenses and usage conditions of the datasets and pretrained model weights.
