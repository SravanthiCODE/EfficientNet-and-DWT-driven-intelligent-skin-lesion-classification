# EfficientNet-and-DWT-driven-intelligent-skin-lesion-classification
EfficientNet B3 a DL model has been trained on HAM10000 dataset, DWT and data agumentation techniques were applied to achieve 89% of accuracy.

This project presents a deep learning approach for binary classification of skin lesions (benign vs malignant) using the HAM10000 dataset. The methodology integrates **Discrete Wavelet Transform (DWT)** for preprocessing and **EfficientNetB3** as the backbone model to enhance diagnostic accuracy.

## Project Highlights
- Dataset: [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Model: EfficientNetB3 (fine-tuned with transfer learning)
- Preprocessing: Grayscale conversion, Gaussian blur, and 2D Haar wavelet transform (DWT)
- Augmentation: Random rotations, flips, zooms
- Evaluation Metrics: Accuracy (0.8907), Precision (0.7700), Recall (0.6278), F1 Score (0.6917)

# Content
- `skin_cancer_diagnosis.py`: Python script containing full pipeline including:
  - Dataset loading and label mapping
  - Preprocessing with DWT (Discrete Wavelet Transform)
  - Data augmentation
  - EfficientNetB3 model training and evaluation


