# Gait-Based Subject Identification (WISDM)  

This project trains a neural network to identify **which person is walking** using short windows of smartphone **gyroscope signals** from the WISDM dataset.

Unlike traditional Human Activity Recognition (HAR), which classifies *what* someone is doing, this project focuses on **biometric identification**, distinguishing *who* is walking based on gait dynamics.

## Problem Framing

Human gait is naturally periodic and biomechanically constrained.  
A full **gait cycle** typically lasts ~1.0–1.2 seconds and exhibits highly individual patterns.

**Hypothesis:**  
A short time window covering approximately one gait cycle contains enough dynamic information to distinguish individuals.

To test this, the raw sensor stream is segmented into ~1-second windows and used to train a subject classifier.

## Dataset

- **Dataset:** WISDM (Weiss et al.)
- **Source:** Kaggle
- **Devices available:** smartphone, smartwatch
- **Sensors available:** accelerometer, gyroscope
- **This implementation uses:** **smartphone gyroscope only**
- **Activity used:** walking (`"A"`)
- **Sampling rate:** ~20 Hz (minor variation across recordings)

The dataset is automatically downloaded via **KaggleHub** when the application runs, no manual dataset configuration is required.

## Windowing Strategy

- **Window size:** 20 samples (~1 second)
- **Step size:** 10 samples (50% overlap)

This produces approximately **360–400 windows per subject**, depending on recording duration.

## Data Splitting

Windows are split using **stratified sampling**:

- **Train:** 72%
- **Validation:** 18%
- **Test:** 10%

Stratification ensures each subject is represented across splits, preventing class imbalance artifacts.

## Model Architecture

The model combines **convolutional feature extraction** with **temporal modeling**.

### 1D Convolutional Layers
- Capture short-term local motion patterns
- Extract structured features from raw angular velocity signals
- Followed by BatchNorm + ReLU

### LSTM Stack
- Models sequential dependencies within each window
- Captures temporal evolution across the gait cycle
- Includes dropout (`0.1`) for regularization

### Fully Connected Layers
- LayerNorm regularization
- Final projection into subject ID space

## Optimization

- **Loss:** Cross-Entropy
- **Optimizer:** AdamW
- **Scheduler:** OneCycleLR
- **Regularization:**  
  - Weight decay  
  - LSTM dropout  
  - BatchNorm + LayerNorm  

Training is seed-controlled for reproducibility.

## Experimental Results

Observed behavior during experimentation:

| Model Variant | Validation Accuracy |
|---------------|--------------------|
| Dense-only baseline | Poor performance |
| LSTM-only | Improved training accuracy but overfit |
| LSTM + AdamW + dropout | ~90% |
| **CNN + LSTM (final model)** | **~97% after 10 epochs** |

> Results may vary slightly depending on random seed and hardware.

## Project Structure

```
src/
 ├── data/        # Dataset loading, filtering, windowing
 ├── models/      # CNN + LSTM model definition
 ├── training/    # Training loop & evaluation
 ├── main.py      # Application entry point
```

## Quickstart

### 1) Install dependencies (Poetry)

```bash
poetry install
```

### 2) Run the application

```bash
poetry run python -m src.main
```

The dataset will be downloaded automatically if not already present.

## Author

**Dio Ngei Okparaji**  
GitHub: https://github.com/Dio358
