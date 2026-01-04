# ASL Real-Time Recognition System

A real-time American Sign Language (ASL) recognition system that uses a CNN to predict hand signs from your camera feed and types the predicted characters.

## Dataset

The improved dataset combines 4 different ASL datasets for better model performance:

1. **Original ASL dataset** (`dataset/asl/`) - Contains lowercase letters and numbers
2. **Kaggle ASL Alphabet Dataset 1** (`dataset/alphabet and numbers 1/`) - From [debashishsau/aslamerican-sign-language-aplhabet-dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)
3. **Kaggle ASL Alphabet Dataset 2** (`dataset/alphabet and numbers 2/`) - From [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
4. **Kaggle ASL Alphabet Train** (`dataset/asl_alphabet_train/`) - Additional training data

The combined dataset contains multiple classes (digits 0-9, letters A-Z, and special classes like "nothing", "space", "unknown", "del") with significantly more images than the original dataset. The training script automatically:

- Combines all datasets
- Normalizes class names (lowercase letters → uppercase)
- Handles different image formats (.jpg, .jpeg, .png)
- Provides detailed statistics about the combined dataset

## Setup

1. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

Train the CNN model on your ASL dataset:

```bash
python train_model.py
```

This will:

- Load and combine images from all 4 datasets in the `dataset/` directory
- Normalize class names across datasets (e.g., 'a' and 'A' → 'A')
- Split data into training and validation sets
- Train a CNN model with data augmentation
- Save the trained model as `asl_model.keras` (or `asl_model.h5`)
- Save class names as `class_names.pkl`
- Generate training history plots
- Display detailed statistics about the combined dataset

### Step 2: Run Real-Time Recognition

After training, run the real-time camera application:

```bash
python real_time_prediction.py
```

**Controls:**

- Show your hand sign in the center of the camera view
- The system will automatically predict and type characters
- Press `q` to quit
- Press `c` to clear the typed text
- Press `s` to display current text in console

## Model Architecture

The CNN model consists of:

- 4 convolutional blocks with batch normalization and dropout
- Max pooling layers for dimensionality reduction
- Dense layers for classification
- Multiple output classes (0-9, A-Z, and special classes like "nothing", "space", "unknown", "del")

## Features

- Real-time camera feed processing
- Confidence threshold filtering
- Prediction buffering for stability
- Automatic character typing
- Visual feedback with ROI rectangle
- Mirror mode for natural interaction

## Notes

- Make sure your camera is connected and working
- Position your hand in the center of the camera view
- The model works best with good lighting and clear hand gestures
- Adjust `CONFIDENCE_THRESHOLD` in `real_time_prediction.py` if needed
