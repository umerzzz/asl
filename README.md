# ASL Real-Time Recognition System

A real-time American Sign Language (ASL) recognition system that uses a CNN to predict hand signs from your camera feed and types the predicted characters.

## Dataset

The dataset contains 36 classes (digits 0-9 and letters a-z) with approximately 25,000+ images organized in folders by class.

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
- Load all images from the `asl/` directory
- Split data into training and validation sets
- Train a CNN model with data augmentation
- Save the trained model as `asl_model.h5`
- Save class names as `class_names.pkl`
- Generate training history plots

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
- 36 output classes (0-9, a-z)

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

