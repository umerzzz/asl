"""
Diagnostic script to identify the root cause of poor model performance.
Checks for:
1. Class distribution imbalance
2. Model architecture vs expected classes
3. Preprocessing consistency
4. Training data availability
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import cv2
from collections import Counter
from pathlib import Path

# Configuration
IMG_SIZE = 64
MODEL_PATH = 'asl_model.h5'
CLASS_NAMES_PATH = 'class_names.pkl'
BASE_DATA_DIR = 'dataset'

# Define all dataset paths
DATASET_PATHS = [
    'dataset/asl',  # Original dataset with lowercase letters
    'dataset/alphabet and numbers 1',  # First Kaggle dataset
    'dataset/alphabet and numbers 2',  # Second Kaggle dataset
    'dataset/asl_alphabet_train/asl_alphabet_train',  # Third Kaggle dataset (nested)
]

def normalize_class_name(class_name):
    """Normalize class names: convert lowercase letters to uppercase, keep special classes as-is"""
    if len(class_name) == 1 and class_name.isalpha() and class_name.islower():
        return class_name.upper()
    return class_name

def check_class_distribution():
    """Check the distribution of classes in the dataset"""
    print("="*70)
    print("CHECKING CLASS DISTRIBUTION IN DATASET")
    print("="*70)
    
    class_counts = Counter()
    class_files = {}
    
    for dataset_path in DATASET_PATHS:
        if not os.path.exists(dataset_path):
            print(f"  [SKIP] Dataset path not found: {dataset_path}")
            continue
        
        print(f"\n  Checking: {dataset_path}")
        if not os.path.isdir(dataset_path):
            continue
        
        classes = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
        
        # Exclude nested 'asl' folder if it exists
        if 'asl' in classes and os.path.isdir(os.path.join(dataset_path, 'asl')):
            asl_path = os.path.join(dataset_path, 'asl')
            nested_classes = [d for d in os.listdir(asl_path) 
                             if os.path.isdir(os.path.join(asl_path, d))]
            if nested_classes:
                classes = [c for c in classes if c != 'asl']
        
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue
            
            normalized_name = normalize_class_name(class_name)
            
            # Count images
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            class_counts[normalized_name] += len(image_files)
            
            if normalized_name not in class_files:
                class_files[normalized_name] = []
            class_files[normalized_name].extend([os.path.join(class_path, f) for f in image_files])
    
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION SUMMARY")
    print("="*70)
    
    if len(class_counts) == 0:
        print("[ERROR] No classes found in dataset!")
        return None, None
    
    # Sort classes
    sorted_classes = sorted(class_counts.keys(), key=lambda x: (
        (0, int(x)) if x.isdigit() else (1, x) if x.isalpha() else (2, x)
    ))
    
    print(f"\nTotal classes found: {len(sorted_classes)}")
    print(f"Total images: {sum(class_counts.values())}")
    print("\nClass distribution:")
    print("-" * 70)
    print(f"{'Class':<12} {'Count':<10} {'Percentage':<12}")
    print("-" * 70)
    
    total = sum(class_counts.values())
    for class_name in sorted_classes:
        count = class_counts[class_name]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{class_name:<12} {count:<10} {percentage:>6.2f}%")
    
    # Check for imbalance
    counts_list = list(class_counts.values())
    min_count = min(counts_list)
    max_count = max(counts_list)
    avg_count = np.mean(counts_list)
    
    print("\n" + "-" * 70)
    print("IMBALANCE ANALYSIS:")
    print(f"  Minimum samples per class: {min_count}")
    print(f"  Maximum samples per class: {max_count}")
    print(f"  Average samples per class: {avg_count:.1f}")
    print(f"  Ratio (max/min): {max_count/min_count:.2f}x" if min_count > 0 else "  Ratio: N/A (some classes have 0 samples)")
    
    if min_count == 0:
        zero_classes = [c for c, count in class_counts.items() if count == 0]
        print(f"\n  [WARNING] {len(zero_classes)} classes have 0 images:")
        for c in zero_classes:
            print(f"    - {c}")
    
    if max_count / min_count > 10 and min_count > 0:
        print(f"\n  [WARNING] Severe class imbalance detected!")
        print(f"    Some classes have {max_count/min_count:.1f}x more samples than others.")
    
    return sorted_classes, class_files

def check_model_architecture():
    """Check the model architecture and output size"""
    print("\n" + "="*70)
    print("CHECKING MODEL ARCHITECTURE")
    print("="*70)
    
    # Load class names
    try:
        with open(CLASS_NAMES_PATH, 'rb') as f:
            saved_class_names = pickle.load(f)
        print(f"\n[OK] Loaded class names from training: {len(saved_class_names)} classes")
        print(f"     Classes: {saved_class_names}")
    except Exception as e:
        print(f"\n[ERROR] Could not load class names: {e}")
        return None, None
    
    # Load model
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"\n[OK] Model loaded successfully")
    except:
        try:
            alt_path = MODEL_PATH.replace('.h5', '.keras')
            model = keras.models.load_model(alt_path)
            print(f"\n[OK] Model loaded successfully (from .keras file)")
        except Exception as e:
            print(f"\n[ERROR] Could not load model: {e}")
            return None, None
    
    # Check model output
    try:
        model_output_size = model.output_shape[-1]
        print(f"\n[OK] Model output size: {model_output_size} classes")
    except:
        try:
            model_output_size = model.layers[-1].units
            print(f"\n[OK] Model output size (from last layer): {model_output_size} classes")
        except:
            print(f"\n[WARNING] Could not determine model output size")
            model_output_size = None
    
    # Check if sizes match
    if model_output_size:
        if model_output_size != len(saved_class_names):
            print(f"\n[CRITICAL ERROR] Model output size ({model_output_size}) does NOT match")
            print(f"                saved class names count ({len(saved_class_names)})!")
            print(f"                This will cause prediction errors!")
        else:
            print(f"\n[OK] Model output size matches saved class names count")
    
    # Model summary
    print("\n" + "-" * 70)
    print("MODEL SUMMARY:")
    print("-" * 70)
    total_params = model.count_params()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    return model, saved_class_names

def check_preprocessing_consistency():
    """Check if preprocessing is consistent across scripts"""
    print("\n" + "="*70)
    print("CHECKING PREPROCESSING CONSISTENCY")
    print("="*70)
    
    # Check training script
    train_preprocessing = """
    Training (train_model.py):
      - cv2.imread() -> BGR format
      - cv2.resize(img, (64, 64))
      - img.astype('float32') / 255.0
      - NO color space conversion
    """
    
    # Check verification script
    verify_preprocessing = """
    Verification (verify_model.py):
      - cv2.imread() -> BGR format
      - cv2.resize(img, (64, 64))
      - img.astype('float32') / 255.0
      - NO color space conversion (correct)
    """
    
    # Check real-time script
    realtime_preprocessing = """
    Real-time (real_time_prediction.py):
      - cv2.resize(frame, (64, 64))
      - cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  <-- CONVERTS TO RGB!
      - normalized.astype('float32') / 255.0
      - ISSUE: Converts BGR to RGB (mismatch with training)
    """
    
    print(train_preprocessing)
    print(verify_preprocessing)
    print(realtime_preprocessing)
    print("\n[WARNING] Real-time prediction script converts BGR to RGB,")
    print("          but training uses BGR format. This is a mismatch!")
    print("          However, this doesn't explain the verification script's poor performance.")

def test_model_on_sample_images(model, class_names, dataset_classes, class_files):
    """Test the model on a few sample images"""
    print("\n" + "="*70)
    print("TESTING MODEL ON SAMPLE IMAGES")
    print("="*70)
    
    if model is None or class_names is None:
        print("[SKIP] Cannot test - model or class names not loaded")
        return
    
    # Find classes that exist in both model and dataset
    model_class_set = set(class_names)
    dataset_class_set = set(dataset_classes) if dataset_classes else set()
    
    common_classes = model_class_set & dataset_class_set
    missing_in_dataset = model_class_set - dataset_class_set
    missing_in_model = dataset_class_set - model_class_set
    
    print(f"\nClass comparison:")
    print(f"  Classes in model: {len(model_class_set)}")
    print(f"  Classes in dataset: {len(dataset_class_set)}")
    print(f"  Common classes: {len(common_classes)}")
    print(f"  Classes in model but NOT in dataset: {len(missing_in_dataset)}")
    if missing_in_dataset:
        print(f"    {sorted(missing_in_dataset)}")
    print(f"  Classes in dataset but NOT in model: {len(missing_in_model)}")
    if missing_in_model:
        print(f"    {sorted(missing_in_model)}")
    
    if len(common_classes) == 0:
        print("\n[CRITICAL ERROR] No common classes between model and dataset!")
        print("                This explains the poor performance!")
        return
    
    # Test on a few samples from common classes
    print(f"\nTesting on samples from common classes...")
    test_count = 0
    correct_count = 0
    
    for class_name in sorted(common_classes)[:5]:  # Test first 5 common classes
        if class_name not in class_files or len(class_files[class_name]) == 0:
            continue
        
        # Get a sample image
        sample_path = class_files[class_name][0]
        img = cv2.imread(sample_path)
        
        if img is None:
            continue
        
        # Preprocess as in training
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        predictions = model.predict(img_batch, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else "UNKNOWN"
        
        # Get true class index
        true_idx = class_names.index(class_name) if class_name in class_names else -1
        
        is_correct = (predicted_idx == true_idx)
        if is_correct:
            correct_count += 1
        test_count += 1
        
        status = "[OK]" if is_correct else "[FAIL]"
        print(f"  {status} {class_name} -> {predicted_class} (conf: {confidence:.2%}, idx: {predicted_idx}/{true_idx})")
    
    if test_count > 0:
        accuracy = correct_count / test_count
        print(f"\n  Sample test accuracy: {accuracy:.2%} ({correct_count}/{test_count})")

def main():
    print("="*70)
    print("MODEL DIAGNOSTIC ANALYSIS")
    print("="*70)
    print("\nThis script will diagnose the root cause of poor model performance.")
    print("Checking: class distribution, model architecture, preprocessing, etc.\n")
    
    # 1. Check class distribution
    dataset_classes, class_files = check_class_distribution()
    
    # 2. Check model architecture
    model, saved_class_names = check_model_architecture()
    
    # 3. Check preprocessing consistency
    check_preprocessing_consistency()
    
    # 4. Test model on samples
    test_model_on_sample_images(model, saved_class_names, dataset_classes, class_files)
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    if model is None:
        print("\n[ERROR] Could not load model. Please check if model file exists.")
        return
    
    if saved_class_names is None:
        print("\n[ERROR] Could not load class names. Please check if class_names.pkl exists.")
        return
    
    if dataset_classes is None:
        print("\n[ERROR] Could not find dataset. Please check dataset paths.")
        return
    
    # Check for mismatches
    model_class_set = set(saved_class_names)
    dataset_class_set = set(dataset_classes)
    
    if model_class_set != dataset_class_set:
        print("\n[CRITICAL] Class mismatch detected!")
        print(f"  Model classes: {sorted(model_class_set)}")
        print(f"  Dataset classes: {sorted(dataset_class_set)}")
        print("\n  This is likely the root cause of poor performance!")
        print("  The model was trained on different classes than what's being tested.")
    else:
        print("\n[OK] Class sets match between model and dataset")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()

