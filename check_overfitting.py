import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import cv2
import os
import glob
import pickle
from collections import Counter

# Configuration
IMG_SIZE = 64
BASE_DATA_DIR = 'dataset'
CLASS_NAMES_PATH = 'class_names.pkl'

# Define all dataset paths
DATASET_PATHS = [
    'dataset/asl',  # Original dataset with lowercase letters
    'dataset/alphabet and numbers 1',  # First Kaggle dataset
    'dataset/alphabet and numbers 2',  # Second Kaggle dataset
    'dataset/asl_alphabet_train/asl_alphabet_train',  # Third Kaggle dataset (nested)
]

def normalize_class_name(class_name):
    """Normalize class names: convert lowercase letters to uppercase, keep special classes as-is"""
    # If it's a single lowercase letter, convert to uppercase
    if len(class_name) == 1 and class_name.isalpha() and class_name.islower():
        return class_name.upper()
    # Keep numbers and special classes (nothing, space, unknown, del) as-is
    return class_name

def load_data_from_directory(dataset_path, class_names, images, labels):
    """Load images from a single dataset directory using predefined class names"""
    if not os.path.exists(dataset_path):
        print(f"  [SKIP] Dataset path not found: {dataset_path}")
        return
    
    print(f"  Loading from: {dataset_path}")
    
    # Get all class directories
    if not os.path.isdir(dataset_path):
        return
    
    classes = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Exclude nested 'asl' folder if it exists
    if 'asl' in classes and os.path.isdir(os.path.join(dataset_path, 'asl')):
        # Check if 'asl' folder contains more folders (it's a nested structure)
        asl_path = os.path.join(dataset_path, 'asl')
        nested_classes = [d for d in os.listdir(asl_path) 
                         if os.path.isdir(os.path.join(asl_path, d))]
        if nested_classes:
            # Skip the 'asl' folder, it's just a container
            classes = [c for c in classes if c != 'asl']
    
    # Create mapping from class name to index
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Normalize class name
        normalized_name = normalize_class_name(class_name)
        
        # Only load if this class exists in the saved class names
        if normalized_name not in class_to_idx:
            continue
        
        class_idx = class_to_idx[normalized_name]
        
        # Load images (support both .jpg and .jpeg)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        loaded_count = 0
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Resize and normalize
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(class_idx)
                loaded_count += 1
        
        if loaded_count > 0:
            print(f"    Class '{class_name}' -> '{normalized_name}': {loaded_count} images")

def load_data(class_names):
    """Load images and labels from all dataset directories using predefined class names"""
    images = []
    labels = []
    
    print("="*70)
    print("LOADING DATA FROM MULTIPLE DATASETS")
    print("="*70)
    print(f"Using class names from training: {len(class_names)} classes")
    
    # Load from all dataset paths
    for dataset_path in DATASET_PATHS:
        load_data_from_directory(dataset_path, class_names, images, labels)
    
    print("\n" + "="*70)
    print(f"TOTAL DATASET SUMMARY")
    print("="*70)
    print(f"Total images loaded: {len(images)}")
    print(f"Total classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Print class distribution
    label_counts = Counter(labels)
    print("\nClass distribution:")
    for idx, class_name in enumerate(class_names):
        count = label_counts.get(idx, 0)
        print(f"  {class_name}: {count} images")
    
    return np.array(images), np.array(labels), class_names

def analyze_model():
    """Analyze the trained model for overfitting"""
    print("="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)
    
    # Load class names first (must match training)
    print("\n0. Loading class names from training...")
    try:
        with open(CLASS_NAMES_PATH, 'rb') as f:
            class_names = pickle.load(f)
        print(f"   [OK] Loaded {len(class_names)} classes: {class_names}")
    except Exception as e:
        print(f"   [ERROR] Could not load class names: {e}")
        return
    
    # Load model
    print("\n1. Loading model...")
    try:
        model = keras.models.load_model('asl_model.h5')
        print("   [OK] Model loaded successfully")
    except:
        try:
            model = keras.models.load_model('asl_model.keras')
            print("   [OK] Model loaded successfully")
        except Exception as e:
            print(f"   [ERROR] Could not load model: {e}")
            return
    
    # Verify model output size matches class count
    try:
        model_output_size = model.output_shape[-1]
    except:
        # Fallback: get from last layer's units
        model_output_size = model.layers[-1].units if hasattr(model.layers[-1], 'units') else None
    
    if model_output_size and model_output_size != len(class_names):
        print(f"   [WARNING] Model expects {model_output_size} classes but class_names has {len(class_names)}")
        print(f"   This may cause issues. Please ensure you're using the correct model.")
    elif model_output_size:
        print(f"   [OK] Model output size ({model_output_size}) matches class count ({len(class_names)})")
    
    # Load data
    print("\n2. Loading dataset...")
    images, labels, class_names = load_data(class_names)
    print(f"   [OK] Loaded {len(images)} images")
    
    # Split data (same as training)
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"   [OK] Training samples: {len(X_train)}")
    print(f"   [OK] Validation samples: {len(X_val)}")
    
    # Evaluate on training set
    print("\n3. Evaluating on training set...")
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0, batch_size=32)
    print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   Training Loss: {train_loss:.4f}")
    
    # Evaluate on validation set
    print("\n4. Evaluating on validation set...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0, batch_size=32)
    print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"   Validation Loss: {val_loss:.4f}")
    
    # Calculate gap
    print("\n5. Overfitting Analysis:")
    print("   " + "-"*50)
    accuracy_gap = train_accuracy - val_accuracy
    loss_gap = val_loss - train_loss
    
    print(f"   Accuracy Gap (Train - Val): {accuracy_gap:.4f}")
    print(f"   Loss Gap (Val - Train): {loss_gap:.4f}")
    
    # Interpretation
    print("\n6. Interpretation:")
    print("   " + "-"*50)
    
    if accuracy_gap > 0.10:  # More than 10% gap
        print("   [WARNING] Significant overfitting detected!")
        print(f"      Training accuracy is {accuracy_gap*100:.1f}% higher than validation.")
        print("      The model is memorizing training data.")
    elif accuracy_gap > 0.05:  # 5-10% gap
        print("   [CAUTION] Mild overfitting detected.")
        print(f"      Training accuracy is {accuracy_gap*100:.1f}% higher than validation.")
        print("      Consider regularization or early stopping.")
    elif accuracy_gap < -0.05:  # Validation higher than training
        print("   [GOOD] Validation accuracy is higher than training.")
        print("      This suggests data augmentation is working well.")
        print("      The model generalizes better than it fits training data.")
    else:  # Small gap
        print("   [EXCELLENT] Model is well-generalized!")
        print(f"      Small gap ({accuracy_gap*100:.1f}%) indicates good balance.")
        print("      The model learns patterns, not memorizes data.")
    
    # Additional metrics
    print("\n7. Additional Metrics:")
    print("   " + "-"*50)
    print(f"   Training Set Size: {len(X_train)}")
    print(f"   Validation Set Size: {len(X_val)}")
    print(f"   Number of Parameters: {model.count_params():,}")
    print(f"   Model Complexity: {'High' if model.count_params() > 1_000_000 else 'Moderate'}")
    
    # Recommendations
    print("\n8. Recommendations:")
    print("   " + "-"*50)
    if accuracy_gap > 0.10:
        print("   • Increase dropout rates")
        print("   • Add more data augmentation")
        print("   • Reduce model complexity")
        print("   • Use early stopping more aggressively")
    elif accuracy_gap > 0.05:
        print("   • Slightly increase regularization")
        print("   • Monitor validation metrics closely")
    else:
        print("   • Model is performing well!")
        print("   • Current settings are appropriate")
        print("   • Ready for deployment")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    analyze_model()

