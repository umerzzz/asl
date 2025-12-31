import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import cv2
import os
import glob

# Configuration
IMG_SIZE = 64
DATA_DIR = 'asl'

def load_data(data_dir):
    """Load images and labels from directory structure"""
    images = []
    labels = []
    class_names = []
    
    classes = sorted([d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d)) and d != 'asl'])
    
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_path) if f.endswith('.jpeg')]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(idx)
        
        class_names.append(class_name)
    
    return np.array(images), np.array(labels), class_names

def analyze_model():
    """Analyze the trained model for overfitting"""
    print("="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    try:
        model = keras.models.load_model('asl_model.h5')
        print("   [OK] Model loaded successfully")
    except:
        print("   [ERROR] Could not load model")
        return
    
    # Load data
    print("\n2. Loading dataset...")
    images, labels, class_names = load_data(DATA_DIR)
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

