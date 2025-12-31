import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random
from pathlib import Path

# Configuration
IMG_SIZE = 64
MODEL_PATH = 'asl_model.h5'
CLASS_NAMES_PATH = 'class_names.pkl'
DATA_DIR = 'asl'
NUM_TEST_SAMPLES = 20  # Number of random samples to test
NUM_PER_CLASS = 3  # Number of samples per class to show

def load_model():
    """Load the trained model"""
    print("Loading model...")
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("[OK] Model loaded successfully!")
        return model
    except:
        try:
            alt_path = MODEL_PATH.replace('.h5', '.keras')
            model = keras.models.load_model(alt_path)
            print("[OK] Model loaded successfully!")
            return model
        except:
            try:
                alt_path = MODEL_PATH.replace('.keras', '.h5')
                model = keras.models.load_model(alt_path)
                print("[OK] Model loaded successfully!")
                return model
            except Exception as e:
                raise FileNotFoundError(f"Could not load model: {e}")

def load_class_names():
    """Load class names"""
    with open(CLASS_NAMES_PATH, 'rb') as f:
        class_names = pickle.load(f)
    return class_names

def load_test_images(data_dir, class_names, num_per_class=3):
    """Load random test images from each class"""
    test_images = []
    test_labels = []
    test_paths = []
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        image_files = [f for f in os.listdir(class_path) if f.endswith('.jpeg')]
        if len(image_files) == 0:
            continue
        
        # Randomly sample images
        num_samples = min(num_per_class, len(image_files))
        selected_files = random.sample(image_files, num_samples)
        
        for img_file in selected_files:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Preprocess EXACTLY as training (BGR format, no RGB conversion)
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_normalized = img_resized.astype('float32') / 255.0
                
                test_images.append(img_normalized)
                test_labels.append(class_idx)
                test_paths.append((class_name, img_file))
    
    return np.array(test_images), np.array(test_labels), test_paths

def predict_and_visualize(model, test_images, test_labels, test_paths, class_names):
    """Make predictions and create visualizations"""
    print(f"\nMaking predictions on {len(test_images)} test images...")
    
    # Make predictions
    predictions = model.predict(test_images, verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predicted_indices)
    print(f"\n[OK] Overall Accuracy: {accuracy:.2%}")
    
    # Create visualization
    num_images = len(test_images)
    cols = 5
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    correct_count = 0
    incorrect_count = 0
    
    for i in range(num_images):
        ax = axes[i]
        
        # Get prediction info
        true_class = class_names[test_labels[i]]
        pred_class = class_names[predicted_indices[i]]
        confidence = confidences[i]
        is_correct = (test_labels[i] == predicted_indices[i])
        
        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1
        
        # Display image (convert BGR to RGB for display)
        display_img = cv2.cvtColor((test_images[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        ax.imshow(display_img)
        ax.axis('off')
        
        # Color based on correctness
        color = 'green' if is_correct else 'red'
        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2%}"
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Model Verification - Accuracy: {accuracy:.2%} ({correct_count} correct, {incorrect_count} incorrect)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_verification.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved visualization to 'model_verification.png'")
    plt.close()
    
    return predictions, predicted_indices, confidences

def detailed_class_analysis(model, test_images, test_labels, class_names):
    """Detailed analysis per class"""
    print("\n" + "="*70)
    print("DETAILED CLASS ANALYSIS")
    print("="*70)
    
    predictions = model.predict(test_images, verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 70)
    print(f"{'Class':<8} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'Avg Confidence':<15}")
    print("-" * 70)
    
    class_stats = {}
    for class_idx, class_name in enumerate(class_names):
        class_mask = test_labels == class_idx
        if np.sum(class_mask) > 0:
            class_predictions = predicted_indices[class_mask]
            class_correct = np.sum(class_predictions == class_idx)
            class_total = len(class_predictions)
            class_accuracy = class_correct / class_total if class_total > 0 else 0
            
            # Average confidence for this class
            class_confidences = np.max(predictions[class_mask], axis=1)
            avg_confidence = np.mean(class_confidences)
            
            class_stats[class_name] = {
                'accuracy': class_accuracy,
                'correct': class_correct,
                'total': class_total,
                'avg_confidence': avg_confidence
            }
            
            status = "[OK]" if class_accuracy >= 0.9 else "[WARN]" if class_accuracy >= 0.7 else "[FAIL]"
            print(f"{status} {class_name:<6} {class_correct:<10} {class_total:<10} {class_accuracy:<12.2%} {avg_confidence:<15.2%}")
    
    return class_stats

def show_confusion_examples(model, test_images, test_labels, test_paths, class_names):
    """Show examples of common confusions"""
    print("\n" + "="*70)
    print("CONFUSION ANALYSIS")
    print("="*70)
    
    predictions = model.predict(test_images, verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)
    
    # Find incorrect predictions
    incorrect = []
    for i in range(len(test_labels)):
        if test_labels[i] != predicted_indices[i]:
            incorrect.append({
                'true': class_names[test_labels[i]],
                'pred': class_names[predicted_indices[i]],
                'confidence': np.max(predictions[i]),
                'image': test_images[i],
                'path': test_paths[i]
            })
    
    if len(incorrect) == 0:
        print("\n[OK] No incorrect predictions! Perfect accuracy on test set.")
        return
    
    # Group by confusion type
    confusion_groups = {}
    for item in incorrect:
        key = f"{item['true']} -> {item['pred']}"
        if key not in confusion_groups:
            confusion_groups[key] = []
        confusion_groups[key].append(item)
    
    # Show top confusions
    print(f"\nFound {len(incorrect)} incorrect predictions:")
    print("-" * 70)
    for confusion, items in sorted(confusion_groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {confusion}: {len(items)} times")
    
    # Visualize top confusions
    if len(incorrect) > 0:
        num_to_show = min(10, len(incorrect))
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, item in enumerate(incorrect[:num_to_show]):
            ax = axes[i]
            # Convert BGR to RGB for display
            display_img = cv2.cvtColor((item['image'] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            ax.imshow(display_img)
            ax.axis('off')
            title = f"True: {item['true']}\nPred: {item['pred']}\nConf: {item['confidence']:.2%}"
            ax.set_title(title, fontsize=10, color='red', fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        
        for i in range(num_to_show, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Incorrect Predictions (Red = Wrong)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('model_errors.png', dpi=150, bbox_inches='tight')
        print(f"\n[OK] Saved error examples to 'model_errors.png'")

def test_on_full_dataset(model, data_dir, class_names):
    """Test on a larger sample from the dataset"""
    print("\n" + "="*70)
    print("TESTING ON LARGER DATASET SAMPLE")
    print("="*70)
    
    all_images = []
    all_labels = []
    
    print("\nLoading images from dataset...")
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        image_files = [f for f in os.listdir(class_path) if f.endswith('.jpeg')]
        # Sample up to 50 images per class for faster testing
        num_samples = min(50, len(image_files))
        selected_files = random.sample(image_files, num_samples) if len(image_files) > num_samples else image_files
        
        for img_file in selected_files:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Preprocess EXACTLY as training (BGR format)
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_normalized = img_resized.astype('float32') / 255.0
                
                all_images.append(img_normalized)
                all_labels.append(class_idx)
        
        print(f"  Loaded {len(selected_files)} images for class '{class_name}'")
    
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    
    print(f"\nTotal test images: {len(all_images)}")
    print("Making predictions...")
    
    predictions = model.predict(all_images, batch_size=32, verbose=1)
    predicted_indices = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, predicted_indices)
    cm = confusion_matrix(all_labels, predicted_indices)
    
    print(f"\n{'='*70}")
    print(f"OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"Total Images Tested: {len(all_images)}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average Confidence: {np.mean(confidences):.4f} ({np.mean(confidences)*100:.2f}%)")
    print(f"Min Confidence: {np.min(confidences):.4f} ({np.min(confidences)*100:.2f}%)")
    print(f"Max Confidence: {np.max(confidences):.4f} ({np.max(confidences)*100:.2f}%)")
    
    # Per-class summary
    print(f"\n{'='*70}")
    print(f"PER-CLASS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Class':<8} {'Correct':<10} {'Total':<10} {'Accuracy':<12}")
    print("-" * 70)
    
    for class_idx, class_name in enumerate(class_names):
        class_mask = all_labels == class_idx
        if np.sum(class_mask) > 0:
            class_predictions = predicted_indices[class_mask]
            class_correct = np.sum(class_predictions == class_idx)
            class_total = len(class_predictions)
            class_accuracy = class_correct / class_total if class_total > 0 else 0
            
            status = "[OK]" if class_accuracy >= 0.95 else "[WARN]" if class_accuracy >= 0.85 else "[FAIL]"
            print(f"{status} {class_name:<6} {class_correct:<10} {class_total:<10} {class_accuracy:<12.2%}")
    
    return accuracy, cm, all_images, all_labels, predicted_indices, confidences

def main():
    print("="*70)
    print("MODEL VERIFICATION ON DATASET IMAGES")
    print("="*70)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load model and class names
    model = load_model()
    class_names = load_class_names()
    print(f"[OK] Loaded {len(class_names)} classes: {class_names}")
    
    # Test 1: Visual verification on random samples
    print(f"\n{'='*70}")
    print("TEST 1: Visual Verification (Random Samples)")
    print("="*70)
    test_images, test_labels, test_paths = load_test_images(DATA_DIR, class_names, NUM_PER_CLASS)
    print(f"Loaded {len(test_images)} test images")
    
    predictions, predicted_indices, confidences = predict_and_visualize(
        model, test_images, test_labels, test_paths, class_names
    )
    
    # Detailed analysis
    class_stats = detailed_class_analysis(model, test_images, test_labels, class_names)
    
    # Show confusion examples
    show_confusion_examples(model, test_images, test_labels, test_paths, class_names)
    
    # Test 2: Larger dataset test
    print(f"\n{'='*70}")
    print("TEST 2: Larger Dataset Sample Test")
    print("="*70)
    accuracy, cm, all_images, all_labels, pred_indices, confs = test_on_full_dataset(
        model, DATA_DIR, class_names
    )
    
    # Create confusion matrix visualization
    print("\nGenerating confusion matrix...")
    plt.figure(figsize=(14, 12))
    import seaborn as sns
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized'})
    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('model_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved confusion matrix to 'model_confusion_matrix.png'")
    plt.close()
    
    # Summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"[OK] Model loaded successfully")
    print(f"[OK] Tested on {len(all_images)} images")
    print(f"[OK] Overall Accuracy: {accuracy:.2%}")
    print(f"[OK] Average Confidence: {np.mean(confs):.2%}")
    print(f"\nGenerated files:")
    print(f"  - model_verification.png (sample predictions)")
    print(f"  - model_errors.png (incorrect predictions)")
    print(f"  - model_confusion_matrix.png (confusion matrix)")
    print("="*70)

if __name__ == '__main__':
    main()

