import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, 
                            f1_score, precision_score, recall_score,
                            accuracy_score)
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pickle
from pathlib import Path
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
IMG_SIZE = 64
BATCH_SIZE = 32
BASE_DATA_DIR = 'dataset'

# Define all dataset paths
DATASET_PATHS = [
    'dataset/asl',  # Original dataset with lowercase letters
    'dataset/alphabet and numbers 1',  # First Kaggle dataset
    'dataset/alphabet and numbers 2',  # Second Kaggle dataset
    'dataset/asl_alphabet_train/asl_alphabet_train',  # Third Kaggle dataset (nested)
]

MODEL_PATH = 'asl_model.h5'
CLASS_NAMES_PATH = 'class_names.pkl'
OUTPUT_DIR = 'model_metrics'

def normalize_class_name(class_name):
    """Normalize class names: convert lowercase letters to uppercase, keep special classes as-is"""
    # If it's a single lowercase letter, convert to uppercase
    if len(class_name) == 1 and class_name.isalpha() and class_name.islower():
        return class_name.upper()
    # Keep numbers and special classes (nothing, space, unknown, del) as-is
    return class_name

def load_data_from_directory(dataset_path, class_mapping, images, labels):
    """Load images from a single dataset directory"""
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
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Normalize class name
        normalized_name = normalize_class_name(class_name)
        
        # Get or create class index
        if normalized_name not in class_mapping:
            class_mapping[normalized_name] = len(class_mapping)
        
        class_idx = class_mapping[normalized_name]
        
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

def load_data():
    """Load images and labels from all dataset directories"""
    images = []
    labels = []
    class_mapping = {}  # Maps normalized class name to index
    
    print("="*70)
    print("LOADING DATA FROM MULTIPLE DATASETS")
    print("="*70)
    
    # Load from all dataset paths
    for dataset_path in DATASET_PATHS:
        load_data_from_directory(dataset_path, class_mapping, images, labels)
    
    # Create sorted class names list
    class_names = sorted(class_mapping.keys(), key=lambda x: (
        # Sort: numbers first (0-9), then letters (A-Z), then special classes
        (0, int(x)) if x.isdigit() else (1, x) if x.isalpha() else (2, x)
    ))
    
    # Create reverse mapping: old_index -> normalized_name -> new_index
    old_to_name = {idx: name for name, idx in class_mapping.items()}
    name_to_new = {name: idx for idx, name in enumerate(class_names)}
    
    # Remap labels to sorted indices
    labels_remapped = [name_to_new[old_to_name[label]] for label in labels]
    
    print("\n" + "="*70)
    print(f"TOTAL DATASET SUMMARY")
    print("="*70)
    print(f"Total images loaded: {len(images)}")
    print(f"Total classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Print class distribution
    label_counts = Counter(labels_remapped)
    print("\nClass distribution:")
    for idx, class_name in enumerate(class_names):
        count = label_counts.get(idx, 0)
        print(f"  {class_name}: {count} images")
    
    return np.array(images), np.array(labels_remapped), class_names

def create_output_dir():
    """Create output directory for metrics"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}")

def plot_training_history():
    """Plot training history if available"""
    # Try to load from saved history or recreate
    print("\n1. Generating Training History Plot...")
    
    # Since we don't have the history object, we'll note this
    # In a real scenario, you'd save history during training
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Placeholder - in practice, load from saved history
    axes[0].text(0.5, 0.5, 'Training history data\nnot available.\nRe-run training to generate.', 
                ha='center', va='center', fontsize=14, transform=axes[0].transAxes)
    axes[0].set_title('Training History - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    
    axes[1].text(0.5, 0.5, 'Training history data\nnot available.\nRe-run training to generate.', 
                ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
    axes[1].set_title('Training History - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_training_history.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/01_training_history.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    print("\n2. Generating Confusion Matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw confusion matrix
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].tick_params(axis='both', labelsize=8)
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Normalized'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].tick_params(axis='both', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/02_confusion_matrix.png")
    plt.close()
    
    return cm

def plot_class_metrics(y_true, y_pred, class_names):
    """Plot per-class metrics (F1, Precision, Recall)"""
    print("\n3. Generating Class-wise Metrics...")
    
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Create DataFrame for easier plotting
    metrics_df = {
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(20, 8))
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Metrics (Precision, Recall, F1-Score)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only label if significant
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_class_metrics.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/03_class_metrics.png")
    plt.close()
    
    return precision, recall, f1

def plot_f1_score_heatmap(f1_scores, class_names):
    """Plot F1 scores as heatmap"""
    print("\n4. Generating F1-Score Heatmap...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Reshape for heatmap (1D to 2D for visualization)
    f1_2d = f1_scores.reshape(-1, 1)
    class_names_2d = np.array(class_names).reshape(-1, 1)
    
    # Create heatmap
    sns.heatmap(f1_2d, annot=True, fmt='.3f', cmap='YlOrRd',
                yticklabels=class_names, xticklabels=['F1-Score'],
                cbar_kws={'label': 'F1-Score'}, ax=ax, vmin=0, vmax=1)
    
    ax.set_title('F1-Score by Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Class', fontsize=12)
    ax.tick_params(axis='y', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_f1_score_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/04_f1_score_heatmap.png")
    plt.close()

def plot_accuracy_by_class(y_true, y_pred, class_names):
    """Plot accuracy for each class"""
    print("\n5. Generating Per-Class Accuracy...")
    
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    bars = ax.bar(class_names, class_accuracies, color='steelblue', alpha=0.7)
    
    # Color bars based on performance
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        if acc >= 0.95:
            bar.set_color('green')
        elif acc >= 0.85:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% Threshold')
    ax.axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='85% Threshold')
    ax.legend()
    
    # Add value labels
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.3f}',
               ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/05_per_class_accuracy.png")
    plt.close()
    
    return class_accuracies

def plot_error_analysis(cm, class_names):
    """Plot error analysis"""
    print("\n6. Generating Error Analysis...")
    
    # Calculate most confused pairs
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                errors.append((class_names[i], class_names[j], cm[i, j]))
    
    errors.sort(key=lambda x: x[2], reverse=True)
    top_errors = errors[:20]  # Top 20 errors
    
    if top_errors:
        fig, ax = plt.subplots(figsize=(12, 8))
        true_labels = [e[0] for e in top_errors]
        pred_labels = [e[1] for e in top_errors]
        counts = [e[2] for e in top_errors]
        
        y_pos = np.arange(len(top_errors))
        bars = ax.barh(y_pos, counts, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{true} -> {pred}" for true, pred in zip(true_labels, pred_labels)])
        ax.set_xlabel('Number of Misclassifications', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Confusion Pairs', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, bar.get_y() + bar.get_height()/2,
                   f' {count}',
                   ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/06_error_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: {OUTPUT_DIR}/06_error_analysis.png")
        plt.close()

def plot_model_summary_metrics(overall_metrics):
    """Plot overall model metrics summary"""
    print("\n7. Generating Overall Metrics Summary...")
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        overall_metrics['accuracy'],
        overall_metrics['precision'],
        overall_metrics['recall'],
        overall_metrics['f1']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'], alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Overall Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_overall_metrics.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/07_overall_metrics.png")
    plt.close()

def plot_class_distribution(y_train, y_val, class_names):
    """Plot class distribution in train/val sets"""
    print("\n8. Generating Class Distribution...")
    
    train_counts = [np.sum(y_train == i) for i in range(len(class_names))]
    val_counts = [np.sum(y_val == i) for i in range(len(class_names))]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(20, 8))
    bars1 = ax.bar(x - width/2, train_counts, width, label='Training', alpha=0.8)
    bars2 = ax.bar(x + width/2, val_counts, width, label='Validation', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution in Training and Validation Sets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/08_class_distribution.png")
    plt.close()

def plot_sample_predictions(model, X_val, y_val, class_names, n_samples=16):
    """Plot sample predictions with confidence"""
    print("\n9. Generating Sample Predictions...")
    
    indices = np.random.choice(len(X_val), n_samples, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, ax in zip(indices, axes):
        img = X_val[idx]
        true_label = class_names[y_val[idx]]
        
        # Predict
        pred_proba = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        pred_idx = np.argmax(pred_proba)
        pred_label = class_names[pred_idx]
        confidence = pred_proba[pred_idx]
        
        # Display image
        ax.imshow(img)
        ax.axis('off')
        
        # Color based on correctness
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}'
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/09_sample_predictions.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/09_sample_predictions.png")
    plt.close()

def generate_text_report(y_true, y_pred, class_names, overall_metrics, 
                        precision, recall, f1, class_accuracies):
    """Generate text report with all metrics"""
    print("\n10. Generating Text Report...")
    
    report = f"""
================================================================================
                    ASL RECOGNITION MODEL - COMPREHENSIVE REPORT
================================================================================

OVERALL PERFORMANCE METRICS
--------------------------------------------------------------------------------
Accuracy:    {overall_metrics['accuracy']:.4f} ({overall_metrics['accuracy']*100:.2f}%)
Precision:   {overall_metrics['precision']:.4f} ({overall_metrics['precision']*100:.2f}%)
Recall:      {overall_metrics['recall']:.4f} ({overall_metrics['recall']*100:.2f}%)
F1-Score:    {overall_metrics['f1']:.4f} ({overall_metrics['f1']*100:.2f}%)

================================================================================
PER-CLASS METRICS
--------------------------------------------------------------------------------
{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12}
--------------------------------------------------------------------------------
"""
    
    for i, class_name in enumerate(class_names):
        report += f"{class_name:<8} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {class_accuracies[i]:<12.4f}\n"
    
    report += f"""
================================================================================
CLASS PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Classes with F1-Score >= 0.95: {np.sum(f1 >= 0.95)}/{len(class_names)}
Classes with F1-Score >= 0.90: {np.sum(f1 >= 0.90)}/{len(class_names)}
Classes with F1-Score >= 0.85: {np.sum(f1 >= 0.85)}/{len(class_names)}
Classes with F1-Score < 0.85:  {np.sum(f1 < 0.85)}/{len(class_names)}

Best performing classes (F1-Score):
"""
    
    top_indices = np.argsort(f1)[::-1][:5]
    for idx in top_indices:
        report += f"  {class_names[idx]}: {f1[idx]:.4f}\n"
    
    report += "\nWorst performing classes (F1-Score):\n"
    bottom_indices = np.argsort(f1)[:5]
    for idx in bottom_indices:
        report += f"  {class_names[idx]}: {f1[idx]:.4f}\n"
    
    report += """
================================================================================
MODEL INFORMATION
--------------------------------------------------------------------------------
Model Architecture: CNN with 4 Convolutional Blocks
Total Parameters: 1,057,764
Input Size: 64x64x3
Number of Classes: 36
Classes: 0-9, a-z

================================================================================
"""
    
    with open(f'{OUTPUT_DIR}/model_report.txt', 'w') as f:
        f.write(report)
    
    print(f"   Saved: {OUTPUT_DIR}/model_report.txt")
    print(report)

def main():
    print("="*70)
    print("GENERATING COMPREHENSIVE MODEL METRICS AND VISUALIZATIONS")
    print("="*70)
    
    create_output_dir()
    
    # Load model
    print("\nLoading model...")
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except:
        try:
            alt_path = MODEL_PATH.replace('.h5', '.keras')
            model = keras.models.load_model(alt_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    
    # Load class names
    with open(CLASS_NAMES_PATH, 'rb') as f:
        class_names = pickle.load(f)
    
    # Load data
    print("\nLoading dataset...")
    images, labels, _ = load_data()
    print(f"Loaded {len(images)} images")
    
    # Split data (same as training)
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Make predictions
    print("\nMaking predictions on validation set...")
    y_pred_proba = model.predict(X_val, batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(y_val, y_pred)
    overall_precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    overall_recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    
    overall_metrics = {
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1
    }
    
    # Generate all plots
    plot_training_history()
    cm = plot_confusion_matrix(y_val, y_pred, class_names)
    precision, recall, f1 = plot_class_metrics(y_val, y_pred, class_names)
    plot_f1_score_heatmap(f1, class_names)
    class_accuracies = plot_accuracy_by_class(y_val, y_pred, class_names)
    plot_error_analysis(cm, class_names)
    plot_model_summary_metrics(overall_metrics)
    plot_class_distribution(y_train, y_val, class_names)
    plot_sample_predictions(model, X_val, y_val, class_names)
    generate_text_report(y_val, y_pred, class_names, overall_metrics,
                       precision, recall, f1, class_accuracies)
    
    print("\n" + "="*70)
    print("ALL METRICS AND VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print(f"Check the '{OUTPUT_DIR}' directory for all output files.")
    print("="*70)

if __name__ == '__main__':
    main()


