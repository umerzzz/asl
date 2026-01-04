import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import pickle
import cv2
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter

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

def plot_confidence_distribution(model, X_val, y_val, class_names):
    """Plot confidence score distribution"""
    print("\n11. Generating Confidence Distribution...")
    
    predictions = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    confidences = np.max(predictions, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(confidences, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
    axes[0].axvline(np.median(confidences), color='green', linestyle='--', 
                   label=f'Median: {np.median(confidences):.3f}')
    axes[0].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Prediction Confidence Scores', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot by correctness
    correct_preds = np.argmax(predictions, axis=1) == y_val
    correct_conf = confidences[correct_preds]
    incorrect_conf = confidences[~correct_preds]
    
    axes[1].boxplot([correct_conf, incorrect_conf], labels=['Correct', 'Incorrect'])
    axes[1].set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Confidence Scores: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/10_confidence_distribution.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/10_confidence_distribution.png")
    plt.close()

def plot_top_k_accuracy(y_true, y_pred_proba, class_names, k_values=[1, 2, 3, 5]):
    """Plot top-k accuracy"""
    print("\n12. Generating Top-K Accuracy...")
    
    top_k_accs = []
    for k in k_values:
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        top_k_correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        top_k_acc = np.mean(top_k_correct)
        top_k_accs.append(top_k_acc)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar([f'Top-{k}' for k in k_values], top_k_accs, 
                  color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'], alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Top-K Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, top_k_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.4f}\n({acc*100:.2f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/11_top_k_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/11_top_k_accuracy.png")
    plt.close()

def plot_model_complexity():
    """Plot model architecture complexity"""
    print("\n13. Generating Model Complexity Visualization...")
    
    layers_info = [
        ('Conv2D (32)', 896),
        ('Conv2D (64)', 18496),
        ('Conv2D (128)', 73856),
        ('Conv2D (256)', 295168),
        ('Dense (512)', 524800),
        ('Dense (256)', 131328),
        ('Dense (36)', 9252)
    ]
    
    layer_names = [l[0] for l in layers_info]
    param_counts = [l[1] for l in layers_info]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    bars = ax1.barh(layer_names, param_counts, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax1.set_title('Parameters per Layer', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for i, (bar, count) in enumerate(zip(bars, param_counts)):
        ax1.text(count, bar.get_y() + bar.get_height()/2,
                f' {count:,}',
                ha='left', va='center', fontsize=9)
    
    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(layer_names)))
    wedges, texts, autotexts = ax2.pie(param_counts, labels=layer_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax2.set_title('Parameter Distribution by Layer', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/12_model_complexity.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/12_model_complexity.png")
    plt.close()

def plot_performance_comparison(precision, recall, f1, class_names):
    """Create a comprehensive performance comparison"""
    print("\n14. Generating Performance Comparison Matrix...")
    
    # Create a heatmap with all three metrics
    metrics_matrix = np.array([precision, recall, f1]).T
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=['Precision', 'Recall', 'F1-Score'],
                yticklabels=class_names, cbar_kws={'label': 'Score'},
                vmin=0, vmax=1, ax=ax)
    
    ax.set_title('Performance Metrics Heatmap (Precision, Recall, F1-Score)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Class', fontsize=12)
    ax.tick_params(axis='y', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/13_performance_matrix.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/13_performance_matrix.png")
    plt.close()

def plot_class_balance_analysis(y_train, y_val, class_names):
    """Analyze class balance"""
    print("\n15. Generating Class Balance Analysis...")
    
    train_counts = [np.sum(y_train == i) for i in range(len(class_names))]
    val_counts = [np.sum(y_val == i) for i in range(len(class_names))]
    total_counts = [t + v for t, v in zip(train_counts, val_counts)]
    
    # Calculate imbalance ratio
    max_count = max(total_counts)
    min_count = min(total_counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Counts
    x = np.arange(len(class_names))
    width = 0.35
    axes[0].bar(x - width/2, train_counts, width, label='Training', alpha=0.8)
    axes[0].bar(x + width/2, val_counts, width, label='Validation', alpha=0.8)
    axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Class Distribution (Imbalance Ratio: {imbalance_ratio:.2f})', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Percentage
    train_pct = [t/sum(train_counts)*100 for t in train_counts]
    val_pct = [v/sum(val_counts)*100 for v in val_counts]
    axes[1].bar(x - width/2, train_pct, width, label='Training %', alpha=0.8)
    axes[1].bar(x + width/2, val_pct, width, label='Validation %', alpha=0.8)
    axes[1].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/14_class_balance.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/14_class_balance.png")
    plt.close()

def main():
    print("="*70)
    print("GENERATING ADDITIONAL METRICS AND VISUALIZATIONS")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model = keras.models.load_model(MODEL_PATH)
    
    # Load class names
    with open(CLASS_NAMES_PATH, 'rb') as f:
        class_names = pickle.load(f)
    
    # Load data
    print("\nLoading dataset...")
    images, labels, _ = load_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_proba = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_val, y_pred, average=None, zero_division=0)
    recall = recall_score(y_val, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_val, y_pred, average=None, zero_division=0)
    
    # Generate additional plots
    plot_confidence_distribution(model, X_val, y_val, class_names)
    plot_top_k_accuracy(y_val, y_pred_proba, class_names)
    plot_model_complexity()
    plot_performance_comparison(precision, recall, f1, class_names)
    plot_class_balance_analysis(y_train, y_val, class_names)
    
    print("\n" + "="*70)
    print("ADDITIONAL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*70)

if __name__ == '__main__':
    main()


