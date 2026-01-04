import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

# Configure GPU for RTX 3070 (8GB VRAM)
print("="*70)
print("CONFIGURING GPU FOR TRAINING")
print("="*70)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        print("[OK] GPU memory growth enabled")
        # Set mixed precision for better performance on RTX GPUs
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("[OK] Mixed precision training enabled (float16)")
        except:
            print("[INFO] Mixed precision not available, using float32")
    except RuntimeError as e:
        print(f"[WARNING] GPU configuration error: {e}")
        print("[INFO] Continuing with CPU...")
else:
    print("[WARNING] No GPU detected!")
    print("[INFO] Training will use CPU (will be slower)")
    print("\nTo enable GPU support:")
    print("  1. Install CUDA Toolkit (11.8 or 12.x)")
    print("  2. Install cuDNN")
    print("  3. Install tensorflow[and-cuda] or tensorflow-gpu")
    print("  4. Verify with: python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration - Optimized for RTX 3070 8GB VRAM
IMG_SIZE = 64
BATCH_SIZE = 64  # Increased for GPU (can go up to 128 if needed)
EPOCHS = 20
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

def create_model(num_classes, img_size=64):
    """Create CNN model for ASL classification with GPU optimization"""
    # Use mixed precision policy if enabled
    policy = tf.keras.mixed_precision.global_policy()
    use_mixed_precision = (policy.name == 'mixed_float16')
    
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        # Output layer: use float32 for softmax when using mixed precision
        layers.Dense(num_classes, activation='softmax', dtype='float32' if use_mixed_precision else None)
    ])
    
    return model

def main():
    print("Loading dataset...")
    images, labels, class_names = load_data()
    
    print(f"\nDataset loaded:")
    print(f"  Total images: {len(images)}")
    print(f"  Image shape: {images[0].shape}")
    print(f"  Number of classes: {len(class_names)}")
    print(f"  Classes: {class_names}")
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nData split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Don't flip ASL signs
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    # Create model
    print("\nCreating model...")
    model = create_model(len(class_names), IMG_SIZE)
    
    # Compile model with GPU optimizations
    # Use higher learning rate for GPU training
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    # If using mixed precision, wrap optimizer
    policy = tf.keras.mixed_precision.global_policy()
    if policy.name == 'mixed_float16':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        print("[INFO] Using mixed precision with loss scaling")
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'asl_model.keras',  # Use newer Keras format
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(X_val) // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save class names
    import pickle
    with open('class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nTraining history saved to 'training_history.png'")
    
    # Final evaluation
    print("\nEvaluating on validation set...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    print("\nTraining completed! Model saved as 'asl_model.keras'")

if __name__ == '__main__':
    main()

