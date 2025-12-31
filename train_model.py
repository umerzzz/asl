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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = 'asl'

def load_data(data_dir):
    """Load images and labels from directory structure"""
    images = []
    labels = []
    class_names = []
    
    # Get all class directories (excluding 'asl' subfolder)
    classes = sorted([d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d)) and d != 'asl'])
    
    print(f"Found {len(classes)} classes: {classes}")
    
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_path) if f.endswith('.jpeg')]
        
        print(f"Loading {len(image_files)} images for class '{class_name}'")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Resize and normalize
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(idx)
        
        class_names.append(class_name)
    
    return np.array(images), np.array(labels), class_names

def create_model(num_classes, img_size=64):
    """Create CNN model for ASL classification"""
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
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    print("Loading dataset...")
    images, labels, class_names = load_data(DATA_DIR)
    
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
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
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

