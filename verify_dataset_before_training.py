"""
Verify that the dataset has all required classes before training.
This ensures the model will be trained with all 40 classes.
"""

import os
from collections import Counter

# Define all dataset paths
DATASET_PATHS = [
    'dataset/asl',
    'dataset/alphabet and numbers 1',
    'dataset/alphabet and numbers 2',
    'dataset/asl_alphabet_train/asl_alphabet_train',
]

# Expected classes
EXPECTED_CLASSES = set([
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space', 'unknown'
])

def normalize_class_name(class_name):
    """Normalize class names"""
    if len(class_name) == 1 and class_name.isalpha() and class_name.islower():
        return class_name.upper()
    return class_name

def check_dataset():
    """Check if all expected classes are in the dataset"""
    print("="*70)
    print("VERIFYING DATASET BEFORE TRAINING")
    print("="*70)
    
    found_classes = set()
    class_counts = Counter()
    
    for dataset_path in DATASET_PATHS:
        if not os.path.exists(dataset_path):
            print(f"\n[SKIP] Dataset path not found: {dataset_path}")
            continue
        
        print(f"\nChecking: {dataset_path}")
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
            found_classes.add(normalized_name)
            
            # Count images
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_counts[normalized_name] += len(image_files)
    
    print("\n" + "="*70)
    print("DATASET VERIFICATION RESULTS")
    print("="*70)
    
    print(f"\nExpected classes: {len(EXPECTED_CLASSES)}")
    print(f"Found classes: {len(found_classes)}")
    
    missing_classes = EXPECTED_CLASSES - found_classes
    extra_classes = found_classes - EXPECTED_CLASSES
    
    if missing_classes:
        print(f"\n[ERROR] Missing classes ({len(missing_classes)}):")
        for cls in sorted(missing_classes):
            print(f"  - {cls}")
        print("\nThese classes are required but not found in the dataset!")
        return False
    else:
        print("\n[OK] All expected classes found in dataset!")
    
    if extra_classes:
        print(f"\n[INFO] Extra classes found ({len(extra_classes)}):")
        for cls in sorted(extra_classes):
            print(f"  - {cls}")
        print("These will be included in training.")
    
    # Show class distribution
    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTION:")
    print("-" * 70)
    sorted_classes = sorted(found_classes, key=lambda x: (
        (0, int(x)) if x.isdigit() else (1, x) if x.isalpha() else (2, x)
    ))
    
    for cls in sorted_classes:
        count = class_counts[cls]
        print(f"  {cls:<12} {count:>6} images")
    
    # Check for classes with very few samples
    min_samples = min(class_counts.values()) if class_counts else 0
    low_sample_classes = [cls for cls, count in class_counts.items() if count < 100]
    
    if low_sample_classes:
        print(f"\n[WARNING] Classes with less than 100 images ({len(low_sample_classes)}):")
        for cls in sorted(low_sample_classes):
            print(f"  - {cls}: {class_counts[cls]} images")
        print("Consider collecting more data for these classes.")
    
    total_images = sum(class_counts.values())
    print(f"\nTotal images: {total_images:,}")
    print(f"Average per class: {total_images / len(found_classes):.0f}")
    
    print("\n" + "="*70)
    print("[OK] Dataset is ready for training!")
    print("="*70)
    print("\nYou can now run: python train_model.py")
    print("The model will be trained with all", len(found_classes), "classes.")
    
    return True

if __name__ == '__main__':
    success = check_dataset()
    if not success:
        print("\n[ERROR] Please ensure all required classes are present in the dataset.")
        exit(1)

