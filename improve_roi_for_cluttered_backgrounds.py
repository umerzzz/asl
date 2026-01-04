"""
Script to test and improve ROI extraction for cluttered backgrounds.
This will help match the training data format better.
"""

import cv2
import numpy as np
import os
import glob

IMG_SIZE = 64

def analyze_training_image_format():
    """Analyze what training images actually look like"""
    print("="*70)
    print("ANALYZING TRAINING IMAGE FORMAT")
    print("="*70)
    
    # Find sample images from different classes
    sample_paths = []
    for class_name in ['A', '5', 'space']:
        paths = glob.glob(f'dataset/**/{class_name}/*.jpg', recursive=True)
        if paths:
            sample_paths.append((class_name, paths[0]))
    
    if not sample_paths:
        print("[ERROR] No training images found")
        return
    
    print("\nSample training images:")
    for class_name, path in sample_paths:
        img = cv2.imread(path)
        if img is not None:
            print(f"\n  Class '{class_name}': {path}")
            print(f"    Original size: {img.shape}")
            print(f"    After resize to {IMG_SIZE}x{IMG_SIZE}: {IMG_SIZE}x{IMG_SIZE}x3")
            
            # Show what the model sees
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Calculate some statistics
            mean_brightness = np.mean(img)
            std_brightness = np.std(img)
            print(f"    Mean brightness: {mean_brightness:.1f}")
            print(f"    Std brightness: {std_brightness:.1f}")
            
            # Check if image is square
            h, w = img.shape[:2]
            aspect_ratio = w / h
            print(f"    Aspect ratio: {aspect_ratio:.2f} ({'square' if 0.95 < aspect_ratio < 1.05 else 'not square'})")

def test_roi_extraction_methods():
    """Test different ROI extraction methods"""
    print("\n" + "="*70)
    print("TESTING ROI EXTRACTION METHODS")
    print("="*70)
    
    # Load a sample training image
    sample_path = glob.glob('dataset/**/A/*.jpg', recursive=True)
    if not sample_path:
        print("[ERROR] No sample image found")
        return
    
    train_img = cv2.imread(sample_path[0])
    if train_img is None:
        print("[ERROR] Could not load sample image")
        return
    
    h, w = train_img.shape[:2]
    print(f"\nOriginal image: {w}x{h}")
    
    # Method 1: Current (70% crop)
    roi_size_70 = int(min(w, h) * 0.70)
    x_70 = (w - roi_size_70) // 2
    y_70 = (h - roi_size_70) // 2
    roi_70 = train_img[y_70:y_70+roi_size_70, x_70:x_70+roi_size_70]
    roi_70_resized = cv2.resize(roi_70, (IMG_SIZE, IMG_SIZE))
    
    # Method 2: Larger crop (85%)
    roi_size_85 = int(min(w, h) * 0.85)
    x_85 = (w - roi_size_85) // 2
    y_85 = (h - roi_size_85) // 2
    roi_85 = train_img[y_85:y_85+roi_size_85, x_85:x_85+roi_size_85]
    roi_85_resized = cv2.resize(roi_85, (IMG_SIZE, IMG_SIZE))
    
    # Method 3: Full image (what training does)
    roi_full = cv2.resize(train_img, (IMG_SIZE, IMG_SIZE))
    
    # Compare
    print("\nComparison:")
    print(f"  70% crop: {roi_size_70}x{roi_size_70} -> {IMG_SIZE}x{IMG_SIZE}")
    print(f"  85% crop: {roi_size_85}x{roi_size_85} -> {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Full image: {w}x{h} -> {IMG_SIZE}x{IMG_SIZE} (training method)")
    
    # Create comparison image
    comparison = np.hstack([
        roi_70_resized,
        roi_85_resized,
        roi_full
    ])
    
    # Add labels
    cv2.putText(comparison, "70%", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, "85%", (IMG_SIZE + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, "Full", (2*IMG_SIZE + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite('roi_methods_comparison.png', comparison)
    print("\n[OK] Saved comparison to 'roi_methods_comparison.png'")
    print("     Left: 70% crop, Middle: 85% crop, Right: Full image (training)")

if __name__ == '__main__':
    analyze_training_image_format()
    test_roi_extraction_methods()
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("Training data uses FULL images resized to 64x64.")
    print("For real-time, use larger ROI (85%+) to include more context.")
    print("This better matches the training data format.")

