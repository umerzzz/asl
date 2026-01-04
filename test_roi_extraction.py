"""
Test script to visualize ROI extraction and compare with training data format
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

IMG_SIZE = 64

def load_sample_training_image():
    """Load a sample image from training data"""
    # Try to find a sample image
    dataset_paths = [
        'dataset/asl/A',
        'dataset/alphabet and numbers 1/A',
        'dataset/asl_alphabet_train/asl_alphabet_train/A',
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, '*.jpg')) + glob.glob(os.path.join(path, '*.jpeg'))
            if files:
                img = cv2.imread(files[0])
                if img is not None:
                    # Preprocess as training
                    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img_normalized = img_resized.astype('float32') / 255.0
                    return img, img_normalized, files[0]
    
    return None, None, None

def simulate_realtime_roi(frame):
    """Simulate what ROI would be extracted in real-time"""
    h, w = frame.shape[:2]
    
    # Center crop (what we're using)
    roi_size = int(min(w, h) * 0.70)
    x = int((w - roi_size) / 2)
    y = int((h - roi_size) / 2)
    
    roi = frame[y:y+roi_size, x:x+roi_size].copy()
    
    # Make square
    if roi.shape[0] != roi.shape[1]:
        size = max(roi.shape[0], roi.shape[1])
        square_roi = np.zeros((size, size, 3), dtype=roi.dtype)
        y_offset = (size - roi.shape[0]) // 2
        x_offset = (size - roi.shape[1]) // 2
        square_roi[y_offset:y_offset+roi.shape[0], x_offset:x_offset+roi.shape[1]] = roi
        roi = square_roi
    
    # Resize to model input
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    roi_normalized = roi_resized.astype('float32') / 255.0
    
    return roi, roi_normalized, (x, y, roi_size, roi_size)

def main():
    print("="*70)
    print("ROI EXTRACTION TEST")
    print("="*70)
    
    # Load sample training image
    train_img, train_normalized, train_path = load_sample_training_image()
    
    if train_img is None:
        print("[ERROR] Could not find sample training image")
        print("Please ensure dataset is available")
        return
    
    print(f"\n[OK] Loaded training sample: {train_path}")
    print(f"     Original size: {train_img.shape}")
    print(f"     Resized to: {train_normalized.shape}")
    
    # Simulate real-time extraction
    roi, roi_normalized, coords = simulate_realtime_roi(train_img)
    
    print(f"\n[OK] Simulated real-time ROI extraction")
    print(f"     ROI coordinates: {coords}")
    print(f"     ROI size: {roi.shape}")
    print(f"     Final normalized: {roi_normalized.shape}")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    # Visualize
    train_display = (train_normalized * 255).astype(np.uint8)
    roi_display = (roi_normalized * 255).astype(np.uint8)
    
    # Create comparison image
    comparison = np.hstack([train_display, roi_display])
    comparison = cv2.resize(comparison, (256, 128))
    
    cv2.putText(comparison, "Training Format", (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, "Real-time ROI", (138, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    print("\n[INFO] Showing comparison (Training vs Real-time ROI)")
    print("       Close window to continue...")
    
    cv2.imshow('Training Format vs Real-time ROI', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save for inspection
    cv2.imwrite('roi_comparison.png', comparison)
    print("\n[OK] Saved comparison to 'roi_comparison.png'")
    
    # Calculate difference
    diff = np.abs(train_normalized - roi_normalized)
    avg_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    print(f"\n[INFO] Difference analysis:")
    print(f"       Average difference: {avg_diff:.4f}")
    print(f"       Max difference: {max_diff:.4f}")
    
    if avg_diff > 0.1:
        print("\n[WARNING] Significant difference detected!")
        print("          ROI extraction may not match training format well.")
    else:
        print("\n[OK] ROI format matches training data well.")

if __name__ == '__main__':
    main()

