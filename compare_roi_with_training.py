"""
Compare debug ROI images with training images to identify the issue
"""

import cv2
import numpy as np
import os
import glob

def compare_roi_with_training():
    """Compare a debug ROI with a training image"""
    
    # Get a training image
    train_paths = glob.glob('dataset/**/A/*.jpg', recursive=True)
    if not train_paths:
        print("No training images found for 'A'")
        return
    
    train_img = cv2.imread(train_paths[0])
    print(f"Training image: {train_img.shape}")
    
    # Get a recent debug ROI
    debug_files = sorted([f for f in os.listdir('debug_rois') if f.endswith('.jpg')], reverse=True)
    if not debug_files:
        print("No debug ROIs found")
        return
    
    # Try to find a non-unknown ROI
    debug_file = None
    for f in debug_files[:20]:
        if not f.startswith('unknown_'):
            debug_file = f
            break
    
    if not debug_file:
        debug_file = debug_files[0]  # Use unknown if nothing else
    
    debug_img = cv2.imread(f'debug_rois/{debug_file}')
    print(f"Debug ROI: {debug_img.shape} ({debug_file})")
    
    # Resize both to 64x64 (what model sees)
    train_64 = cv2.resize(train_img, (64, 64))
    debug_64 = cv2.resize(debug_img, (64, 64))
    
    # Resize 64x64 images to 224x224 for display
    train_64_display = cv2.resize(train_64, (224, 224), interpolation=cv2.INTER_NEAREST)
    debug_64_display = cv2.resize(debug_64, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    # Create comparison (all 224x224 for display)
    comparison = np.hstack([
        train_img,  # Original training (224x224)
        debug_img,  # Debug ROI (224x224)
        train_64_display,  # Training at 64x64 (upscaled for display)
        debug_64_display   # Debug at 64x64 (upscaled for display)
    ])
    
    # Add labels
    cv2.putText(comparison, "Training 224x224", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, "Debug ROI 224x224", (234, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, "Training 64x64", (458, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, "Debug 64x64", (522, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Calculate statistics
    print("\nStatistics:")
    print(f"Training image - Mean: {np.mean(train_img):.1f}, Std: {np.std(train_img):.1f}")
    print(f"Debug ROI - Mean: {np.mean(debug_img):.1f}, Std: {np.std(debug_img):.1f}")
    
    # Check hand size in image
    train_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
    debug_gray = cv2.cvtColor(debug_img, cv2.COLOR_BGR2GRAY)
    
    # Estimate hand size (non-background pixels)
    train_thresh = cv2.threshold(train_gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
    debug_thresh = cv2.threshold(debug_gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
    
    train_hand_pixels = np.sum(train_thresh > 0)
    debug_hand_pixels = np.sum(debug_thresh > 0)
    
    print(f"\nHand size estimate (non-background pixels):")
    print(f"Training: {train_hand_pixels} ({train_hand_pixels/(224*224)*100:.1f}% of image)")
    print(f"Debug: {debug_hand_pixels} ({debug_hand_pixels/(224*224)*100:.1f}% of image)")
    
    # Save comparison
    cv2.imwrite('roi_comparison.png', comparison)
    print(f"\nComparison saved to 'roi_comparison.png'")
    
    return train_img, debug_img

if __name__ == '__main__':
    compare_roi_with_training()

