# ROI Extraction Fix for Cluttered Backgrounds

## Root Cause Identified

The training data uses:
- **224x224 square images** (full image, not cropped)
- **Resized directly to 64x64** (no intermediate cropping)
- **Hands are centered** in the images

The real-time extraction was:
- Using **70% center crop** (too small, loses context)
- Not matching the training format well

## Fixes Applied

### 1. **Larger ROI Extraction**
- **Before**: 70% center crop
- **After**: 90% center crop (larger, includes more context)
- **MediaPipe**: Increased padding to 100px (more context around hand)

### 2. **Square ROI Matching**
- Ensures ROI is square (training uses square images)
- Centers hand in square ROI
- Matches training data format exactly

### 3. **Better Format Matching**
- ROI extraction now matches training: square region → resize to 64x64
- Same preprocessing pipeline as training

## Key Changes

1. **90% center crop** instead of 70% (more context like training)
2. **Square ROI** enforced (matches training format)
3. **Larger MediaPipe padding** (100px instead of 60px)
4. **Unknown filtering** (only accepts if >95% confident)

## Testing

Run the script and check:
1. **"MODEL INPUT" preview** - should show hand centered in square
2. **ROI rectangle** - should be square and large (90% of frame)
3. **Predictions** - should be more accurate, fewer "unknown"

## Debug Mode

ROI images are saved to `debug_rois/` folder when `SAVE_ROI_DEBUG = True`

Compare these with training images to verify format matching.

