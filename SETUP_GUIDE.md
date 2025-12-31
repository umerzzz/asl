# ASL Recognition - Complete Setup Guide

## Quick Setup Checklist

### ✅ Step 1: Test Your Setup
Run the setup verification script first:
```bash
python test_setup.py
```

This will show you:
- If your camera is working
- If lighting is adequate
- If hand detection is working
- If background is suitable

### ✅ Step 2: Optimal Conditions

#### **Lighting Requirements**
- ✅ **BEST**: Natural daylight or bright, even room lighting
- ✅ **GOOD**: Bright LED/fluorescent lights (evenly distributed)
- ⚠️ **AVOID**: 
  - Harsh shadows on your hand
  - Backlighting (bright light behind you)
  - Too dim (can't see hand clearly)
  - Too bright (washes out details)
  - Uneven lighting (one side bright, one side dark)

**How to check**: Run `test_setup.py` - it will tell you if lighting is "GOOD", "TOO DARK", or "TOO BRIGHT"

#### **Background Requirements**
- ✅ **BEST**: Plain dark wall or dark cloth/paper
- ✅ **GOOD**: Plain light wall (white/beige)
- ✅ **OK**: Plain colored wall (solid color)
- ⚠️ **AVOID**:
  - Busy patterns (wallpaper, posters)
  - Textures (brick, wood grain)
  - Multiple colors/objects
  - Background same color as your skin

**Why**: The model was trained on images with black/dark backgrounds. A contrasting background helps hand detection.

**Quick test**: Hold your hand up - if you can clearly see the outline against the background, it's good!

#### **Hand Positioning**
- ✅ Position hand in **center** of camera view
- ✅ Hand should fill **60-80%** of the detection area
- ✅ Keep hand **steady** (don't move around)
- ✅ Make sure **entire hand** is visible
- ✅ Hold sign **clearly** for 1-2 seconds

#### **Camera Setup**
- ✅ Camera should be **stable** (not moving)
- ✅ Position camera at **eye level** or slightly above
- ✅ Distance: **1-2 feet** from your hand
- ✅ Make sure camera is **focused** (not blurry)

### ✅ Step 3: Visual Indicators

When you run the real-time recognition, look for these indicators:

#### **Green Rectangle** ✅
- Hand is detected successfully
- System is ready to recognize signs
- This is what you want to see!

#### **Orange Rectangle** ⚠️
- Hand not detected, using center region
- Try: Better lighting, move hand closer, plain background

#### **Green Text** ✅
- Confidence above threshold (≥0.85)
- Prediction is reliable
- Safe to type

#### **Orange Text** ⚠️
- Confidence below threshold (<0.85)
- Prediction uncertain
- Wait for better prediction

#### **"Model Preview" Window**
- Shows exactly what the model sees (64x64 image)
- Use this to verify your hand is clearly visible
- Should show your hand clearly, not blurry or cut off

### ✅ Step 4: Common Issues & Solutions

#### **Problem: Getting too many wrong predictions (like "0")**

**Causes:**
- Hand not detected properly (orange rectangle)
- Poor lighting
- Busy background
- Hand too small or too large in frame

**Solutions:**
1. Run `test_setup.py` to diagnose
2. Improve lighting (add lamp, move to brighter area)
3. Use plain background (hang dark cloth behind you)
4. Move hand closer/farther to get right size
5. Make sure green rectangle appears around hand

#### **Problem: Hand not detected (orange rectangle)**

**Solutions:**
1. **Lighting**: Add more light, avoid shadows
2. **Background**: Use plain dark/light background
3. **Distance**: Move hand closer to camera
4. **Position**: Center hand in frame
5. **Size**: Hand should be clearly visible but not fill entire frame

#### **Problem: Low confidence scores**

**Solutions:**
1. Improve lighting (most common issue)
2. Use plain background
3. Hold sign more clearly and steadily
4. Make sure hand is well-lit (no shadows)
5. Check "Model Preview" - hand should be clear

#### **Problem: Camera not working**

**Solutions:**
1. Check camera is connected
2. Close other apps using camera
3. Try different camera (if available)
4. Restart computer if needed

### ✅ Step 5: Best Practices

1. **Start with setup test**: Always run `test_setup.py` first
2. **Use manual mode**: Press 't' to type (prevents mistakes)
3. **Check preview window**: See what model sees
4. **Start simple**: Test with clear signs like 'A', 'B', 'C' first
5. **Wait for stability**: Don't rush - wait for stable predictions
6. **Good lighting is key**: This is the #1 factor for success

### ✅ Step 6: Quick Reference

**Optimal Setup:**
```
[Camera] ← 1-2 feet ← [Your Hand] ← [Plain Dark Background]
           ↑
      Good lighting from front/side
```

**What to see:**
- ✅ Green rectangle around hand
- ✅ Green text with confidence > 0.85
- ✅ Clear hand in "Model Preview" window
- ✅ Hand fills 60-80% of detection area

**What to avoid:**
- ✗ Orange rectangle (hand not detected)
- ✗ Shadows on hand
- ✗ Busy background
- ✗ Hand too small or too large
- ✗ Poor lighting

### ✅ Step 7: Testing Your Setup

1. **Run setup test:**
   ```bash
   python test_setup.py
   ```
   - Should show "READY - All systems good!"
   - Hand detection rate should be > 80%

2. **Run real-time recognition:**
   ```bash
   python real_time_prediction.py
   ```
   - Should see green rectangle
   - Should see clear hand in preview window
   - Try simple signs first ('A', 'B', '1', '2')

3. **If not working:**
   - Check lighting (most common issue)
   - Check background (use plain dark/light)
   - Check hand position (center, right size)
   - Run setup test again

### Summary

**The 3 Most Important Things:**
1. **Good Lighting** - Even, bright, no shadows
2. **Plain Background** - Dark or light, no patterns
3. **Proper Hand Position** - Center, right size, steady

If you have these three, the system will work well! 🎯

