# ASL Real-Time Recognition - Usage Guide

## Tips for Best Results

### 1. **Hand Positioning**
- Position your hand in the **center** of the camera view
- Keep your hand **steady** - don't move it around too much
- Make sure your **entire hand** is visible in the green rectangle
- The hand should fill about **60-80%** of the detection area

### 2. **Lighting**
- Use **good, even lighting** - avoid shadows on your hand
- Avoid backlighting (don't have bright light behind you)
- Natural daylight or bright room lighting works best
- Avoid harsh shadows that split your hand

### 3. **Background**
- Use a **plain, contrasting background** (not the same color as your skin)
- A dark or light wall works well
- Avoid busy backgrounds with patterns
- The model was trained on black backgrounds, so dark backgrounds work best

### 4. **Hand Gesture**
- Make the sign **clearly and distinctly**
- Hold the sign **steady** for 1-2 seconds
- Ensure your fingers are **clearly separated** (not overlapping)
- Make sure the sign matches the **ASL alphabet** format

### 5. **Using the Application**

#### Manual Mode (Recommended)
- The app starts in **Manual Mode** by default
- When you see a stable prediction with high confidence:
  - Press **'t'** to type the character
  - This prevents accidental typing of wrong predictions

#### Auto Mode
- Change `MANUAL_MODE = False` in the script to enable auto-typing
- Characters will be typed automatically when confidence is high

### 6. **Troubleshooting**

**Problem: Getting too many "0" predictions**
- Your hand might not be detected properly
- Check if the green rectangle appears around your hand
- If you see orange rectangle, hand detection failed - try better lighting
- Make sure your hand is large enough in the frame

**Problem: Low confidence scores**
- Improve lighting
- Use a plain background
- Make sure your hand is clearly visible
- Hold the sign steady for longer

**Problem: Wrong predictions**
- Make sure you're making the correct ASL sign
- Check the "Model Preview" window to see what the model sees
- Try holding the sign more clearly
- Some signs are similar (like '0' and 'O') - be very distinct

### 7. **Controls**
- **'q'** - Quit the application
- **'c'** - Clear typed text
- **'s'** - Show current text in console
- **'t'** - Type the current stable prediction (Manual Mode)
- **'space'** - Add a space

### 8. **Understanding the Display**

- **Green rectangle** = Hand detected successfully
- **Orange rectangle** = Using center region (hand not detected)
- **Green text** = Confidence above threshold (good prediction)
- **Orange text** = Confidence below threshold (uncertain)
- **Yellow text** = Stable prediction ready to type

### 9. **Best Practices**

1. **Start with simple signs** (like 'A', 'B', 'C') to test
2. **Wait for stable predictions** before typing
3. **Check the preview window** to see what the model sees
4. **Use Manual Mode** to avoid mistakes
5. **Clear text frequently** if you make mistakes

### 10. **Common ASL Signs Reference**

Make sure you're using the correct ASL alphabet signs:
- **A**: Fist with thumb to side
- **B**: Flat hand, palm forward
- **C**: Curved hand like 'C'
- **0**: Circle with thumb and index finger
- **1**: Index finger up
- etc.

The model was trained on specific ASL signs, so make sure your gestures match!

