import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import pyautogui
import time

# Try to import MediaPipe - use tasks API for MediaPipe 0.10+
MP_AVAILABLE = False
MP_IMPORT_METHOD = None

try:
    import mediapipe as mp
    # Try tasks API (MediaPipe 0.10+)
    try:
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.core import base_options
        from mediapipe import Image, ImageFormat
        from mediapipe.tasks.python.vision.core import vision_task_running_mode
        MP_AVAILABLE = True
        MP_IMPORT_METHOD = 'tasks'
    except ImportError as e:
        print(f"MediaPipe tasks API not available: {e}")
        MP_AVAILABLE = False
except ImportError:
    MP_AVAILABLE = False

# Configuration
IMG_SIZE = 64
MODEL_PATH = 'asl_model.keras'
CLASS_NAMES_PATH = 'class_names.pkl'
CONFIDENCE_THRESHOLD = 0.70  # Lower threshold for faster response
FILTER_UNKNOWN = True  # Filter out "unknown" predictions unless very confident
UNKNOWN_THRESHOLD = 0.95  # Only accept "unknown" if confidence is very high (increased from 0.90)
PREDICTION_DELAY = 0.1  # Much faster for real-time (was 0.2)
STABLE_PREDICTIONS_NEEDED = 3  # Fewer needed for faster response (was 5)
MIN_TIME_BETWEEN_TYPING = 1.0  # Faster typing response
MIN_TIME_BETWEEN_ANY_TYPING = 0.5  # Much faster
USE_ENSEMBLE = False  # Disable ensemble for speed (was True)
CONFIDENCE_SMOOTHING_WINDOW = 3  # Smaller window for faster response
DEBUG_MODE = True  # Set to True to see ROI being sent to model
SAVE_ROI_DEBUG = True  # Set to True to save ROI images for debugging

class ASLRealTimePredictor:
    def __init__(self):
        """Initialize the real-time ASL predictor"""
        print("Loading ASL recognition model...")
        try:
            self.model = keras.models.load_model(MODEL_PATH)
        except Exception as e:
            try:
                h5_path = MODEL_PATH.replace('.keras', '.h5')
                print(f"Trying fallback format: {h5_path}")
                self.model = keras.models.load_model(h5_path)
            except:
                raise FileNotFoundError(f"Could not load model. Please train the model first.")
        
        print("Loading class names...")
        try:
            with open(CLASS_NAMES_PATH, 'rb') as f:
                self.class_names = pickle.load(f)
        except Exception as e:
            raise FileNotFoundError(f"Could not load class names from {CLASS_NAMES_PATH}")
        
        print(f"Model loaded successfully! Classes: {self.class_names}")
        
        # Initialize hand detection - REQUIRE MediaPipe (no fallback)
        if not MP_AVAILABLE:
            raise RuntimeError(
                "MediaPipe is required but not available. Please install it with: pip install mediapipe\n"
                "Or if using MediaPipe 0.10+, ensure the tasks API is available."
            )
        
        print("Initializing MediaPipe hand detection...")
        if MP_IMPORT_METHOD == 'tasks':
            try:
                # Use tasks API (MediaPipe 0.10+)
                # Download model automatically from URL if not available locally
                import os
                model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
                model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                
                if not os.path.exists(model_path):
                    print("Downloading hand landmarker model...")
                    import urllib.request
                    urllib.request.urlretrieve(model_url, model_path)
                    print("Model downloaded successfully!")
                
                # Create options - Lower thresholds for better detection
                options = HandLandmarkerOptions(
                    base_options=base_options.BaseOptions(model_asset_path=model_path),
                    running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.3,  # Lowered from 0.5 for better detection
                    min_hand_presence_confidence=0.3,  # Lowered from 0.5 for better detection
                    min_tracking_confidence=0.3  # Lowered from 0.5 for better tracking
                )
                
                # Create HandLandmarker
                self.hand_landmarker = HandLandmarker.create_from_options(options)
                self.use_mediapipe = True
                self.mp_import_method = 'tasks'
                print("MediaPipe hand detection ready! (tasks API)")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize MediaPipe HandLandmarker: {e}\n"
                                 "Please ensure MediaPipe is properly installed: pip install mediapipe")
        else:
            raise RuntimeError(f"Unsupported MediaPipe import method: {MP_IMPORT_METHOD}")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera. Please check your camera connection.")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.last_prediction_time = 0
        self.current_text = ""
        self.prediction_history = []
        self.last_typed_char = None
        self.last_typed_time = 0
        self.stable_prediction_count = 0
        self.current_stable_char = None
        self.frame_timestamp_ms = 0  # For MediaPipe video detection
        self.confidence_history = []  # Store confidence scores for smoothing
        
        # Optimize model for inference speed
        try:
            # Enable optimizations for faster inference
            tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
        except:
            pass  # XLA not available, continue without it
    
    def detect_hand_region(self, frame):
        """Detect hand region using MediaPipe (REQUIRED - no fallback)"""
        h, w = frame.shape[:2]
        
        if not self.use_mediapipe:
            raise RuntimeError("MediaPipe is required but not initialized")
        
        # Use MediaPipe tasks API for hand detection
        if self.mp_import_method == 'tasks':
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hand landmarks (timestamp in milliseconds)
            self.frame_timestamp_ms += 33  # ~30 FPS = 33ms per frame
            try:
                result = self.hand_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠ MediaPipe detection error: {e}")
                result = None
            
            if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand_landmarks = result.hand_landmarks[0]
                x_coords = [landmark.x * w for landmark in hand_landmarks]
                y_coords = [landmark.y * h for landmark in hand_landmarks]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Calculate hand bounding box
                hand_width = x_max - x_min
                hand_height = y_max - y_min
                hand_center_x = (x_min + x_max) // 2
                hand_center_y = (y_min + y_max) // 2
                
                # CRITICAL: Match training data format - hand should fill ~20% of 224x224 image
                # Training images: 224x224 with hand taking ~21% of image area
                # Problem: 400x400 ROI includes too much background, hand becomes too small when resized
                # Solution: Extract smaller ROI so hand fills appropriate portion
                hand_max_dimension = max(hand_width, hand_height)
                
                # Calculate ROI size: hand should be ~20-25% of ROI area
                # If hand is 100px, we want ROI where hand is ~20% = sqrt(100/0.2) ≈ 224px
                # Use 2.0-2.5x hand size to get proper hand-to-background ratio
                # This ensures hand fills ~16-25% of ROI (matching training ~21%)
                target_roi_size = int(hand_max_dimension * 2.2)  # 2.2x gives ~20% hand coverage
                
                # Clamp to reasonable range: 224-280px (not too large!)
                # Training images are 224x224, so we want similar size
                target_roi_size = max(224, min(280, target_roi_size))
                target_roi_size = min(target_roi_size, min(w, h))  # Don't exceed frame
                
                # Center ROI on hand center
                x = max(0, hand_center_x - target_roi_size // 2)
                y = max(0, hand_center_y - target_roi_size // 2)
                
                # Adjust if we hit frame boundaries
                if x + target_roi_size > w:
                    x = w - target_roi_size
                if y + target_roi_size > h:
                    y = h - target_roi_size
                x = max(0, x)
                y = max(0, y)
                
                # Final ROI (must be square)
                w_roi = min(target_roi_size, w - x)
                h_roi = min(target_roi_size, h - y)
                final_size = min(w_roi, h_roi)
                
                # Re-center to ensure perfect square while keeping hand centered
                ideal_x = hand_center_x - final_size // 2
                ideal_y = hand_center_y - final_size // 2
                
                # Clamp ideal position to frame bounds
                x = max(0, min(ideal_x, w - final_size))
                y = max(0, min(ideal_y, h - final_size))
                
                w_roi = h_roi = final_size
                
                # Accept ROI if it's at least 200px (will be resized to 224x224 later)
                if w_roi >= 200 and h_roi >= 200:
                    if DEBUG_MODE:
                        print(f"✓ Hand detected! ROI: {w_roi}x{h_roi} @ ({x},{y})")
                    return x, y, w_roi, h_roi, True, hand_landmarks
                else:
                    if DEBUG_MODE:
                        print(f"⚠ Hand detected but ROI size too small: {w_roi}x{h_roi} (need >= 224px)")
        
        # No hand detected - return center crop matching training size exactly
        # Training data: original images (any size) resized directly to 64x64
        # Use 224x224 center crop for consistency (will be resized to 64x64 in preprocessing)
        target_size = 224
        # For 640x480 frames, use exactly 224px
        if min(w, h) >= 224:
            roi_size = 224
        else:
            # For smaller frames, use as much as possible
            roi_size = min(w, h)
        roi_size = min(roi_size, min(w, h))  # Don't exceed frame size
        x = int((w - roi_size) / 2)
        y = int((h - roi_size) / 2)
        if DEBUG_MODE:
            print(f"⚠ No hand detected - using center crop: {roi_size}x{roi_size} @ ({x},{y})")
        return x, y, roi_size, roi_size, False, None
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction - EXACTLY as training"""
        # IMPORTANT: Keep BGR format to match training (cv2.imread uses BGR)
        # Do NOT convert to RGB - training uses BGR format
        
        # Handle empty or invalid ROI
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            # Return black image if ROI is invalid
            black_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            return np.expand_dims(black_img, axis=0)
        
        # Training data: 224x224 square images resized to 64x64
        # ROI should already be square at this point
        # Resize using same method as training (INTER_AREA is good for downscaling)
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # Normalize exactly as training
        normalized = resized.astype('float32') / 255.0
        
        return np.expand_dims(normalized, axis=0)
    
    def predict(self, frame):
        """Predict ASL sign from frame"""
        preprocessed = self.preprocess_frame(frame)
        predictions = self.model.predict(preprocessed, verbose=0)
        confidence = np.max(predictions[0])
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3_predictions = [(self.class_names[i], predictions[0][i]) for i in top3_indices]
        
        return predicted_class, confidence, top3_predictions
    
    def predict_ensemble(self, frames):
        """Predict using ensemble of multiple frames - OPTIMIZED with batch prediction"""
        if not frames or len(frames) == 0:
            return None, 0.0, []
        
        # Batch preprocess all frames at once (much faster)
        preprocessed_batch = []
        for frame in frames:
            preprocessed = self.preprocess_frame(frame)
            preprocessed_batch.append(preprocessed[0])  # Remove batch dimension
        
        # Batch predict all frames at once (single model call instead of N calls)
        batch_input = np.array(preprocessed_batch)
        all_predictions = self.model.predict(batch_input, verbose=0)
        
        # Average predictions across frames
        ensemble_pred = np.mean(all_predictions, axis=0)
        
        # Get final prediction from ensemble
        confidence = np.max(ensemble_pred)
        predicted_class_idx = np.argmax(ensemble_pred)
        predicted_class = self.class_names[predicted_class_idx]
        
        top3_indices = np.argsort(ensemble_pred)[-3:][::-1]
        top3_predictions = [(self.class_names[i], ensemble_pred[i]) for i in top3_indices]
        
        return predicted_class, confidence, top3_predictions
    
    def is_number(self, char):
        """Check if character is a number"""
        return char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    def is_letter(self, char):
        """Check if character is a letter"""
        return char in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    def type_character(self, character):
        """Type the predicted character"""
        try:
            pyautogui.write(character)
            print(f"✓ Typed: {character}")
            return True
        except Exception as e:
            print(f"✗ Could not type: {e}")
            return False
    
    def draw_ui(self, frame, roi, x, y, w, h, hand_detected, predicted_class, 
                confidence, top3, hand_roi_img, hand_landmarks):
        """Draw comprehensive UI with all information"""
        h_frame, w_frame = frame.shape[:2]
        
        overlay = frame.copy()
        status_bar_height = 120
        cv2.rectangle(overlay, (0, 0), (w_frame, status_bar_height), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw hand landmarks if using MediaPipe (tasks API doesn't have drawing utils in the same way)
        # We'll draw basic landmarks manually if needed
        if hand_landmarks and self.use_mediapipe and self.mp_import_method == 'tasks':
            try:
                # Draw hand landmarks as circles (use different variable names to avoid shadowing ROI x,y)
                for landmark in hand_landmarks:
                    lm_x = int(landmark.x * w_frame)
                    lm_y = int(landmark.y * h_frame)
                    cv2.circle(frame, (lm_x, lm_y), 3, (0, 255, 0), -1)
            except:
                pass
        
        # Draw ROI rectangle
        color = (0, 255, 0) if hand_detected else (0, 165, 255)
        thickness = 3 if hand_detected else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Corner markers
        corner_size = 15
        cv2.line(frame, (x, y), (x + corner_size, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_size), color, thickness)
        cv2.line(frame, (x + w, y), (x + w - corner_size, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_size), color, thickness)
        cv2.line(frame, (x, y + h), (x + corner_size, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_size), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w - corner_size, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_size), color, thickness)
        
        # Status text
        if hand_detected:
            status_text = "✓ HAND DETECTED (MediaPipe)"
        else:
            status_text = "⚠ NO HAND - Position hand in CENTER"
        cv2.putText(frame, status_text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw center region guide when no hand detected
        if not hand_detected:
            h_frame, w_frame = frame.shape[:2]
            center_guide_size = min(w_frame, h_frame) * 0.5
            center_x = w_frame // 2
            center_y = h_frame // 2
            guide_x1 = int(center_x - center_guide_size // 2)
            guide_y1 = int(center_y - center_guide_size // 2)
            guide_x2 = int(center_x + center_guide_size // 2)
            guide_y2 = int(center_y + center_guide_size // 2)
            
            # Draw dashed rectangle for center guide
            dash_length = 10
            gap = 5
            # Top
            for i in range(guide_x1, guide_x2, dash_length + gap):
                cv2.line(frame, (i, guide_y1), (min(i + dash_length, guide_x2), guide_y1), (255, 255, 0), 2)
            # Bottom
            for i in range(guide_x1, guide_x2, dash_length + gap):
                cv2.line(frame, (i, guide_y2), (min(i + dash_length, guide_x2), guide_y2), (255, 255, 0), 2)
            # Left
            for i in range(guide_y1, guide_y2, dash_length + gap):
                cv2.line(frame, (guide_x1, i), (guide_x1, min(i + dash_length, guide_y2)), (255, 255, 0), 2)
            # Right
            for i in range(guide_y1, guide_y2, dash_length + gap):
                cv2.line(frame, (guide_x2, i), (guide_x2, min(i + dash_length, guide_y2)), (255, 255, 0), 2)
            
            cv2.putText(frame, "POSITION HAND HERE", (guide_x1, guide_y1 - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Main prediction display
        y_offset = 35
        pred_text = f"PREDICTED: {predicted_class.upper()}"
        conf_text = f"CONFIDENCE: {confidence:.1%}"
        
        if confidence >= CONFIDENCE_THRESHOLD:
            text_color = (0, 255, 0)
            status = "✓ READY"
        elif confidence >= 0.6:
            text_color = (0, 200, 255)
            status = "⚠ LOW CONFIDENCE"
        else:
            text_color = (0, 0, 255)
            status = "✗ TOO LOW - Position hand in center"
        
        cv2.putText(frame, pred_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3)
        cv2.putText(frame, conf_text, (10, y_offset + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(frame, status, (10, y_offset + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Top 3 predictions
        top3_x = w_frame - 200
        cv2.putText(frame, "TOP 3:", (top3_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for i, (char, conf) in enumerate(top3):
            color_top = (0, 255, 0) if i == 0 else (200, 200, 200)
            cv2.putText(frame, f"{i+1}. {char} ({conf:.2f})", 
                       (top3_x, y_offset + 25 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_top, 1)
        
        # Stable prediction indicator
        if self.current_stable_char:
            stable_y = y_offset + 85
            stable_text = f"STABLE: {self.current_stable_char.upper()} ({self.stable_prediction_count}/{STABLE_PREDICTIONS_NEEDED})"
            stable_color = (0, 255, 255)
            cv2.putText(frame, stable_text, (10, stable_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, stable_color, 2)
        
        # Model preview (shows what's being sent to model)
        preview_size = 128
        preview_x = w_frame - preview_size - 10
        preview_y = h_frame - preview_size - 10
        
        # Show the actual preprocessed image that goes to model
        if hand_roi_img.size > 0:
            # Resize ROI to show what model sees
            model_input_preview = cv2.resize(hand_roi_img, (IMG_SIZE, IMG_SIZE))
            model_input_preview = cv2.resize(model_input_preview, (preview_size, preview_size), interpolation=cv2.INTER_NEAREST)
        else:
            model_input_preview = np.zeros((preview_size, preview_size, 3), dtype=np.uint8)
        
        preview_frame = np.zeros((preview_size + 20, preview_size + 20, 3), dtype=np.uint8)
        preview_frame[10:10+preview_size, 10:10+preview_size] = model_input_preview
        cv2.rectangle(preview_frame, (5, 5), (preview_size + 15, preview_size + 15), (255, 255, 255), 2)
        cv2.putText(preview_frame, "MODEL INPUT", (15, preview_size + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        frame[preview_y-10:preview_y+preview_size+10, preview_x-10:preview_x+preview_size+10] = preview_frame
        
        # Debug: Show ROI info and hand detection status
        if DEBUG_MODE:
            debug_text = f"ROI: {w}x{h} @ ({x},{y}) | Hand: {'YES' if hand_detected else 'NO'}"
            cv2.putText(frame, debug_text, (10, h_frame - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            # Show MediaPipe status
            mp_status = "MediaPipe: ACTIVE" if self.use_mediapipe else "MediaPipe: INACTIVE"
            cv2.putText(frame, mp_status, (10, h_frame - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Typed text display
        text_y = h_frame - 60
        cv2.rectangle(frame, (0, text_y - 5), (w_frame, h_frame), (0, 0, 0), -1)
        cv2.putText(frame, "TYPED TEXT:", (10, text_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        display_text = self.current_text[-50:] if len(self.current_text) > 50 else self.current_text
        if display_text:
            cv2.putText(frame, display_text, (10, text_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "(No text typed yet)", (10, text_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Instructions
        instructions = ["Q: Quit | C: Clear | S: Show text"]
        for i, inst in enumerate(instructions):
            cv2.putText(frame, inst, (10, h_frame - 15 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """Run real-time prediction loop"""
        print("\n" + "="*70)
        print(f"ASL REAL-TIME RECOGNITION - AUTO TYPING MODE")
        print(f"Using MediaPipe hand detection")
        print("="*70)
        print("Instructions:")
        print("  - Position your hand in the center of the camera")
        print("  - Hold your sign steady for best results")
        print("  - The system will automatically type when confident")
        print("  - Press 'q' to quit, 'c' to clear text, 's' to show text")
        print("="*70 + "\n")
        
        print("Starting camera feed... Press 'q' to quit.\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            x, y, w, h, hand_detected, hand_landmarks = self.detect_hand_region(frame)
            
            # Ensure valid ROI bounds
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = max(1, min(w, frame.shape[1] - x))
            h = max(1, min(h, frame.shape[0] - y))
            
            # Extract ROI
            roi = frame[y:y+h, x:x+w].copy() if w > 0 and h > 0 else frame.copy()
            
            # Training data: 224x224 square images resized to 64x64
            # Ensure ROI is square to match training format
            if roi.size > 0:
                h_roi, w_roi = roi.shape[:2]
                if h_roi != w_roi:
                    # Make square by cropping to center (training uses square images)
                    size = min(h_roi, w_roi)
                    y_start = (h_roi - size) // 2
                    x_start = (w_roi - size) // 2
                    roi = roi[y_start:y_start+size, x_start:x_start+size]
                    w = h = size
            
            hand_roi_img = roi.copy()
            
            # Only make predictions if ROI is valid and size is at least 200px
            # Training data: 224x224 images resized directly to 64x64
            # We extract ROI with padding (200-400px), resize to 224x224, then to 64x64 in preprocessing
            if roi.size > 0 and w >= 200 and h >= 200:
                # ALWAYS resize ROI to 224x224 to match training data format exactly
                # This standardizes the size regardless of original ROI size (200-400px)
                roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
                w = h = 224
                current_time = time.time()
                if current_time - self.last_prediction_time >= PREDICTION_DELAY:
                    # Fast single-frame prediction (ensemble disabled for speed)
                    predicted_class, confidence, top3 = self.predict(roi)
                    
                    # Light confidence smoothing (smaller window for speed)
                    self.confidence_history.append(confidence)
                    if len(self.confidence_history) > CONFIDENCE_SMOOTHING_WINDOW:
                        self.confidence_history.pop(0)
                    smoothed_confidence = np.mean(self.confidence_history)
                    
                    # Use smoothed confidence for filtering
                    confidence = smoothed_confidence
                    
                    # Debug: Save ROI if enabled (save the actual ROI before preprocessing)
                    if SAVE_ROI_DEBUG and confidence > 0.5:
                        import os
                        os.makedirs('debug_rois', exist_ok=True)
                        # Save the original ROI (hand_roi_img) not the preprocessed one
                        # Ensure it's in uint8 format
                        if hand_roi_img.dtype != np.uint8:
                            roi_to_save = (hand_roi_img * 255).astype(np.uint8) if hand_roi_img.dtype == np.float32 else hand_roi_img.astype(np.uint8)
                        else:
                            roi_to_save = hand_roi_img.copy()
                        # Resize to 224x224 if needed for consistency
                        if roi_to_save.shape[0] != 224 or roi_to_save.shape[1] != 224:
                            roi_to_save = cv2.resize(roi_to_save, (224, 224), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(f'debug_rois/{predicted_class}_{confidence:.2f}_{int(time.time()*1000)}.jpg', roi_to_save)
                    
                    # Filter predictions
                    roi_size_valid = 200 <= w <= 250 and 200 <= h <= 250
                    is_ambiguous = False
                    
                    # Check for number/letter confusion - require higher confidence
                    if len(top3) >= 2:
                        top1_class, top1_conf = top3[0]
                        top2_class, top2_conf = top3[1]
                        
                        # If confusing between number and letter, require higher confidence
                        if ((self.is_number(top1_class) and self.is_letter(top2_class)) or 
                            (self.is_letter(top1_class) and self.is_number(top2_class))):
                            # If confidences are close (within 15%), require higher threshold
                            if abs(top1_conf - top2_conf) < 0.15:
                                # Require higher confidence for ambiguous cases
                                if confidence < 0.80:
                                    is_ambiguous = True
                    
                    if not roi_size_valid:
                        # ROI size is wrong - skip this prediction
                        pass
                    elif is_ambiguous:
                        # Too ambiguous between number/letter - skip
                        pass
                    elif FILTER_UNKNOWN and predicted_class == 'unknown' and confidence < UNKNOWN_THRESHOLD:
                        # Skip unknown predictions unless very confident
                        pass
                    elif confidence >= CONFIDENCE_THRESHOLD:
                        # Valid prediction - add to history
                        self.prediction_history.append((predicted_class, confidence))
                        if len(self.prediction_history) > 10:  # Smaller history for speed
                            self.prediction_history.pop(0)
                        # Use longer history window for more stable predictions
                        window_size = max(STABLE_PREDICTIONS_NEEDED, len(self.prediction_history))
                        recent_predictions = [p[0] for p in self.prediction_history[-window_size:]]
                        recent_confidences = [p[1] for p in self.prediction_history[-window_size:]]
                        
                        if len(recent_predictions) >= STABLE_PREDICTIONS_NEEDED:
                            # Find most common prediction
                            most_common = max(set(recent_predictions), key=recent_predictions.count)
                            count = recent_predictions.count(most_common)
                            
                            # Calculate average confidence for most common prediction
                            most_common_confidences = [conf for pred, conf in zip(recent_predictions, recent_confidences) if pred == most_common]
                            avg_confidence = np.mean(most_common_confidences) if most_common_confidences else confidence
                            
                            # Require both stability (count) and confidence
                            if count >= STABLE_PREDICTIONS_NEEDED and avg_confidence >= CONFIDENCE_THRESHOLD:
                                if most_common != self.current_stable_char:
                                    self.current_stable_char = most_common
                                    self.stable_prediction_count = count
                                else:
                                    self.stable_prediction_count = count
                                
                                time_since_last_type = current_time - self.last_typed_time
                                # Only type if:
                                # 1. Hand is actually detected (not using center region)
                                # 2. Either different character OR enough time passed for same character  
                                # 3. Minimum time between any typing
                                if ((most_common != self.last_typed_char or time_since_last_type > MIN_TIME_BETWEEN_TYPING) and
                                    time_since_last_type >= MIN_TIME_BETWEEN_ANY_TYPING):
                                    if self.type_character(most_common):
                                        self.current_text += most_common
                                        self.last_typed_char = most_common
                                        self.last_typed_time = current_time
                                        self.prediction_history = []  # Clear after typing
                                        self.current_stable_char = None
                                        print(f"  → Typed '{most_common}' (confidence: {avg_confidence:.2%}, stable: {count}/{STABLE_PREDICTIONS_NEEDED})")
                                        # Clear confidence history after successful prediction
                                        self.confidence_history = []
                            else:
                                self.current_stable_char = None
                        else:
                            self.current_stable_char = predicted_class
                            self.stable_prediction_count = len(recent_predictions)
                    else:
                        self.current_stable_char = None
                        self.stable_prediction_count = 0
                    
                    self.last_prediction_time = current_time
                else:
                    if self.prediction_history:
                        predicted_class, confidence = self.prediction_history[-1]
                        top3 = [(predicted_class, confidence)]
                    else:
                        predicted_class, confidence, top3 = "?", 0.0, []
            else:
                predicted_class, confidence, top3 = "?", 0.0, []
                hand_landmarks = None
            
            frame = self.draw_ui(frame, roi, x, y, w, h, hand_detected, 
                               predicted_class, confidence, top3, hand_roi_img, hand_landmarks)
            
            cv2.imshow('ASL Real-Time Recognition - Auto Typing', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.current_text = ""
                self.prediction_history = []
                self.current_stable_char = None
                print("Text cleared")
            elif key == ord('s'):
                print(f"Current text: {self.current_text}")
            elif key == ord(' '):
                self.current_text += " "
                self.type_character(" ")
        
        self.cap.release()
        if self.use_mediapipe and self.mp_import_method == 'tasks':
            try:
                self.hand_landmarker.close()
            except:
                pass
        cv2.destroyAllWindows()
        print(f"\nFinal text: {self.current_text}")
        print("Application closed")

def main():
    try:
        predictor = ASLRealTimePredictor()
        predictor.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease make sure 'asl_model.keras' (or 'asl_model.h5') and 'class_names.pkl' exist.")
        print("Run 'python train_model.py' first to train the model.")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Closing...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
