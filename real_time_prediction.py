import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import pyautogui
import time

# Try to import MediaPipe with multiple fallback methods
MP_AVAILABLE = False
MP_IMPORT_METHOD = None
mp_hands_module = None
mp_drawing_module = None

try:
    import mediapipe as mp
    # Method 1: Try new tasks API (MediaPipe 0.10+)
    try:
        from mediapipe.tasks.python.vision import hand_landmarker
        MP_AVAILABLE = True
        MP_IMPORT_METHOD = 'tasks'
    except ImportError:
        # Method 2: Try classic solutions API
        try:
            mp_hands_module = mp.solutions.hands
            mp_drawing_module = mp.solutions.drawing_utils
            MP_AVAILABLE = True
            MP_IMPORT_METHOD = 'solutions'
        except AttributeError:
            # Method 3: Try direct import
            try:
                from mediapipe.python.solutions import hands as mp_hands_module
                from mediapipe.python.solutions import drawing_utils as mp_drawing_module
                MP_AVAILABLE = True
                MP_IMPORT_METHOD = 'direct'
            except ImportError:
                MP_AVAILABLE = False
except ImportError:
    MP_AVAILABLE = False

# Configuration
IMG_SIZE = 64
MODEL_PATH = 'asl_model.keras'
CLASS_NAMES_PATH = 'class_names.pkl'
CONFIDENCE_THRESHOLD = 0.90  # Higher threshold for accuracy
PREDICTION_DELAY = 0.5  # Slower predictions for stability
STABLE_PREDICTIONS_NEEDED = 6  # More consistent predictions needed
MIN_TIME_BETWEEN_TYPING = 3.0  # Minimum 3 seconds between typing same character
MIN_TIME_BETWEEN_ANY_TYPING = 1.5  # Minimum time between any typing

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
        
        # Initialize hand detection
        self.use_mediapipe = False
        if MP_AVAILABLE and MP_IMPORT_METHOD == 'solutions':
            print("Initializing MediaPipe hand detection...")
            try:
                self.mp_hands = mp_hands_module
                self.mp_drawing = mp_drawing_module
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                self.use_mediapipe = True
                print("MediaPipe hand detection ready!")
            except Exception as e:
                print(f"Warning: Could not initialize MediaPipe: {e}")
                print("Falling back to improved color-based detection...")
                self.setup_color_detection()
        else:
            if not MP_AVAILABLE:
                print("MediaPipe not installed. Using improved color-based detection...")
            else:
                print(f"MediaPipe available but using {MP_IMPORT_METHOD} API (not yet implemented).")
                print("Falling back to improved color-based detection...")
            self.setup_color_detection()
        
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
    
    def setup_color_detection(self):
        """Setup color-based hand detection as fallback"""
        self.lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        self.lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        self.upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    
    def detect_hand_color_based(self, frame):
        """Improved color-based hand detection with center focus"""
        h, w = frame.shape[:2]
        
        # Focus on center region (where hands are typically positioned)
        center_region_ratio = 0.6  # Use 60% of frame centered
        center_x_start = int(w * (1 - center_region_ratio) / 2)
        center_y_start = int(h * (1 - center_region_ratio) / 2)
        center_x_end = int(w * (1 + center_region_ratio) / 2)
        center_y_end = int(h * (1 + center_region_ratio) / 2)
        
        # Extract center region for processing
        center_region = frame[center_y_start:center_y_end, center_x_start:center_x_end].copy()
        
        if center_region.size == 0:
            # Fallback to full frame
            center_region = frame.copy()
            center_x_start = center_y_start = 0
        
        hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_skin1, self.upper_skin1)
        mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # More aggressive morphological operations
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        
        # Remove small noise
        mask = cv2.erode(mask, kernel_small, iterations=1)
        mask = cv2.dilate(mask, kernel_large, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            hand_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # More restrictive size range for hands
                if 5000 < area < 40000:  # Hand-sized objects
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    
                    # Adjust coordinates to full frame
                    x += center_x_start
                    y += center_y_start
                    
                    aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0
                    frame_h, frame_w = frame.shape[:2]
                    center_y_ratio = (y + h_rect // 2) / frame_h
                    center_x_ratio = (x + w_rect // 2) / frame_w
                    
                    # Strict filtering: hands are in center area, roughly square
                    if (0.6 < aspect_ratio < 1.4 and  # More square
                        0.25 < center_y_ratio < 0.75 and  # Center vertical area
                        0.2 < center_x_ratio < 0.8):  # Center horizontal area
                        
                        # Check for convexity defects (fingers)
                        score = 0.5
                        try:
                            hull = cv2.convexHull(contour, returnPoints=False)
                            if len(hull) > 3:
                                defects = cv2.convexityDefects(contour, hull)
                                if defects is not None:
                                    defect_count = len(defects)
                                    if defect_count >= 3:  # Has finger-like features
                                        score = 1.0
                                    elif defect_count >= 1:
                                        score = 0.7
                        except:
                            pass
                        
                        # Bonus for being in ideal center position
                        if 0.35 < center_y_ratio < 0.65 and 0.3 < center_x_ratio < 0.7:
                            score += 0.2
                        
                        # Bonus for good aspect ratio (close to square)
                        if 0.8 < aspect_ratio < 1.2:
                            score += 0.1
                        
                        hand_candidates.append((contour, area, score, x, y, w_rect, h_rect))
            
            if hand_candidates:
                # Sort by score, then by proximity to center
                hand_candidates.sort(key=lambda x: (x[2], -abs(x[3] + x[5]//2 - w//2) - abs(x[4] + x[6]//2 - h//2)), reverse=True)
                best_contour, _, _, x, y, w_rect, h_rect = hand_candidates[0]
                
                padding = 40
                x = max(0, x - padding)
                y = max(0, y - padding)
                w_rect = min(frame.shape[1] - x, w_rect + 2 * padding)
                h_rect = min(frame.shape[0] - y, h_rect + 2 * padding)
                
                # Make it square
                size = max(w_rect, h_rect)
                center_x = x + w_rect // 2
                center_y = y + h_rect // 2
                x = max(0, center_x - size // 2)
                y = max(0, center_y - size // 2)
                w_rect = h_rect = min(size, min(frame.shape[1] - x, frame.shape[0] - y))
                
                if w_rect > 100 and h_rect > 100:  # Minimum size increased
                    return x, y, w_rect, h_rect, True, None
        
        # Fallback to center crop (preferred region)
        h, w = frame.shape[:2]
        roi_size = min(w, h) * 0.55  # Slightly smaller, more focused
        x = int((w - roi_size) / 2)
        y = int((h - roi_size) / 2)
        return x, y, int(roi_size), int(roi_size), False, None
    
    def detect_hand_region(self, frame):
        """Detect hand region using MediaPipe or fallback"""
        if self.use_mediapipe:
            # Use MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w = frame.shape[:2]
                x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                padding = 40
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                width = x_max - x_min
                height = y_max - y_min
                size = max(width, height)
                
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                
                x = max(0, center_x - size // 2)
                y = max(0, center_y - size // 2)
                x_max = min(w, x + size)
                y_max = min(h, y + size)
                
                if x_max - x < size:
                    x = max(0, x_max - size)
                if y_max - y < size:
                    y = max(0, y_max - size)
                
                w_roi = x_max - x
                h_roi = y_max - y
                
                if w_roi > 80 and h_roi > 80:
                    return x, y, w_roi, h_roi, True, hand_landmarks
            
            return self.detect_hand_color_based(frame)
        else:
            # Use color-based detection
            return self.detect_hand_color_based(frame)
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = resized_rgb.astype('float32') / 255.0
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
        
        # Draw hand landmarks if using MediaPipe
        if hand_landmarks and self.use_mediapipe:
            try:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
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
        detection_method = "MediaPipe" if self.use_mediapipe else "Color-based"
        if hand_detected:
            status_text = f"✓ HAND DETECTED ({detection_method})"
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
        
        # Model preview
        preview_size = 128
        preview_x = w_frame - preview_size - 10
        preview_y = h_frame - preview_size - 10
        
        preview_frame = np.zeros((preview_size + 20, preview_size + 20, 3), dtype=np.uint8)
        preview_frame[10:10+preview_size, 10:10+preview_size] = cv2.resize(hand_roi_img, (preview_size, preview_size))
        cv2.rectangle(preview_frame, (5, 5), (preview_size + 15, preview_size + 15), (255, 255, 255), 2)
        cv2.putText(preview_frame, "MODEL INPUT", (15, preview_size + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        frame[preview_y-10:preview_y+preview_size+10, preview_x-10:preview_x+preview_size+10] = preview_frame
        
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
        method = "MediaPipe" if self.use_mediapipe else "Color-based"
        print("\n" + "="*70)
        print(f"ASL REAL-TIME RECOGNITION - AUTO TYPING MODE")
        print(f"Using {method} hand detection")
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
            roi = frame[y:y+h, x:x+w].copy() if w > 0 and h > 0 else frame.copy()
            hand_roi_img = roi.copy()
            
            if roi.size > 0 and w > 50 and h > 50:
                current_time = time.time()
                if current_time - self.last_prediction_time >= PREDICTION_DELAY:
                    predicted_class, confidence, top3 = self.predict(roi)
                    
                    self.prediction_history.append((predicted_class, confidence))
                    if len(self.prediction_history) > 10:
                        self.prediction_history.pop(0)
                    
                    # Only process if hand is detected AND confidence is high
                    if hand_detected and confidence >= CONFIDENCE_THRESHOLD:
                        recent_predictions = [p[0] for p in self.prediction_history[-STABLE_PREDICTIONS_NEEDED:]]
                        if len(recent_predictions) >= STABLE_PREDICTIONS_NEEDED:
                            most_common = max(set(recent_predictions), key=recent_predictions.count)
                            count = recent_predictions.count(most_common)
                            
                            if count >= STABLE_PREDICTIONS_NEEDED:
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
                                if (hand_detected and 
                                    (most_common != self.last_typed_char or time_since_last_type > MIN_TIME_BETWEEN_TYPING) and
                                    time_since_last_type >= MIN_TIME_BETWEEN_ANY_TYPING):
                                    if self.type_character(most_common):
                                        self.current_text += most_common
                                        self.last_typed_char = most_common
                                        self.last_typed_time = current_time
                                        self.prediction_history = []  # Clear after typing
                                        self.current_stable_char = None
                                        print(f"  → Typed '{most_common}' (confidence: {confidence:.2%}, stable: {count}/{STABLE_PREDICTIONS_NEEDED})")
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
        if self.use_mediapipe:
            try:
                self.hands.close()
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
