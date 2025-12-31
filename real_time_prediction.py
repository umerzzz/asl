import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import pyautogui
import time

# Configuration
IMG_SIZE = 64
MODEL_PATH = 'asl_model.keras'  # Try new format first, fallback to .h5
CLASS_NAMES_PATH = 'class_names.pkl'
CONFIDENCE_THRESHOLD = 0.85  # Increased threshold for better accuracy
PREDICTION_DELAY = 0.5  # Increased delay to reduce false predictions
MANUAL_MODE = True  # Set to False for auto-typing

class ASLRealTimePredictor:
    def __init__(self):
        """Initialize the real-time ASL predictor"""
        print("Loading model...")
        try:
            self.model = keras.models.load_model(MODEL_PATH)
        except Exception as e:
            # Try fallback to .h5 format
            try:
                h5_path = MODEL_PATH.replace('.keras', '.h5')
                print(f"Trying fallback format: {h5_path}")
                self.model = keras.models.load_model(h5_path)
            except:
                raise FileNotFoundError(f"Could not load model from {MODEL_PATH} or {h5_path}. Please train the model first using train_model.py")
        
        print("Loading class names...")
        try:
            with open(CLASS_NAMES_PATH, 'rb') as f:
                self.class_names = pickle.load(f)
        except Exception as e:
            raise FileNotFoundError(f"Could not load class names from {CLASS_NAMES_PATH}")
        
        print(f"Model loaded successfully! Classes: {self.class_names}")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera. Please check your camera connection.")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.last_prediction_time = 0
        self.current_text = ""
        self.prediction_buffer = []
        self.buffer_size = 7  # Increased buffer size
        self.last_stable_prediction = None
        self.manual_mode = MANUAL_MODE
        
        # Initialize hand detection
        self.setup_hand_detection()
    
    def setup_hand_detection(self):
        """Setup hand detection using improved skin color detection"""
        # Improved skin color range in HSV (works better for different skin tones)
        self.lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        self.lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)  # For red tones
        self.upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    
    def is_likely_hand(self, contour, frame_shape):
        """Analyze contour to determine if it's likely a hand"""
        area = cv2.contourArea(contour)
        if area < 3000 or area > 50000:  # Too small or too large (likely not a hand)
            return False, 0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Aspect ratio check - hands are roughly square-ish (0.6 to 1.5)
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # Too wide or too tall
            return False, 0
        
        # Position check - hands are usually in center/lower center, not top (where faces are)
        frame_h, frame_w = frame_shape[:2]
        center_y_ratio = (y + h // 2) / frame_h
        if center_y_ratio < 0.15:  # Too high (likely face)
            return False, 0
        
        # Size check - hands should be reasonable size relative to frame
        frame_area = frame_w * frame_h
        area_ratio = area / frame_area
        if area_ratio < 0.01 or area_ratio > 0.3:  # Too small or too large
            return False, 0
        
        # Convexity defects - hands have fingers which create convexity defects
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    defect_count = len(defects)
                    # Hands typically have 3-8 significant convexity defects (fingers)
                    if defect_count >= 3:
                        # Calculate hand score based on multiple factors
                        score = 1.0
                        # Penalize if too high (likely face)
                        if center_y_ratio < 0.3:
                            score *= 0.5
                        # Reward if in good position (center area)
                        if 0.3 < center_y_ratio < 0.7:
                            score *= 1.2
                        # Reward good aspect ratio
                        if 0.7 < aspect_ratio < 1.3:
                            score *= 1.1
                        return True, score
        except:
            pass
        
        # If no convexity defects but other checks pass, might still be a hand
        if 0.3 < center_y_ratio < 0.7 and 0.7 < aspect_ratio < 1.3:
            return True, 0.8
        
        return False, 0
    
    def detect_hand_region(self, frame):
        """Detect hand region using improved filtering to distinguish hands from other objects"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin color (two ranges)
        mask1 = cv2.inRange(hsv, self.lower_skin1, self.upper_skin1)
        mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours to find the most likely hand
            hand_candidates = []
            for contour in contours:
                is_hand, score = self.is_likely_hand(contour, frame.shape)
                if is_hand:
                    area = cv2.contourArea(contour)
                    hand_candidates.append((contour, area, score))
            
            if hand_candidates:
                # Sort by score (if available) or area, prioritizing center position
                hand_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
                best_contour, area, score = hand_candidates[0]
                
                x, y, w, h = cv2.boundingRect(best_contour)
                
                # Add padding
                padding = 30
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                
                # Make it square
                size = max(w, h)
                center_x = x + w // 2
                center_y = y + h // 2
                x = max(0, center_x - size // 2)
                y = max(0, center_y - size // 2)
                w = h = min(size, min(frame.shape[1] - x, frame.shape[0] - y))
                
                if w > 80 and h > 80:
                    return x, y, w, h, True  # Return True to indicate hand detected
        
        # Fallback to center crop if no hand detected
        h, w = frame.shape[:2]
        roi_size = min(w, h) * 0.65
        x = int((w - roi_size) / 2)
        y = int((h - roi_size) / 2)
        return x, y, int(roi_size), int(roi_size), False
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction - match training data format"""
        # Resize to model input size
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        
        # Convert to RGB (model was trained on RGB)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = resized_rgb.astype('float32') / 255.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def predict(self, frame):
        """Predict ASL sign from frame"""
        preprocessed = self.preprocess_frame(frame)
        predictions = self.model.predict(preprocessed, verbose=0)
        confidence = np.max(predictions[0])
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        
        # Get top 3 predictions
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3_predictions = [(self.class_names[i], predictions[0][i]) for i in top3_indices]
        
        return predicted_class, confidence, top3_predictions
    
    def draw_roi(self, frame, x, y, w, h, hand_detected):
        """Draw region of interest rectangle with status"""
        color = (0, 255, 0) if hand_detected else (0, 165, 255)  # Green if hand detected, orange if not
        thickness = 3 if hand_detected else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Draw corner markers
        corner_size = 15
        cv2.line(frame, (x, y), (x + corner_size, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_size), color, thickness)
        cv2.line(frame, (x + w, y), (x + w - corner_size, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_size), color, thickness)
        cv2.line(frame, (x, y + h), (x + corner_size, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_size), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w - corner_size, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_size), color, thickness)
        
        # Add status text
        status = "Hand Detected" if hand_detected else "Using Center Region"
        cv2.putText(frame, status, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def type_character(self, character):
        """Type the predicted character"""
        try:
            pyautogui.write(character)
            print(f"✓ Typed: {character}")
        except Exception as e:
            print(f"✗ Could not type character: {e}")
    
    def run(self):
        """Run real-time prediction loop"""
        mode_text = "MANUAL MODE (Press 't' to type)" if self.manual_mode else "AUTO MODE"
        print("\n" + "="*60)
        print("ASL Real-Time Recognition Started")
        print("="*60)
        print("Instructions:")
        print("  - Position your hand in the center of the camera")
        print("  - Make sure your hand is well-lit and clearly visible")
        print("  - Hold your sign steady for best results")
        print(f"  - Mode: {mode_text}")
        print("  - Press 'q' to quit")
        print("  - Press 'c' to clear typed text")
        print("  - Press 's' to show current text")
        print("  - Press 'space' to add a space")
        if self.manual_mode:
            print("  - Press 't' to type the current prediction")
        print("="*60 + "\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand region
            x, y, w, h, hand_detected = self.detect_hand_region(frame)
            
            # Extract ROI
            roi = frame[y:y+h, x:x+w].copy()
            
            if roi.size > 0 and w > 50 and h > 50:
                # Draw ROI rectangle
                frame = self.draw_roi(frame, x, y, w, h, hand_detected)
                
                # Make prediction if enough time has passed
                current_time = time.time()
                if current_time - self.last_prediction_time >= PREDICTION_DELAY:
                    predicted_class, confidence, top3 = self.predict(roi)
                    
                    # Add to buffer
                    self.prediction_buffer.append((predicted_class, confidence))
                    if len(self.prediction_buffer) > self.buffer_size:
                        self.prediction_buffer.pop(0)
                    
                    # Use majority vote from buffer
                    if len(self.prediction_buffer) == self.buffer_size:
                        classes_in_buffer = [p[0] for p in self.prediction_buffer]
                        most_common = max(set(classes_in_buffer), key=classes_in_buffer.count)
                        count = classes_in_buffer.count(most_common)
                        avg_confidence = np.mean([p[1] for p in self.prediction_buffer if p[0] == most_common])
                        
                        # Only use if majority agrees and confidence is high
                        if count >= (self.buffer_size // 2 + 1) and avg_confidence >= CONFIDENCE_THRESHOLD:
                            self.last_stable_prediction = (most_common, avg_confidence)
                            self.prediction_buffer = []  # Clear buffer after stable prediction
                    
                    self.last_prediction_time = current_time
                    
                    # Display prediction on frame
                    status_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                    text = f"Predicted: {predicted_class} ({confidence:.2f})"
                    cv2.putText(frame, text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
                    # Show top 3 predictions
                    top3_text = "Top 3: " + ", ".join([f"{p[0]}({p[1]:.2f})" for p in top3])
                    cv2.putText(frame, top3_text, (10, 55), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    # Show stable prediction if available
                    if self.last_stable_prediction:
                        stable_char, stable_conf = self.last_stable_prediction
                        stable_text = f"Stable: {stable_char} ({stable_conf:.2f}) - "
                        if self.manual_mode:
                            stable_text += "Press 't' to type"
                        else:
                            stable_text += "Will auto-type"
                        cv2.putText(frame, stable_text, (10, 80), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Create preview window showing what model sees
                preview_roi = cv2.resize(roi, (128, 128))
                preview_frame = np.zeros((160, 160, 3), dtype=np.uint8)
                preview_frame[16:144, 16:144] = preview_roi
                cv2.putText(preview_frame, "Model Input", (5, 12), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.imshow('Model Preview', preview_frame)
            
            # Display current typed text
            text_display = f"Text: {self.current_text}"
            cv2.putText(frame, text_display, (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display mode and instructions
            mode_display = f"Mode: {mode_text}"
            cv2.putText(frame, mode_display, (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            cv2.putText(frame, "Press 'q' to quit, 'c' to clear, 's' to show, 't' to type", 
                       (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow('ASL Real-Time Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.current_text = ""
                self.last_stable_prediction = None
                print("Text cleared")
            elif key == ord('s'):
                print(f"Current text: {self.current_text}")
            elif key == ord('t') and self.manual_mode:
                if self.last_stable_prediction:
                    char, conf = self.last_stable_prediction
                    self.current_text += char
                    self.type_character(char)
                    self.last_stable_prediction = None
            elif key == ord(' '):
                self.current_text += " "
                self.type_character(" ")
                print("Space added")
        
        # Cleanup
        self.cap.release()
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
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
