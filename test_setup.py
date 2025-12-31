"""
Setup Verification Script for ASL Recognition
This script helps you verify your camera, lighting, and hand detection are working properly.
"""

import cv2
import numpy as np
import time

class SetupTester:
    def __init__(self):
        """Initialize the setup tester"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera. Please check your camera connection.")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Setup hand detection
        self.lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        self.lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        self.upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    
    def is_likely_hand(self, contour, frame_shape):
        """Analyze contour to determine if it's likely a hand"""
        area = cv2.contourArea(contour)
        if area < 3000 or area > 50000:
            return False, 0
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Aspect ratio check
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, 0
        
        # Position check - hands not in top area (where faces are)
        frame_h, frame_w = frame_shape[:2]
        center_y_ratio = (y + h // 2) / frame_h
        if center_y_ratio < 0.15:
            return False, 0
        
        # Size check
        frame_area = frame_w * frame_h
        area_ratio = area / frame_area
        if area_ratio < 0.01 or area_ratio > 0.3:
            return False, 0
        
        # Convexity defects check
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    defect_count = len(defects)
                    if defect_count >= 3:
                        score = 1.0
                        if center_y_ratio < 0.3:
                            score *= 0.5
                        if 0.3 < center_y_ratio < 0.7:
                            score *= 1.2
                        if 0.7 < aspect_ratio < 1.3:
                            score *= 1.1
                        return True, score
        except:
            pass
        
        if 0.3 < center_y_ratio < 0.7 and 0.7 < aspect_ratio < 1.3:
            return True, 0.8
        
        return False, 0
    
    def detect_hand(self, frame):
        """Detect hand in frame with improved filtering"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_skin1, self.upper_skin1)
        mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter to find most likely hand
            hand_candidates = []
            for contour in contours:
                is_hand, score = self.is_likely_hand(contour, frame.shape)
                if is_hand:
                    area = cv2.contourArea(contour)
                    hand_candidates.append((contour, area, score))
            
            if hand_candidates:
                hand_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
                best_contour, area, score = hand_candidates[0]
                x, y, w, h = cv2.boundingRect(best_contour)
                return x, y, w, h, area, True
        
        return 0, 0, 0, 0, 0, False
    
    def analyze_lighting(self, frame):
        """Analyze lighting conditions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Good lighting: mean between 80-180, std > 20 (some contrast)
        if 80 <= mean_brightness <= 180 and std_brightness > 20:
            return "GOOD", mean_brightness, std_brightness
        elif mean_brightness < 80:
            return "TOO DARK", mean_brightness, std_brightness
        elif mean_brightness > 180:
            return "TOO BRIGHT", mean_brightness, std_brightness
        else:
            return "POOR CONTRAST", mean_brightness, std_brightness
    
    def analyze_background(self, frame, hand_detected):
        """Analyze background quality"""
        if not hand_detected:
            return "CANNOT ANALYZE - No hand detected"
        
        # Sample background around hand
        h, w = frame.shape[:2]
        bg_samples = []
        
        # Sample corners
        corners = [
            frame[0:50, 0:50],  # Top-left
            frame[0:50, w-50:w],  # Top-right
            frame[h-50:h, 0:50],  # Bottom-left
            frame[h-50:h, w-50:w]  # Bottom-right
        ]
        
        for corner in corners:
            if corner.size > 0:
                bg_samples.append(np.mean(corner))
        
        if bg_samples:
            bg_variance = np.var(bg_samples)
            bg_mean = np.mean(bg_samples)
            
            # Good background: relatively uniform, not too bright/dark
            if bg_variance < 500 and 30 < bg_mean < 200:
                return "GOOD - Plain background detected"
            elif bg_variance > 1000:
                return "BUSY - Background has patterns/textures"
            else:
                return f"OK - Mean brightness: {bg_mean:.0f}"
        
        return "UNKNOWN"
    
    def run_test(self):
        """Run setup verification test"""
        print("="*70)
        print("ASL RECOGNITION - SETUP VERIFICATION")
        print("="*70)
        print("\nThis test will help you verify:")
        print("  1. Camera is working")
        print("  2. Lighting conditions are good")
        print("  3. Hand detection is working")
        print("  4. Background is suitable")
        print("\nPress 'q' to quit, 's' to save current frame")
        print("="*70 + "\n")
        
        frame_count = 0
        hand_detections = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror
            frame_count += 1
            
            # Detect hand and show all detected objects
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, self.lower_skin1, self.upper_skin1)
            mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
            mask = cv2.bitwise_or(mask1, mask2)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Show all detected skin-colored objects (for debugging)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Only show significant contours
                    x_obj, y_obj, w_obj, h_obj = cv2.boundingRect(contour)
                    is_hand, score = self.is_likely_hand(contour, frame.shape)
                    if is_hand:
                        cv2.rectangle(frame, (x_obj, y_obj), (x_obj + w_obj, y_obj + h_obj), (0, 255, 0), 2)
                        cv2.putText(frame, "HAND", (x_obj, y_obj - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    else:
                        cv2.rectangle(frame, (x_obj, y_obj), (x_obj + w_obj, y_obj + h_obj), (0, 0, 255), 1)
                        cv2.putText(frame, "FILTERED", (x_obj, y_obj - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            # Detect hand
            x, y, w, h, area, hand_detected = self.detect_hand(frame)
            
            if hand_detected:
                hand_detections += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, "HAND DETECTED", (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Draw center region
                h_frame, w_frame = frame.shape[:2]
                center_size = int(min(w_frame, h_frame) * 0.6)
                cx = (w_frame - center_size) // 2
                cy = (h_frame - center_size) // 2
                cv2.rectangle(frame, (cx, cy), (cx + center_size, cy + center_size),
                            (0, 165, 255), 2)
                cv2.putText(frame, "NO HAND DETECTED - Position hand in center",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Analyze lighting
            lighting_status, mean_bright, std_bright = self.analyze_lighting(frame)
            lighting_color = (0, 255, 0) if lighting_status == "GOOD" else (0, 165, 255)
            
            # Analyze background
            bg_status = self.analyze_background(frame, hand_detected)
            
            # Display information
            y_offset = 30
            # Count filtered vs accepted objects
            total_objects = len([c for c in contours if cv2.contourArea(c) > 1000])
            filtered_objects = total_objects - (1 if hand_detected else 0)
            
            info_lines = [
                f"Lighting: {lighting_status} (Brightness: {mean_bright:.0f}, Contrast: {std_bright:.0f})",
                f"Background: {bg_status}",
                f"Hand Detection: {'YES' if hand_detected else 'NO'}",
                f"Detection Rate: {(hand_detections/frame_count*100):.1f}%",
                f"Objects Detected: {total_objects} (Filtered: {filtered_objects})",
            ]
            
            if hand_detected:
                info_lines.append(f"Hand Area: {area:.0f} pixels")
                info_lines.append(f"Hand Size: {w}x{h} pixels")
            
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, y_offset + i * 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, lighting_color, 1)
            
            # Recommendations
            recommendations = []
            if lighting_status != "GOOD":
                if "DARK" in lighting_status:
                    recommendations.append("Add more light")
                elif "BRIGHT" in lighting_status:
                    recommendations.append("Reduce lighting or move away from light")
                else:
                    recommendations.append("Improve lighting contrast")
            
            if not hand_detected and frame_count > 30:
                recommendations.append("Move hand closer to camera")
                recommendations.append("Ensure hand is well-lit")
            
            if "BUSY" in bg_status:
                recommendations.append("Use a plain background (wall, cloth)")
            
            if recommendations:
                cv2.putText(frame, "Recommendations:", (10, frame.shape[0] - 20 - len(recommendations) * 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                for i, rec in enumerate(recommendations):
                    cv2.putText(frame, f"  - {rec}", (10, frame.shape[0] - 10 - (len(recommendations) - i - 1) * 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Status indicator
            all_good = (lighting_status == "GOOD" and hand_detected and 
                       "GOOD" in bg_status or "OK" in bg_status)
            
            status_text = "READY - All systems good!" if all_good else "ADJUST SETTINGS"
            status_color = (0, 255, 0) if all_good else (0, 0, 255)
            cv2.putText(frame, status_text, (frame.shape[1] - 250, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.imshow('Setup Verification - Press Q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"setup_test_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
        
        # Final report
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("FINAL REPORT")
        print("="*70)
        print(f"Total frames analyzed: {frame_count}")
        print(f"Hand detection rate: {(hand_detections/frame_count*100):.1f}%")
        
        if hand_detections / frame_count > 0.8:
            print("✓ Hand detection: EXCELLENT")
        elif hand_detections / frame_count > 0.5:
            print("⚠ Hand detection: GOOD (but could be better)")
        else:
            print("✗ Hand detection: POOR (needs improvement)")
        
        print("\nRecommendations:")
        print("  1. Ensure good, even lighting (not too bright or dark)")
        print("  2. Use a plain background (dark wall or cloth works best)")
        print("  3. Position hand in center of frame")
        print("  4. Keep hand steady and well-lit")
        print("  5. Make sure entire hand is visible")
        print("="*70)

def main():
    try:
        tester = SetupTester()
        tester.run_test()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

