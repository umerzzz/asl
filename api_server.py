"""
Flask API server for ASL recognition model
This server loads the model once and handles prediction requests from the Next.js app
"""
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import traceback

# Configuration
IMG_SIZE = 64
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'asl_model.keras')
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), 'class_names.pkl')
CONFIDENCE_THRESHOLD = 0.70
NUMBER_CLASS_LABELS = [str(i) for i in range(10)]  # "0" - "9"
ALPHABET_CLASS_LABELS = [chr(ord('A') + i) for i in range(26)]  # "A" - "Z"

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Global model instance (loaded once at startup)
model = None
class_names = None
number_class_indices = []
alphabet_class_indices = []

def load_model():
    """Load the ASL recognition model and class names"""
    global model, class_names, number_class_indices, alphabet_class_indices
    
    print("Loading ASL recognition model...")
    try:
        # Try loading .keras format first
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print(f"✓ Model loaded from {MODEL_PATH}")
        else:
            # Try .h5 format as fallback
            h5_path = MODEL_PATH.replace('.keras', '.h5')
            if os.path.exists(h5_path):
                model = keras.models.load_model(h5_path)
                print(f"✓ Model loaded from {h5_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH} or {h5_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise
    
    print("Loading class names...")
    try:
        with open(CLASS_NAMES_PATH, 'rb') as f:
            class_names = pickle.load(f)
        print(f"✓ Class names loaded: {len(class_names)} classes")
        print(f"  Classes: {class_names[:10]}..." if len(class_names) > 10 else f"  Classes: {class_names}")

        # Precompute indices for number and alphabet classes for fast filtering
        number_class_indices = [
            idx for idx, name in enumerate(class_names) if str(name) in NUMBER_CLASS_LABELS
        ]
        alphabet_class_indices = [
            idx for idx, name in enumerate(class_names) if str(name) in ALPHABET_CLASS_LABELS
        ]
        print(f"  Number class indices: {number_class_indices}")
        print(f"  Alphabet class indices: {alphabet_class_indices}")
    except Exception as e:
        print(f"✗ Error loading class names: {e}")
        raise
    
    # Optimize model for inference
    try:
        tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
        print("✓ Model optimization enabled")
    except:
        print("⚠ XLA optimization not available, continuing without it")
    
    print("=" * 70)
    print("ASL Recognition API Server Ready!")
    print("=" * 70)

def preprocess_image(image_array):
    """
    Preprocess image for model prediction - EXACTLY as training
    Args:
        image_array: numpy array of image (BGR format from OpenCV)
    Returns:
        Preprocessed image ready for model input
    """
    # Handle empty or invalid image
    if image_array.size == 0 or image_array.shape[0] == 0 or image_array.shape[1] == 0:
        black_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        return np.expand_dims(black_img, axis=0)
    
    # Resize to model input size (64x64)
    resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1] range
    normalized = resized.astype('float32') / 255.0
    
    # Add batch dimension
    return np.expand_dims(normalized, axis=0)

def decode_base64_image(base64_string):
    """
    Decode base64 image string to numpy array
    Args:
        base64_string: Base64 encoded image (with or without data URL prefix)
    Returns:
        numpy array of image in BGR format (OpenCV format)
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV (model was trained with BGR)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    except Exception as e:
        print(f"Error decoding image: {e}")
        raise ValueError(f"Invalid image data: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes_count': len(class_names) if class_names else 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict ASL sign from image
    Expected JSON body:
    {
        "image": "base64_encoded_image_string" (with or without data URL prefix)
    }
    
    Returns:
    {
        "success": true,
        "predicted_class": "A",
        "confidence": 0.95,
        "top3": [["A", 0.95], ["B", 0.03], ["C", 0.02]]
    }
    """
    try:
        if model is None or class_names is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Get image and optional mode hint from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing image data'
            }), 400
        
        image_base64 = data['image']
        is_number_mode = bool(data.get('isNumber', False))
        
        # Decode image
        try:
            image_array = decode_base64_image(image_base64)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid image format: {str(e)}'
            }), 400
        
        # Preprocess image
        preprocessed = preprocess_image(image_array)
        
        # Make prediction
        predictions = model.predict(preprocessed, verbose=0)[0]

        # Optionally restrict predictions to numbers or alphabets based on mode
        filtered_probs = predictions.copy()
        if is_number_mode and number_class_indices:
            # Zero out all non-number classes
            mask = np.zeros_like(filtered_probs)
            mask[number_class_indices] = 1.0
            filtered_probs *= mask
        elif (not is_number_mode) and alphabet_class_indices:
            # Zero out all non-alphabet classes
            mask = np.zeros_like(filtered_probs)
            mask[alphabet_class_indices] = 1.0
            filtered_probs *= mask

        # Fallback: if filter produced all zeros, use original predictions
        if np.all(filtered_probs == 0):
            filtered_probs = predictions

        # Get top prediction
        predicted_class_idx = int(np.argmax(filtered_probs))
        confidence = float(filtered_probs[predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        # Get top 3 predictions
        top3_indices = np.argsort(filtered_probs)[-3:][::-1]
        top3 = [
            [class_names[i], float(filtered_probs[i])]
            for i in top3_indices
        ]
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top3': top3
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of all class names"""
    if class_names is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'success': True,
        'classes': class_names
    })

if __name__ == '__main__':
    # Load model at startup
    try:
        load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Server will not start without a valid model.")
        sys.exit(1)
    
    # Run server
    # Default to port 5000, but allow override via environment variable
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    
    print(f"\nStarting ASL Recognition API Server on {host}:{port}")
    print("Endpoints:")
    print("  GET  /health  - Health check")
    print("  POST /predict - Predict ASL sign from image")
    print("  GET  /classes - Get list of classes")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(host=host, port=port, debug=False, threaded=True)
