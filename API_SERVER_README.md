# ASL Recognition API Server

This Python Flask server provides the ASL recognition functionality for the SignFlow application.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Flask and Flask-CORS (for the API server)
- TensorFlow (for the ML model)
- OpenCV (for image processing)
- MediaPipe (for hand detection)
- Other required packages

### 2. Ensure Model Files Exist

The server requires:
- `asl_model.keras` (or `asl_model.h5`) - The trained CNN model
- `class_names.pkl` - The class labels

If these don't exist, train the model first:
```bash
python train_model.py
```

### 3. Start the Server

**Option A: Use the provided scripts**

**Windows:**
```bash
start_api_server.bat
```

**Linux/Mac:**
```bash
./start_api_server.sh
```

**Option B: Run manually**
```bash
python api_server.py
```

The server will start on `http://127.0.0.1:5000` by default.

### 4. Verify It's Working

Open your browser and visit:
```
http://127.0.0.1:5000/health
```

You should see:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes_count": 36
}
```

## API Endpoints

### `GET /health`
Health check endpoint to verify the server is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes_count": 36
}
```

### `POST /predict`
Predict ASL sign from an image.

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "isNumber": true
}
```

- `image` (required): The image to classify. It can be:
  - Base64 encoded string
  - Data URL format (`data:image/jpeg;base64,...`)
- `isNumber` (optional, default: `false`):
  - When `true`, the server restricts prediction to **digit** classes (`"0"`–`"9"`).
  - When `false` or omitted, the server restricts prediction to **alphabet** classes (`"A"`–`"Z"`).
  - If filtering by mode would result in no valid classes, the server falls back to using the full, unfiltered prediction distribution.

**Response:**
```json
{
  "success": true,
  "predicted_class": "4",
  "confidence": 0.95,
  "top3": [
    ["4", 0.95],
    ["B", 0.03],
    ["3", 0.02]
  ]
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error message here"
}
```

### `GET /classes`
Get list of all class names the model can recognize.

**Response:**
```json
{
  "success": true,
  "classes": ["A", "B", "C", ..., "Z", "0", "1", ..., "9"]
}
```

## Configuration

### Environment Variables

- `PORT` - Server port (default: 5000)
- `HOST` - Server host (default: 127.0.0.1)

Example:
```bash
PORT=8080 HOST=0.0.0.0 python api_server.py
```

### Model Configuration

The server uses the same configuration as the training script:
- `IMG_SIZE = 64` - Input image size for the model
- `CONFIDENCE_THRESHOLD = 0.70` - Minimum confidence for predictions

These can be adjusted in `api_server.py` if needed.

## Integration with Next.js

The Next.js app connects to this server via the bridge API route:
- Next.js route: `/api/recognition/predict`
- Python server: `http://127.0.0.1:5000/predict`

The bridge route handles:
- Forwarding requests to the Python server
- Error handling and timeouts
- Connection status checking

## Troubleshooting

### Server Won't Start

1. **Model files missing:**
   ```
   Error: Model file not found
   ```
   Solution: Train the model first with `python train_model.py`

2. **Port already in use:**
   ```
   OSError: [Errno 48] Address already in use
   ```
   Solution: Change the port using `PORT=5001 python api_server.py` or stop the process using port 5000

3. **Dependencies missing:**
   ```
   ModuleNotFoundError: No module named 'flask'
   ```
   Solution: Install dependencies with `pip install -r requirements.txt`

### Predictions Not Working

1. **Low confidence predictions:**
   - Ensure good lighting
   - Position hand clearly in camera view
   - Hold sign steady for a moment

2. **Connection errors from Next.js:**
   - Verify server is running: `curl http://127.0.0.1:5000/health`
   - Check firewall settings
   - Verify `PYTHON_API_URL` environment variable in Next.js

### Performance Issues

- The model loads once at startup (takes a few seconds)
- Each prediction takes ~50-200ms depending on hardware
- For better performance, use GPU acceleration (see `asl/setup_gpu_training.md`)

## Development

### Running in Debug Mode

The server runs in production mode by default. For debugging:

```python
# In api_server.py, change:
app.run(host=host, port=port, debug=True, threaded=True)
```

### Testing the API

Using curl:
```bash
# Health check
curl http://127.0.0.1:5000/health

# Prediction (with base64 image)
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

Using Python:
```python
import requests
import base64

# Read image and encode
with open("test_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Send prediction request
response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"image": f"data:image/jpeg;base64,{image_data}"}
)

print(response.json())
```

## Production Deployment

For production, consider:

1. **Use a production WSGI server:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
   ```

2. **Add authentication** to protect the API

3. **Use environment variables** for configuration

4. **Set up monitoring** and logging

5. **Use a reverse proxy** (nginx) for better performance

## License

Same as the main SignFlow project.
