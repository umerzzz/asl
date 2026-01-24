#!/bin/bash
# Script to start the Python ASL recognition API server

echo "Starting ASL Recognition API Server..."
echo "======================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "api_server.py" ]; then
    echo "Error: api_server.py not found. Please run this script from the 'asl' directory"
    exit 1
fi

# Check if model files exist
if [ ! -f "asl_model.keras" ] && [ ! -f "asl_model.h5" ]; then
    echo "Warning: Model file not found. Please train the model first:"
    echo "  python train_model.py"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if class_names.pkl exists
if [ ! -f "class_names.pkl" ]; then
    echo "Error: class_names.pkl not found. Please train the model first:"
    echo "  python train_model.py"
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
python3 -c "import flask" 2>/dev/null || {
    echo "Installing Flask and dependencies..."
    pip install flask flask-cors
}

# Start the server
echo ""
echo "Starting server on http://127.0.0.1:5000"
echo "Press Ctrl+C to stop"
echo ""

python3 api_server.py
