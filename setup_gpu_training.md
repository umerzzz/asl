# Setting Up GPU Training for RTX 3070

## Current Status

Your TensorFlow installation doesn't detect the GPU. To use your RTX 3070 for training, you need to install GPU support.

## Option 1: Install TensorFlow with CUDA Support (Recommended)

### Step 1: Install CUDA Toolkit
1. Download CUDA Toolkit 11.8 or 12.x from: https://developer.nvidia.com/cuda-downloads
2. Install it (make sure to add to PATH)

### Step 2: Install cuDNN
1. Download cuDNN from: https://developer.nvidia.com/cudnn
2. Extract and copy files to CUDA installation directory

### Step 3: Install TensorFlow with GPU Support
```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

Or for TensorFlow 2.15+:
```bash
pip install tensorflow[and-cuda]
```

### Step 4: Verify GPU Detection
```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## Option 2: Use Pre-built TensorFlow-GPU (Alternative)

```bash
pip uninstall tensorflow
pip install tensorflow-gpu
```

**Note**: TensorFlow 2.10+ includes GPU support in the main package, so `tensorflow-gpu` is deprecated.

## Training Configuration

The training script has been optimized for your RTX 3070 (8GB VRAM):

- **Batch Size**: 64 (can be increased to 128 if you have memory)
- **Mixed Precision**: Enabled (float16) for 2x speedup
- **Memory Growth**: Enabled to prevent OOM errors
- **GPU Memory**: Dynamically allocated

## Performance Expectations

With RTX 3070, you should see:
- **Training Speed**: ~10-20x faster than CPU
- **Time per Epoch**: ~2-5 minutes (depending on batch size)
- **Total Training Time**: ~40-100 minutes for 20 epochs

## Troubleshooting

### GPU Not Detected
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA: `nvcc --version`
3. Check TensorFlow GPU: `python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"`

### Out of Memory (OOM) Errors
- Reduce batch size in `train_model.py` (change `BATCH_SIZE = 64` to `BATCH_SIZE = 32`)
- Disable mixed precision if needed

### CUDA Version Mismatch
- Make sure CUDA version matches TensorFlow requirements
- TensorFlow 2.20.0 requires CUDA 11.8 or 12.x

## Current Training Script Features

The updated `train_model.py` includes:
- ✅ Automatic GPU detection and configuration
- ✅ Memory growth to prevent OOM
- ✅ Mixed precision training (float16)
- ✅ Optimized batch size for 8GB VRAM
- ✅ Fallback to CPU if GPU not available

## Next Steps

1. Install GPU support (see Option 1 above)
2. Verify GPU detection
3. Run training: `python train_model.py`
4. Monitor GPU usage: `nvidia-smi` (in another terminal)

