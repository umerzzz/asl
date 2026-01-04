"""
Fix the class mismatch between model and class_names.pkl

The model was trained with 36 classes (0-9, A-Z) but class_names.pkl has 40 classes
(0-9, A-Z, del, nothing, space, unknown).

This script provides two options:
1. Quick fix: Update class_names.pkl to match the model (36 classes) - loses 4 classes
2. Proper fix: Instructions to retrain the model with all 40 classes
"""

import pickle
import tensorflow as tf
from tensorflow import keras
import shutil
from datetime import datetime

MODEL_PATH = 'asl_model.h5'
CLASS_NAMES_PATH = 'class_names.pkl'
BACKUP_SUFFIX = '_backup'

def get_model_output_size():
    """Get the number of output classes from the model"""
    try:
        model = keras.models.load_model(MODEL_PATH)
        output_size = model.output_shape[-1]
        return output_size, model
    except:
        try:
            alt_path = MODEL_PATH.replace('.h5', '.keras')
            model = keras.models.load_model(alt_path)
            output_size = model.output_shape[-1]
            return output_size, model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

def get_saved_class_names():
    """Load saved class names"""
    try:
        with open(CLASS_NAMES_PATH, 'rb') as f:
            class_names = pickle.load(f)
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
        return None

def create_backup():
    """Create backup of class_names.pkl"""
    backup_path = CLASS_NAMES_PATH + BACKUP_SUFFIX + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        shutil.copy2(CLASS_NAMES_PATH, backup_path)
        print(f"[OK] Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"[ERROR] Could not create backup: {e}")
        return None

def fix_option1_quick():
    """Option 1: Update class_names.pkl to match model (36 classes)"""
    print("\n" + "="*70)
    print("OPTION 1: QUICK FIX - Update class_names.pkl to match model")
    print("="*70)
    print("\nThis will:")
    print("  - Remove 'del', 'nothing', 'space', 'unknown' from class_names.pkl")
    print("  - Keep only the 36 classes the model was trained on (0-9, A-Z)")
    print("  - Create a backup of the original file")
    print("\nWARNING: This means you won't be able to predict those 4 classes")
    print("         until you retrain the model with all 40 classes.")
    
    response = input("\nProceed with quick fix? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Cancelled.")
        return False
    
    # Get model output size
    model_output_size, model = get_model_output_size()
    if model_output_size is None:
        print("[ERROR] Could not load model")
        return False
    
    # Get saved class names
    saved_class_names = get_saved_class_names()
    if saved_class_names is None:
        print("[ERROR] Could not load class names")
        return False
    
    # Create backup
    backup_path = create_backup()
    if backup_path is None:
        print("[ERROR] Could not create backup. Aborting for safety.")
        return False
    
    # Create corrected class names (first 36 classes)
    corrected_class_names = saved_class_names[:model_output_size]
    
    print(f"\nOriginal class names ({len(saved_class_names)}): {saved_class_names}")
    print(f"Corrected class names ({len(corrected_class_names)}): {corrected_class_names}")
    print(f"Removed classes: {saved_class_names[model_output_size:]}")
    
    # Save corrected class names
    try:
        with open(CLASS_NAMES_PATH, 'wb') as f:
            pickle.dump(corrected_class_names, f)
        print(f"\n[OK] Updated {CLASS_NAMES_PATH} with {len(corrected_class_names)} classes")
        print(f"[OK] Backup saved at: {backup_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Could not save corrected class names: {e}")
        return False

def show_option2_instructions():
    """Option 2: Show instructions to retrain with all 40 classes"""
    print("\n" + "="*70)
    print("OPTION 2: PROPER FIX - Retrain model with all 40 classes")
    print("="*70)
    print("\nTo properly fix this, you need to retrain the model with all 40 classes.")
    print("\nSteps:")
    print("1. Make sure your dataset has all 40 classes:")
    print("   - 0-9 (digits)")
    print("   - A-Z (uppercase letters)")
    print("   - del, nothing, space, unknown (special classes)")
    print("\n2. Run the training script:")
    print("   python train_model.py")
    print("\n3. The training script will:")
    print("   - Load all classes from the dataset")
    print("   - Create a model with the correct number of output classes")
    print("   - Save class_names.pkl with all classes")
    print("\n4. After training, verify the model:")
    print("   python verify_model.py")
    print("\nThis is the recommended approach as it will give you a model")
    print("that can predict all 40 classes, including the special ones.")

def main():
    print("="*70)
    print("FIX MODEL CLASS MISMATCH")
    print("="*70)
    print("\nProblem identified:")
    print("  - Model was trained with 36 output classes (0-9, A-Z)")
    print("  - class_names.pkl has 40 classes (includes del, nothing, space, unknown)")
    print("  - This mismatch causes prediction errors")
    
    # Get current state
    model_output_size, model = get_model_output_size()
    saved_class_names = get_saved_class_names()
    
    if model_output_size is None or saved_class_names is None:
        print("\n[ERROR] Could not load model or class names")
        return
    
    print(f"\nCurrent state:")
    print(f"  Model output size: {model_output_size} classes")
    print(f"  Saved class names: {len(saved_class_names)} classes")
    print(f"  Mismatch: {len(saved_class_names) - model_output_size} extra classes in class_names.pkl")
    
    if model_output_size == len(saved_class_names):
        print("\n[OK] No mismatch detected. Model and class names are aligned.")
        return
    
    # Show options
    print("\n" + "="*70)
    print("FIX OPTIONS")
    print("="*70)
    print("\n1. Quick fix: Update class_names.pkl to match model (36 classes)")
    print("   - Fast solution, but loses 4 classes")
    print("   - Use this if you need immediate functionality")
    print("\n2. Proper fix: Retrain model with all 40 classes")
    print("   - Takes time but gives full functionality")
    print("   - Recommended for production use")
    
    choice = input("\nChoose option (1 or 2): ").strip()
    
    if choice == '1':
        fix_option1_quick()
    elif choice == '2':
        show_option2_instructions()
    else:
        print("Invalid choice. Showing instructions for both options:")
        show_option2_instructions()
        print("\n" + "-"*70)
        print("For quick fix, run this script again and choose option 1")

if __name__ == '__main__':
    main()

