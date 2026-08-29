# =================================================================
# PRODUCTION TINYML COMPILATION & QUANTIZATION PIPELINE
# PROJECT: Elder-Care Real-Time HAR System
# FILE: d:\Github\TinyML-HAR\notebooks\WISDM Dataset Models\quantize.py
# =================================================================

import os
import numpy as np
import tensorflow as tf

def run_production_quantization():
    print("="*65)
    print("      INITIALIZING ADVANCED TINYML QUANTIZATION PIPELINE")
    print("="*65)

    # -----------------------------------------------------------------
    # STEP 1: COMPONENT INGESTION
    # -----------------------------------------------------------------
    print("\n[Step 1/3] Ingesting trained baseline Keras model...")
    
    keras_model_path = 'best_1d_depth_wise_cnn_model.keras'
    
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Native baseline '{keras_model_path}' not found.")

    model = tf.keras.models.load_model(keras_model_path)
    print(f"-> Success: Loaded '{keras_model_path}' featuring {model.count_params():,} total parameters.")

    # -----------------------------------------------------------------
    # STEP 2: CONVERTER CONFIGURATION (DYNAMIC RANGE OPTIMIZATION)
    # -----------------------------------------------------------------
    print("\n[Step 2/3] Configuring TensorFlow Lite Converter for Recurrent-Safe Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable standard optimization tracks (quantizes weights to 8-bit integers)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Inject Select TF Ops support alongside native TFLite kernels to handle the GRU timeline loops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Core optimized mobile operators
        tf.lite.OpsSet.SELECT_TF_OPS     # Fallback operations for dynamic GRU variables
    ]

    # Bypasses the Windows variable mapping bug cleanly
    converter._experimental_lower_tensor_list_ops = False

    # -----------------------------------------------------------------
    # STEP 3: EXECUTING COMPILATION & DISK EXPORT
    # -----------------------------------------------------------------
    print("\n[Step 3/3] Executing structural transformations and weight mapping...")
    try:
        tflite_model_quant = converter.convert()
        
        output_filename = "1D_depth_wise_cnn_model_quantized.tflite"
        with open(output_filename, "wb") as f:
            f.write(tflite_model_quant)
            
        print("\n" + "="*22 + " QUANTIZATION REPORT " + "="*22)
        original_size = os.path.getsize(keras_model_path) / 1024
        quantized_size = os.path.getsize(output_filename) / 1024

        print(f"Original Keras Model Size   : {original_size:.2f} KB")
        print(f"Quantized TFLite Model Size : {quantized_size:.2f} KB  (Target: <300 KB)")
        print(f"Total Disk Volume Reduction : {original_size / quantized_size:.1f}x Smaller Footprint")
        print(f"Deployment Status           : SUCCESS. Ready for Android asset placement.")
        print("="*65)

    except Exception as e:
        print("\n" + "!"*20 + " COMPILATION FAILED " + "!"*20)
        print("Error details encountered by the TFLite Transformation engine:")
        print(str(e))
        print("!"*60)

if __name__ == "__main__":
    run_production_quantization()