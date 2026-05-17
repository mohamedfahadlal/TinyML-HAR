import os
import tensorflow as tf

print("Initializing DeepConvLSTM TFLite Conversion...")

keras_model_path = 'best_deepconvlstm_model.keras'
output_filename = "deepconvlstm_quantized.tflite"

if not os.path.exists(keras_model_path):
    raise FileNotFoundError(f"Model file '{keras_model_path}' not found. Please train it first!")

# Load the trained DeepConvLSTM weights
model = tf.keras.models.load_model(keras_model_path)

# Configure the TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Enable Select TF Ops to support the dual LSTM layers
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

try:
    print("Compiling graph and freezing LSTM layers...")
    tflite_model_quant = converter.convert()
    
    with open(output_filename, "wb") as f:
        f.write(tflite_model_quant)
        
    print("\n" + "="*22 + " QUANTIZATION REPORT " + "="*22)
    original_size = os.path.getsize(keras_model_path) / 1024
    quantized_size = os.path.getsize(output_filename) / 1024

    print(f"Original Keras Model Size   : {original_size:.2f} KB")
    print(f"Quantized TFLite Model Size : {quantized_size:.2f} KB")
    print(f"Status                      : Success!")
    print("="*65)

except Exception as e:
    print(f"Conversion failed: {e}")