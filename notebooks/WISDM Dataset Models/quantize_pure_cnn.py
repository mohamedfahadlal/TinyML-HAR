import os
import numpy as np
import tensorflow as tf

print("Initializing Pure 1D-CNN Native Quantization...")
X_train = np.load("X_train_processed.npy")
model = tf.keras.models.load_model('best_pure_cnn_model.keras')

def representative_data_gen():
    for i in range(100):
        sample = X_train[i:i+1].astype(np.float32)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# STRICT FULL INT8: Bypasses SELECT_TF_OPS completely!
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

try:
    print("Converting directly to full native 8-bit integer graph...")
    tflite_model_quant = converter.convert()
    
    output_filename = "pure_cnn_quantized.tflite"
    with open(output_filename, "wb") as f:
        f.write(tflite_model_quant)
        
    print("\n" + "="*22 + " QUANTIZATION REPORT " + "="*22)
    original_size = os.path.getsize('best_pure_cnn_model.keras') / 1024
    quantized_size = os.path.getsize(output_filename) / 1024

    print(f"Original Keras Model Size   : {original_size:.2f} KB")
    print(f"Quantized TFLite Model Size : {quantized_size:.2f} KB  (Target: <300 KB)")
    print(f"Status                      : SUCCESS! Native Binary Created.")
    print("="*65)

except Exception as e:
    print(f"Conversion failed: {e}")