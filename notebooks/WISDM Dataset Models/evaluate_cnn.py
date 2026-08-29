import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# 1. Configuration & Label Mapping
CLASS_NAMES = [
    "Walking", "Jogging", "Stairs", "Sitting", 
    "Standing", "Postural Transitions", "Not-On-Body Anomaly"
]
ANOMALY_CLASS_IDX = 6

print("Step 1: Loading test arrays and trained Keras model...")
X_test = np.load("X_test_processed.npy")
y_test = np.load("y_test_processed.npy")

# Load the best performing model saved by your checkpoint callback
model = tf.keras.models.load_model('best_har_windows_model.keras')
print(f"Model loaded successfully. Input footprint expected: {model.input_shape}")

# --- METRIC CATEGORY 1: OVERALL CLASSIFICATION PERFORMANCE ---
print("\nStep 2: Evaluating overall baseline metrics...")
y_pred_probs = model.predict(X_test, batch_size=256, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Extract macro-averaged precision, recall, and f1-score
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
overall_accuracy = np.mean(y_pred == y_test)

print("\n" + "="*20 + " OVERALL HAR PERFORMANCE REPORT " + "="*20)
print(f"Overall Classification Accuracy : {overall_accuracy*100:.2f}%  (Target: >90%)")
print(f"Macro Precision                 : {prec_macro*100:.2f}%  (Target: >85%)")
print(f"Macro Recall (Sensitivity)      : {rec_macro*100:.2f}%  (Target: >85%)")
print(f"Macro F1-Score                  : {f1_macro*100:.2f}%  (Target: >85%)")
print("="*72)

# --- METRIC CATEGORY 2: SAFETY-CRITICAL ANOMALY DETECTION ---
print("\nStep 3: Isolating elderly safety anomaly metrics...")
cm = confusion_matrix(y_test, y_pred)

# Binary classification isolation for Class 6 (Anomaly) vs All Other Classes
TP = cm[ANOMALY_CLASS_IDX, ANOMALY_CLASS_IDX]
FN = np.sum(cm[ANOMALY_CLASS_IDX, :]) - TP
FP = np.sum(cm[:, ANOMALY_CLASS_IDX]) - TP
TN = np.sum(cm) - (TP + FN + FP)

# Calculate anomaly-specific performance indicators
anomaly_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
anomaly_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
anomaly_f1 = 2 * (anomaly_precision * anomaly_recall) / (anomaly_precision + anomaly_recall) if (anomaly_precision + anomaly_recall) > 0 else 0
anomaly_fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

print("\n" + "="*20 + " ANOMALY DETECTION CRITERIA REPORT " + "="*20)
print(f"Anomaly Precision        : {anomaly_precision*100:.2f}%  (Target: >80%)")
print(f"Anomaly Recall (Sens.)   : {anomaly_recall*100:.2f}%  (Target: >85%)")
print(f"Anomaly F1-Score         : {anomaly_f1*100:.2f}%  (Target: >80%)")
print(f"False Positive Rate (FPR): {anomaly_fpr*100:.2f}%  (Target: <10%)")
print("="*71)

# --- METRIC CATEGORY 3: ORIENTATION INVARIANCE STRESS-TEST ---
print("\nStep 4: Running random 3D spatial rotation stress-tests...")

def build_rotation_matrix():
    tx, ty, tz = np.random.uniform(0, 2*np.pi, size=3)
    Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

# Note: Testing rotation is ideally performed on un-normalized tracking variables.
# We simulate a directional permutation check directly across the raw axis channels (0:3 and 6:9)
num_stress_runs = 10
rotated_accuracies = []
rotated_f1_scores = []

for run in range(num_stress_runs):
    X_stressed = X_test.copy()
    R = build_rotation_matrix()
    
    # Rotate raw accelerometer sub-vectors (channels 0,1,2) and gyro sub-vectors (channels 6,7,8)
    for w in range(len(X_stressed)):
        X_stressed[w, :, 0:3] = X_stressed[w, :, 0:3] @ R.T
        X_stressed[w, :, 6:9] = X_stressed[w, :, 6:9] @ R.T
        
    y_stressed_pred = np.argmax(model.predict(X_stressed, batch_size=512, verbose=0), axis=1)
    acc_run = np.mean(y_stressed_pred == y_test)
    _, _, f1_run, _ = precision_recall_fscore_support(y_test, y_stressed_pred, average='macro')
    
    rotated_accuracies.append(acc_run)
    rotated_f1_scores.append(f1_run)

avg_rotated_accuracy = np.mean(rotated_accuracies)
accuracy_drop = (overall_accuracy - avg_rotated_accuracy) * 100
worst_case_f1 = np.min(rotated_f1_scores) * 100

print("\n" + "="*19 + " ORIENTATION INVARIANCE REPORT " + "="*19)
print(f"Average Accuracy Under Stress : {avg_rotated_accuracy*100:.2f}%")
print(f"Induced Accuracy Drop         : {accuracy_drop:.2f}%  (Target: <5% drop)")
print(f"Worst-Case Rotation F1-Score  : {worst_case_f1:.2f}%  (Target: >80%)")
print("="*71)

# --- METRIC CATEGORY 4: COMPILING MODEL SIZE FOOTPRINT ---
print("\nStep 5: Benchmarking binary parameter footprints...")
keras_file_size_kb = os.path.getsize('best_har_windows_model.keras') / 1024
# Standard integer quantization (int8) reduces overall size by roughly 4x
estimated_tflite_quant_kb = (model.count_params() * 1) / 1024 

print("\n" + "="*22 + " ON-DEVICE SIZE FOOTPRINT " + "="*22)
print(f"Uncompressed Keras File Size   : {keras_file_size_kb:.2f} KB")
print(f"Estimated 8-bit Quantized Size : {estimated_tflite_quant_kb:.2f} KB  (Target: <300 KB)")
print("="*71)

# --- STEP 6: VISUALIZATION ---
print("\nStep 6: Generating and saving Normalized Confusion Matrix...")
plt.figure(figsize=(10, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Normalized Confusion Matrix - TinyML HAR Suite")
plt.ylabel("Ground Truth Habits")
plt.xlabel("Model Predictions")
plt.tight_layout()
plt.savefig("har_confusion_matrix.png", dpi=300)
print("Plot successfully compiled and saved to disk as 'har_confusion_matrix.png'.")