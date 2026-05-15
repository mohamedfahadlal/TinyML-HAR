'''
# =========================
# Core Libraries
# =========================
import numpy as np
import os
import pandas as pd
import time

# =========================
# Data Splitting
# =========================
from sklearn.model_selection import train_test_split

# =========================
# TensorFlow / Keras
# =========================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    LSTM,
    GRU,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# =========================
# Evaluation Metrics
# =========================
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# =========================
# ROC & Multiclass Support
# =========================
from sklearn.preprocessing import label_binarize

# =========================
# Visualization
# =========================
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Model Efficiency
# =========================
import tracemalloc  # RAM usage measurement
import tensorflow.lite as tflite  # TFLite conversion


# =========================
# DATA LOADING
# =========================

base_path = r"D:\Github\TinyML-HAR\data\UCI HAR Dataset"
train_path = os.path.join(base_path, "train", "Inertial Signals")
test_path  = os.path.join(base_path, "test", "Inertial Signals")

# Train data
acc_x = np.loadtxt(os.path.join(train_path, "total_acc_x_train.txt"))
acc_y = np.loadtxt(os.path.join(train_path, "total_acc_y_train.txt"))
acc_z = np.loadtxt(os.path.join(train_path, "total_acc_z_train.txt"))

gyro_x = np.loadtxt(os.path.join(train_path, "body_gyro_x_train.txt"))
gyro_y = np.loadtxt(os.path.join(train_path, "body_gyro_y_train.txt"))
gyro_z = np.loadtxt(os.path.join(train_path, "body_gyro_z_train.txt"))

X_train = np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=-1)
y_train = np.loadtxt(os.path.join(base_path, "train", "y_train.txt")).astype(int)

# Test data
acc_x_test = np.loadtxt(os.path.join(test_path, "total_acc_x_test.txt"))
acc_y_test = np.loadtxt(os.path.join(test_path, "total_acc_y_test.txt"))
acc_z_test = np.loadtxt(os.path.join(test_path, "total_acc_z_test.txt"))

gyro_x_test = np.loadtxt(os.path.join(test_path, "body_gyro_x_test.txt"))
gyro_y_test = np.loadtxt(os.path.join(test_path, "body_gyro_y_test.txt"))
gyro_z_test = np.loadtxt(os.path.join(test_path, "body_gyro_z_test.txt"))

X_test = np.stack([acc_x_test, acc_y_test, acc_z_test,
                   gyro_x_test, gyro_y_test, gyro_z_test], axis=-1)

y_test = np.loadtxt(os.path.join(base_path, "test", "y_test.txt")).astype(int)

# Combine original train + test
X = np.concatenate([X_train, X_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

# =========================
# LABEL MAPPING
# =========================

def map_label(l):
    if l in [1,2,3]:
        return 0  # walk
    if l == 4:
        return 1  # sit
    if l == 5:
        return 2  # stand
    if l == 6:
        return 3  # lying

y_mapped = np.array([map_label(i) for i in y])

# =========================
# TRANSITION CLASS
# =========================

y_final = y_mapped.copy()
transition_indices = []

for i in range(1, len(y_mapped)-1):
    if y_mapped[i] != y_mapped[i-1] or y_mapped[i] != y_mapped[i+1]:
        transition_indices.append(i)
        y_final[i] = 4  # transition class

# =========================
# SYNTHETIC NOT-ON-BODY DATA
# =========================

def generate_not_on_body(num_windows=1000):
    windows = []
    for _ in range(num_windows):
        g = np.random.uniform(-1, 1, size=3)
        g = g / np.linalg.norm(g) * 9.8

        acc = np.tile(g, (128, 1)) + np.random.normal(0, 0.03, (128, 3))
        gyro = np.random.normal(0, 0.01, (128, 3))

        window = np.concatenate([acc, gyro], axis=1)
        windows.append(window)

    return np.array(windows)

not_on_body = generate_not_on_body(1000)
labels_nob = np.full((not_on_body.shape[0],), 5)

X = np.concatenate([X, not_on_body], axis=0)
y_final = np.concatenate([y_final, labels_nob], axis=0)

# =========================
# ADD MAGNITUDE CHANNELS
# =========================

def add_magnitude_channels(X):
    ax, ay, az = X[:,:,0], X[:,:,1], X[:,:,2]
    gx, gy, gz = X[:,:,3], X[:,:,4], X[:,:,5]

    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    X_with_mag = np.concatenate(
        [X, acc_mag[..., np.newaxis], gyro_mag[..., np.newaxis]], axis=2
    )

    return X_with_mag

X_mag = add_magnitude_channels(X)

# =========================
# TRAIN-TEST SPLIT
# =========================

X_train_raw8, X_test_raw8, y_train, y_test = train_test_split(
    X_mag, y_final, test_size=0.2, random_state=42, shuffle=True
)

X_train_raw6 = X_train_raw8[:, :, :6]

# =========================
# ROTATION AUGMENTATION
# =========================

def random_rotation_matrix():
    theta_x = np.random.uniform(0, 2*np.pi)
    theta_y = np.random.uniform(0, 2*np.pi)
    theta_z = np.random.uniform(0, 2*np.pi)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x),  np.cos(theta_x)]
    ])

    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

def rotate_window6_and_add_mags(window6, R):
    acc = window6[:, :3]
    gyro = window6[:, 3:6]

    acc_rot = acc @ R.T
    gyro_rot = gyro @ R.T

    acc_mag = np.sqrt((acc_rot**2).sum(axis=1))[:, None]
    gyro_mag = np.sqrt((gyro_rot**2).sum(axis=1))[:, None]

    return np.concatenate([acc_rot, gyro_rot, acc_mag, gyro_mag], axis=1)

def augment_train_rotations(X6, y, num_aug=1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    X_aug, y_aug = [], []

    for i in range(len(X6)):
        window6 = X6[i]
        for _ in range(num_aug):
            R = random_rotation_matrix()
            rotated8 = rotate_window6_and_add_mags(window6, R)
            X_aug.append(rotated8)
            y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)

num_aug = 1
X_rot, y_rot = augment_train_rotations(X_train_raw6, y_train, num_aug=num_aug, seed=42)

X_train_combined = np.concatenate([X_train_raw8, X_rot], axis=0)
y_train_combined = np.concatenate([y_train, y_rot], axis=0)

# =========================
# NORMALIZATION
# =========================

mean = X_train_combined.mean(axis=(0,1))
std  = X_train_combined.std(axis=(0,1))

def normalize(X, mean, std):
    return (X - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)

X_train_norm = normalize(X_train_combined, mean, std)
X_test_norm  = normalize(X_test_raw8, mean, std)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
model = Sequential([
    Input(shape=(128, 8)),

    # --- Feature Extraction ---
    Conv1D(32, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),

    Conv1D(32, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),

    # --- Temporal Modeling ---
    LSTM(48, return_sequences=False),

    Dropout(0.3),

    # --- Classification ---
    Dense(24, activation='relu'),
    Dense(6, activation='softmax')
])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
history = model.fit(
    X_train_norm,
    y_train_combined,
    validation_data=(X_test_norm, y_test),
    epochs=30,                                                      # Reduced epochs for quick testing; increase for better performance
    batch_size=32,
    verbose=1
)
loss, acc = model.evaluate(X_test_norm, y_test)
print("Test Accuracy:", acc)
import time

start = time.time()
_ = model.predict(X_test_norm)
end = time.time()

print("Inference Time (full test set):", end - start)
from sklearn.metrics import classification_report
import numpy as np

# Predict
y_pred_probs = model.predict(X_test_norm)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Light DeepConvLSTM")
plt.show()
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

print("\nPer-Class Metrics:")
for i in range(len(precision)):
    print(f"Class {i}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall:    {recall[i]:.4f}")
    print(f"  F1 Score:  {f1[i]:.4f}")
    print(f"  Support:   {support[i]}")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Flatten dataset: (samples, time, channels) → (samples*time, channels)
X_flat = X_train_norm.reshape(-1, X_train_norm.shape[2])

# Create DataFrame
df = pd.DataFrame(X_flat, columns=[
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "acc_mag", "gyro_mag"
])

# Correlation matrix
corr = df.corr()

# Plot
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Feature Correlation Heatmap")
plt.show()
# Compute mean window per class
classes = np.unique(y_train_combined)

plt.figure(figsize=(12,6))

for c in classes:
    class_indices = np.where(y_train_combined == c)[0]
    class_mean = X_train_norm[class_indices].mean(axis=0)  # (128,8)

    plt.imshow(class_mean.T, aspect='auto')
    plt.title(f"Average Window Heatmap - Class {c}")
    plt.xlabel("Time")
    plt.ylabel("Channels")
    plt.colorbar()
    plt.show()


#LOSO CV
subject_train = np.loadtxt(os.path.join(base_path, "train", "subject_train.txt"))
subject_test  = np.loadtxt(os.path.join(base_path, "test", "subject_test.txt"))

subjects = np.concatenate([subject_train, subject_test])
X_all = X_mag
y_all = y_final
from sklearn.metrics import accuracy_score

unique_subjects = np.unique(subjects)

loso_accuracies = []

for s in unique_subjects:
    print(f"Testing on Subject {s}")

    test_idx  = np.where(subjects == s)[0]
    train_idx = np.where(subjects != s)[0]

    X_train_loso = X_all[train_idx]
    y_train_loso = y_all[train_idx]
    X_test_loso  = X_all[test_idx]
    y_test_loso  = y_all[test_idx]

    # Normalize using train only
    mean = X_train_loso.mean(axis=(0,1))
    std  = X_train_loso.std(axis=(0,1))

    X_train_loso = (X_train_loso - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)
    X_test_loso  = (X_test_loso - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)

    # Build new model each fold
    model = build_light_deepconv_lstm()

    model.fit(X_train_loso, y_train_loso,
              epochs=30,
              batch_size=32,
              verbose=0)

    y_pred = np.argmax(model.predict(X_test_loso), axis=1)

    acc = accuracy_score(y_test_loso, y_pred)
    loso_accuracies.append(acc)

    print("Accuracy:", acc)
print("Mean LOSO Accuracy:", np.mean(loso_accuracies))
print("Std LOSO Accuracy:", np.std(loso_accuracies))
'''
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================
# 1. DATA LOADING (Raw Separated)
# =========================
base_path = r"D:\Github\TinyML-HAR\data\UCI HAR Dataset"

def load_dataset_split(subset):
    path = os.path.join(base_path, subset, "Inertial Signals")
    files = [
        f"total_acc_x_{subset}.txt", f"total_acc_y_{subset}.txt", f"total_acc_z_{subset}.txt",
        f"body_gyro_x_{subset}.txt", f"body_gyro_y_{subset}.txt", f"body_gyro_z_{subset}.txt"
    ]
    signals = [np.loadtxt(os.path.join(path, f)) for f in files]
    X = np.stack(signals, axis=-1)
    y = np.loadtxt(os.path.join(base_path, subset, f"y_{subset}.txt")).astype(int)
    return X, y

X_train_raw, y_train_raw = load_dataset_split("train")
X_test_raw, y_test_raw   = load_dataset_split("test")

# =========================
# 2. THE FIREWALL: PREPROCESSING
# =========================
def process_labels(y_raw):
    # Map 1,2,3 -> 0 (Walk), 4 -> 1 (Sit), 5 -> 2 (Stand), 6 -> 3 (Lying)
    mapped = np.array([0 if l in [1,2,3] else (l-3) for l in y_raw])
    final = mapped.copy()
    # Transition detection (Class 4)
    for i in range(1, len(mapped)-1):
        if mapped[i] != mapped[i-1] or mapped[i] != mapped[i+1]:
            final[i] = 4
    return final

def add_mags(X):
    acc_mag = np.sqrt((X[:,:,:3]**2).sum(axis=2))[..., np.newaxis]
    gyro_mag = np.sqrt((X[:,:,3:6]**2).sum(axis=2))[..., np.newaxis]
    return np.concatenate([X, acc_mag, gyro_mag], axis=2)

def generate_nob(num):
    # Generates 1.0g gravity windows for Not-on-body (Class 5)
    g = np.random.uniform(-1, 1, (num, 3))
    g = (g / np.linalg.norm(g, axis=1)[:, None]) 
    acc = np.tile(g[:, None, :], (1, 128, 1)) + np.random.normal(0, 0.02, (num, 128, 3))
    gyro = np.random.normal(0, 0.005, (num, 128, 3))
    return np.concatenate([acc, gyro], axis=2)

# Step A: Labeling
y_train = process_labels(y_train_raw)
y_test  = process_labels(y_test_raw)

# Step B: Add Not-On-Body (Class 5)
X_train = np.concatenate([X_train_raw, generate_nob(1000)], axis=0)
y_train = np.concatenate([y_train, np.full(1000, 5)], axis=0)
X_test  = np.concatenate([X_test_raw, generate_nob(250)], axis=0)
y_test  = np.concatenate([y_test, np.full(250, 5)], axis=0)

# Step C: Magnitude Channels
X_train_8ch = add_mags(X_train)
X_test_8ch  = add_mags(X_test)

# Step D: Normalization (Strictly using Train Stats)
mu  = X_train_8ch.mean(axis=(0,1))
std = X_train_8ch.std(axis=(0,1))

X_train_final = (X_train_8ch - mu) / std
X_test_final  = (X_test_8ch - mu) / std

# =========================
# 3. CLASS WEIGHTS (Critical Improvement)
# =========================
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(weights))

# =========================
# 4. OPTIMIZED DeepConvLSTM MODEL
# =========================
model = Sequential([
    Input(shape=(128, 8)),
    
    # Conv Layers (Spatial Feature Extraction)
    Conv1D(32, 5, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.2),
    
    Conv1D(64, 5, activation='relu', padding='same'),
    BatchNormalization(),
    
    # LSTM Layer (Temporal Feature Extraction)
    LSTM(64, return_sequences=False),
    Dropout(0.4),
    
    # Dense Classifier
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')
])

model.compile(
    optimizer=Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 5. TRAINING WITH CALLBACKS
# =========================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = model.fit(
    X_train_final, y_train,
    validation_split=0.15, # Use a slice of train to monitor progress
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# =========================
# 6. EVALUATION
# =========================
print("\n--- FINAL TEST EVALUATION ---")
y_pred = np.argmax(model.predict(X_test_final), axis=1)
print(classification_report(y_test, y_pred, target_names=['Walk','Sit','Stand','Lying','Trans','NOB']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Walk','Sit','Stand','Lying','Trans','NOB'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Optimized DeepConvLSTM")
plt.show()

# Per-class metrics
print("\nPer-Class Metrics:")
for i in range(6):
    class_indices = np.where(y_test == i)[0]
    if len(class_indices) > 0:
        print(f"Class {i} ({['Walk','Sit','Stand','Lying','Trans','NOB'][i]}):")
        print(classification_report(y_test[class_indices], y_pred[class_indices], digits=4))

    # Within-class accuracy
    acc_within_class = accuracy_score(y_test[class_indices], y_pred[class_indices])
    print(f"  Within Class Accuracy: {acc_within_class:.4f}")
    #all classes confusion matrix
    cm_class = confusion_matrix(y_test[class_indices], y_pred[class_indices])
    plt.figure(figsize=(4,3))
    sns.heatmap(cm_class, annot=True
                , fmt='d', cmap='Blues')
    plt.xlabel("Predicted") 
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for Class {i} ({['Walk','Sit','Stand','Lying','Trans','NOB'][i]})")
    plt.show()  

    # all evaluation metrics for class i
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test[class_indices], y_pred[class_indices], average='binary'
    )   
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # per class accuracy for class i
    acc_within_class = accuracy_score(y_test[class_indices], y_pred[class_indices])
    print(f"  Within Class Accuracy: {acc_within_class:.4f}")

    # per class confusion matrix for class i
    cm_class = confusion_matrix(y_test[class_indices], y_pred[class_indices])
    plt.figure(figsize=(4,3))
    sns.heatmap(cm_class, annot=True
                , fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for Class {i} ({['Walk','Sit','Stand','Lying','Trans','NOB'][i]})")
    plt.show()


#LOSO CV
LOSO_accuracies = []
subjects = np.concatenate([np.loadtxt(os.path.join(base_path, "train", "subject_train.txt")),
                            np.loadtxt(os.path.join(base_path, "test", "subject_test.txt"))])
unique_subjects = np.unique(subjects)
for s in unique_subjects:
    print(f"LOSO CV - Testing on Subject {s}")
    test_idx = np.where(subjects == s)[0]
    train_idx = np.where(subjects != s)[0]

    X_train_loso = X_train_final[train_idx]
    y_train_loso = y_train[train_idx]
    X_test_loso  = X_train_final[test_idx]  # Using train split for LOSO
    y_test_loso  = y_train[test_idx]

    model_loso = Sequential([
        Input(shape=(128, 8)),
        Conv1D(32, 5, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(64, 5, activation='relu', padding='same'),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax')
    ])
    
    model_loso.compile(
        optimizer=Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model_loso.fit(
        X_train_loso, y_train_loso,
        epochs=30,
        batch_size=32,
        class_weight=class_weight_dict,
        verbose=0
    )
    
    y_pred_loso = np.argmax(model_loso.predict(X_test_loso), axis=1)
    acc = accuracy_score(y_test_loso, y_pred_loso)
    LOSO_accuracies.append(acc)
    print(f"Subject {s} Accuracy: {acc:.4f}")
print(f"\nMean LOSO Accuracy: {np.mean(LOSO_accuracies):.4f}")

# per class metrics for LOSO
print("\nLOSO CV - Per Class Metrics:")
for i in range(6):
    class_indices = np.where(y_test_loso == i)[0]
    if len(class_indices) > 0:
        print(f"Class {i}:")
        print(classification_report(y_test_loso[class_indices], y_pred_loso[class_indices], digits=4))

        #within class accuracy for LOSO
        acc_within_class = accuracy_score(y_test_loso[class_indices], y_pred_loso[class_indices])
        print(f"  Within Class Accuracy: {acc_within_class:.4f}")

        #with class confusion matrix for LOSO
        cm_class = confusion_matrix(y_test_loso[class_indices], y_pred_loso[class_indices
        ])
        plt.figure(figsize=(4,3))

        sns.heatmap(cm_class, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"LOSO CV - Confusion Matrix for Class {i}")
        plt.show()

        #per class precision, recall, f1 for LOSO
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_loso[class_indices], y_pred_loso[class_indices], average='binary'
        )
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")


