import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 1. Load your clean 7-class preprocessed arrays from disk
print("Loading preprocessed 7-class datasets...")
X_train = np.load("X_train_processed.npy")
X_test = np.load("X_test_processed.npy")
y_train = np.load("y_train_processed.npy")
y_test = np.load("y_test_processed.npy")

# 2. Design the Baseline DeepConvLSTM Network
def build_deepconvlstm(input_shape=(50, 11), num_classes=7):
    """
    Implements the standard DeepConvLSTM architecture for HAR.
    Uses 4 dense 1D-CNN layers followed by a 2-layer stacked LSTM network.
    """
    model = models.Sequential(name="DeepConvLSTM_HAR")
    model.add(layers.Input(shape=input_shape))
    
    # 4 Dense 1D-CNN Layers (Feature extraction without pooling to maintain temporal resolution)
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'))
    
    # 2 Stacked LSTM Layers for Deep Temporal Context Tracking
    model.add(layers.LSTM(64, return_sequences=True, dropout=0.2))
    model.add(layers.LSTM(64, return_sequences=False, dropout=0.2))
    
    # Classification Head
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# 3. Instantiate and Compile
model = build_deepconvlstm(input_shape=(50, 11), num_classes=7)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print structural summary to inspect parameter footprint differences
model.summary()

# 4. Training Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ModelCheckpoint('best_deepconvlstm_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# 5. Run Training on CPU
print("\nLaunching training loop across preprocessed windows...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=256, 
    callbacks=callbacks,
    verbose=1
)