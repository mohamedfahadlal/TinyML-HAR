import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 1. Load preprocessed arrays instantly from disk
print("Loading preprocessed datasets...")
X_train = np.load("X_train_processed.npy")
X_test = np.load("X_test_processed.npy")
y_train = np.load("y_train_processed.npy")
y_test = np.load("y_test_processed.npy")
print(f"Loaded successfully! Training shapes: {X_train.shape}")

# 2. Define the Lightweight TinyML Network
def build_tinyml_har_model(input_shape=(50, 11), num_classes=20):
    model = models.Sequential(name="Separable_CNN_GRU_TinyML")
    
    # Feature Extraction: Depthwise Separable 1D-CNN
    model.add(layers.Input(shape=input_shape))
    model.add(layers.SeparableConv1D(filters=32, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    
    model.add(layers.SeparableConv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Sequence Context Tracking: GRU
    model.add(layers.GRU(48, return_sequences=False, dropout=0.2, recurrent_dropout=0.0))
    
    # Classification Head
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# 3. Instantiate and Compile
model = build_tinyml_har_model(input_shape=(50, 11), num_classes=20)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 4. Configure Training Adjustments
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ModelCheckpoint('best_har_windows_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# 5. Run Training
print("Starting training loop across preprocessed windows...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)