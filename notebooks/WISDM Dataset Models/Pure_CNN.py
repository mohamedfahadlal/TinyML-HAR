import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Load your clean 7-class preprocessed arrays from disk
print("Loading preprocessed 7-class datasets...")
X_train = np.load("X_train_processed.npy")
X_test = np.load("X_test_processed.npy")
y_train = np.load("y_train_processed.npy")
y_test = np.load("y_test_processed.npy")

def build_pure_tinyml_cnn(input_shape=(50, 11), num_classes=7):
    """
    A pure 1D-CNN optimized for ultra-low power TinyML edges.
    Completely eliminates recurrent layers to bypass the Flex Delegate.
    """
    model = models.Sequential(name="Pure_TinyML_1D_CNN")
    model.add(layers.Input(shape=input_shape))
    
    # Block 1: Lower level spatial tracking
    model.add(layers.Conv1D(filters=32, kernel_size=5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Block 2: Spatial reduction via strided convolutions (Bypasses pooling to save memory)
    model.add(layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    
    # Block 3: Deep feature refinement
    model.add(layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.3))
    
    # Global Flattening to replace temporal sequences
    model.add(layers.GlobalAveragePooling1D())
    
    # Compact Classification Output Head
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Instantiate and Compile
model = build_pure_tinyml_cnn(input_shape=(50, 11), num_classes=7)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ModelCheckpoint('best_pure_cnn_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

print("\nTraining Purely Convolutional TinyML Network...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)