import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# 1) LOAD YOUR DATA
# ----------------------------------------------------

X = np.load("data/processed/features/X.npy")   # shape: (N, 64, T)
y = np.load("data/processed/features/y.npy")   # shape: (N,)

# Add channel dimension for CNN: (N, 64, T, 1)
X = X[..., np.newaxis]

print("Dataset shape:", X.shape)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------
# 2) BUILD LIGHTWEIGHT TFLITE-FRIENDLY MODEL
# ----------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X.shape[1:]),   # (64, T, 1)

    tf.keras.layers.Conv2D(16, (3,3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------------------------------
# 3) TRAIN
# ----------------------------------------------------

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32
)

# Save normal Keras model (optional)
os.makedirs("models", exist_ok=True)
model.save("models/mantra_keras.keras")

# ----------------------------------------------------
# 4) CONVERT DIRECTLY TO TFLITE
# ----------------------------------------------------

converter = tf.lite.TFLiteConverter.from_keras_model(model)


# Make it smaller & faster for phone
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("models/mantra_cnn.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved TFLite model to: models/mantra_cnn.tflite")
