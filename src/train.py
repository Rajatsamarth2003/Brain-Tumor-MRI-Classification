
import os
import json
import tensorflow as tf

from preprocessing import create_data_generators
from custom_cnn import build_custom_cnn


# ==========================================
# 1. Dataset paths
# ==========================================

TRAIN_DIR = "Datasets/train/train"
VALID_DIR = "Datasets/valid/valid"
TEST_DIR = "Datasets/test/test"


# ==========================================
# 2. Load dataset
# ==========================================

train_data, valid_data, test_data = create_data_generators(
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR
)


# ==========================================
# 3. Build Custom CNN
# ==========================================

model = build_custom_cnn()


# ==========================================
# 4. Compile model
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 5. Create output folders
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)


# ==========================================
# 6. Configure callbacks
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/custom_cnn.keras",
    monitor="val_loss",
    save_best_only=True
)


# ==========================================
# 7. Train model
# ==========================================

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=20,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


# ==========================================
# 8. Save training history
# ==========================================

history_path = "results/metrics/custom_cnn_history.json"

with open(history_path, "w") as file:
    json.dump(history.history, file, indent=4)


# ==========================================
# 9. Print completion message
# ==========================================

print("\nTraining completed!")
print("Best model saved to: models/custom_cnn.keras")
print(f"Training history saved to: {history_path}")
print(f"Epochs completed: {len(history.history['loss'])}")
