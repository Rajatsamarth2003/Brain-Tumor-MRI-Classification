
import os
import json
import tensorflow as tf

from config import IMAGE_SIZE, NUM_CLASSES


# ==========================================
# 1. Dataset paths
# ==========================================

TRAIN_DIR = "Datasets/train/train"
VALID_DIR = "Datasets/valid/valid"
TEST_DIR = "Datasets/test/test"


# ==========================================
# 2. Data generators for EfficientNet
# ==========================================

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()


train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_data = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)


# ==========================================
# 3. Load pretrained EfficientNetB0
# ==========================================

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

# Freeze pretrained layers initially
base_model.trainable = False


# ==========================================
# 4. Build model
# ==========================================

inputs = tf.keras.Input(
    shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

x = base_model(inputs, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.5)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)


# ==========================================
# 5. Compile model
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 6. Create output folders
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)


# ==========================================
# 7. Callbacks
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/efficientnetb0_fixed.keras",
    monitor="val_loss",
    save_best_only=True
)


# ==========================================
# 8. Stage 1: Train classification head
# ==========================================

print("\n==========================================")
print("STAGE 1: TRAINING CLASSIFICATION HEAD")
print("==========================================\n")

history_head = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=8,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


# ==========================================
# 9. Stage 2: Fine-tuning
# ==========================================

print("\n==========================================")
print("STAGE 2: FINE-TUNING EFFICIENTNETB0")
print("==========================================\n")

# Unfreeze EfficientNet layers
base_model.trainable = True

# Freeze all except the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


# ==========================================
# 10. Save combined training history
# ==========================================

combined_history = {}

for key in history_head.history:
    combined_history[key] = (
        history_head.history[key]
        + history_fine.history[key]
    )

history_path = "results/metrics/efficientnet_history.json"

with open(history_path, "w") as file:
    json.dump(combined_history, file, indent=4)


# ==========================================
# 11. Print completion message
# ==========================================

print("\n==========================================")
print("TRAINING COMPLETED")
print("==========================================")

print("Best model saved to:")
print("models/efficientnetb0_fixed.keras")

print("\nTraining history saved to:")
print(history_path)

print(
    "Total epochs recorded:",
    len(combined_history["loss"])
)
