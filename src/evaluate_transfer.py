import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# Dataset path
# ==========================================

TEST_DIR = "Datasets/test/test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ==========================================
# Load test data
# ==========================================

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# ==========================================
# Load EfficientNetB0 model
# ==========================================

model = tf.keras.models.load_model(
    "models/efficientnetb0_fixed.keras"
)


# ==========================================
# Evaluate model
# ==========================================

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)


# ==========================================
# Predictions
# ==========================================

test_data.reset()

predictions = model.predict(test_data)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_data.classes

class_names = list(
    test_data.class_indices.keys()
)


# ==========================================
# Metrics
# ==========================================

accuracy = accuracy_score(
    true_classes,
    predicted_classes
)

precision = precision_score(
    true_classes,
    predicted_classes,
    average="weighted"
)

recall = recall_score(
    true_classes,
    predicted_classes,
    average="weighted"
)

f1 = f1_score(
    true_classes,
    predicted_classes,
    average="weighted"
)


# ==========================================
# Print results
# ==========================================

print("\n")
print("=" * 55)
print("EFFICIENTNETB0 - DETAILED EVALUATION")
print("=" * 55)

print(f"Test Loss       : {test_loss:.4f}")
print(f"Test Accuracy   : {accuracy:.4f}")
print(f"Accuracy        : {accuracy * 100:.2f}%")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1-Score        : {f1:.4f}")


# ==========================================
# Classification Report
# ==========================================

print("\n")
print("CLASSIFICATION REPORT")
print("=" * 55)

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\n")
print("CONFUSION MATRIX")
print("=" * 55)

print(cm)


# ==========================================
# Save confusion matrix
# ==========================================

os.makedirs(
    "results/confusion_matrices",
    exist_ok=True
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("EfficientNetB0 - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrices/efficientnetb0_confusion_matrix.png"
)

plt.show()


print("\nConfusion matrix saved to:")
print(
    "results/confusion_matrices/"
    "efficientnetb0_confusion_matrix.png"
)