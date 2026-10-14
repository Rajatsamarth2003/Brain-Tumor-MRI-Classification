
import os
import json
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

histories = {
    "Custom CNN": "results/metrics/custom_cnn_history.json",
    "EfficientNetB0": "results/metrics/efficientnet_history.json"
}

for model_name, file_path in histories.items():

    if not os.path.exists(file_path):
        print(f"History not found: {file_path}")
        continue

    with open(file_path, "r") as file:
        history = json.load(file)

    # Accuracy graph
    plt.figure(figsize=(8, 5))
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")
    plt.savefig(f"results/plots/{filename}_accuracy.png")
    plt.close()

    # Loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"results/plots/{filename}_loss.png")
    plt.close()

    print(f"Graphs created for {model_name}")

print("\nCheck the results/plots folder.")
