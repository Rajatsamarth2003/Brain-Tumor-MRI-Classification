
import os
import matplotlib.pyplot as plt


# ==========================================
# 1. Create output folder
# ==========================================

os.makedirs("results/plots", exist_ok=True)


# ==========================================
# 2. Custom CNN training results
# ==========================================

custom_cnn_accuracy = [
    0.4997, 0.4407, 0.4702, 0.4566, 0.4808,
    0.4838, 0.5180, 0.4525, 0.4643, 0.5327,
    0.5280, 0.5310, 0.5192, 0.5522, 0.5522,
    0.5552, 0.5711, 0.5740, 0.5805, 0.5729
]

custom_cnn_val_accuracy = [
    0.2570, 0.2351, 0.2371, 0.2669, 0.2749,
    0.4283, 0.5100, 0.5179, 0.5677, 0.5916,
    0.6315, 0.6295, 0.6275, 0.6594, 0.6932,
    0.6454, 0.6892, 0.6693, 0.7072, 0.6853
]

custom_cnn_loss = [
    7.1845, 1.5328, 1.3588, 1.1958, 1.1659,
    1.3803, 1.2720, 1.2173, 1.1282, 1.0736,
    1.0138, 1.0641, 1.0563, 1.0016, 1.0079,
    1.0037, 0.9698, 0.9683, 0.9468, 0.9422
]

custom_cnn_val_loss = [
    12.3655, 45.4878, 33.7937, 13.4900, 2.5354,
    5.9540, 2.9030, 1.5838, 1.0161, 1.0008,
    0.9275, 0.8291, 0.8469, 0.8137, 0.7802,
    0.8060, 0.7221, 0.7251, 0.7672, 0.7569
]


# ==========================================
# 3. EfficientNetB0 training results
# ==========================================

# Stage 1: 8 epochs
efficientnet_accuracy_stage1 = [
    0.7068, 0.8313, 0.8578, 0.8714,
    0.8743, 0.8909, 0.9032, 0.9192
]

efficientnet_val_accuracy_stage1 = [
    0.8187, 0.8466, 0.8645, 0.8665,
    0.8765, 0.8745, 0.8586, 0.8785
]

efficientnet_loss_stage1 = [
    0.7535, 0.4342, 0.3851, 0.3444,
    0.3358, 0.2992, 0.2544, 0.2433
]

efficientnet_val_loss_stage1 = [
    0.4749, 0.4228, 0.3717, 0.3778,
    0.3409, 0.3471, 0.3566, 0.3236
]


# Stage 2: 5 epochs
efficientnet_accuracy_stage2 = [
    0.7906, 0.8578, 0.8560, 0.8767, 0.8720
]

efficientnet_val_accuracy_stage2 = [
    0.8745, 0.8645, 0.8665, 0.8625, 0.8625
]

efficientnet_loss_stage2 = [
    0.5505, 0.4232, 0.4123, 0.3641, 0.3462
]

efficientnet_val_loss_stage2 = [
    0.3230, 0.3417, 0.3542, 0.3612, 0.3644
]


# Combine stages for plotting
efficientnet_accuracy = (
    efficientnet_accuracy_stage1
    + efficientnet_accuracy_stage2
)

efficientnet_val_accuracy = (
    efficientnet_val_accuracy_stage1
    + efficientnet_val_accuracy_stage2
)

efficientnet_loss = (
    efficientnet_loss_stage1
    + efficientnet_loss_stage2
)

efficientnet_val_loss = (
    efficientnet_val_loss_stage1
    + efficientnet_val_loss_stage2
)


# ==========================================
# 4. Function to create graphs
# ==========================================

def create_training_graphs(
    model_name,
    train_accuracy,
    val_accuracy,
    train_loss,
    val_loss
):
    epochs = range(1, len(train_accuracy) + 1)

    filename = model_name.lower().replace(" ", "_")

    # Accuracy graph
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        train_accuracy,
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        marker="o",
        label="Validation Accuracy"
    )

    plt.title(f"{model_name} - Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(list(epochs))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    accuracy_path = (
        f"results/plots/{filename}_accuracy.png"
    )

    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    # Loss graph
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        train_loss,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation Loss"
    )

    plt.title(f"{model_name} - Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(list(epochs))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_path = (
        f"results/plots/{filename}_loss.png"
    )

    plt.savefig(loss_path, dpi=300)
    plt.close()

    print(f"\n{model_name} graphs created:")
    print(f"Accuracy: {accuracy_path}")
    print(f"Loss:     {loss_path}")


# ==========================================
# 5. Generate Custom CNN graphs
# ==========================================

create_training_graphs(
    "Custom CNN",
    custom_cnn_accuracy,
    custom_cnn_val_accuracy,
    custom_cnn_loss,
    custom_cnn_val_loss
)


# ==========================================
# 6. Generate EfficientNetB0 graphs
# ==========================================

create_training_graphs(
    "EfficientNetB0",
    efficientnet_accuracy,
    efficientnet_val_accuracy,
    efficientnet_loss,
    efficientnet_val_loss
)


# ==========================================
# 7. Finish
# ==========================================

print("\nAll training graphs generated successfully!")
print("Check the results/plots folder.")
