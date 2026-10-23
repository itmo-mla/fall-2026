import matplotlib.pyplot as plt
import numpy as np


def draw_loss(loss):
    # 1. Extract values using list comprehensions
    # Note: Use dict.get() to safely avoid KeyErrors on the missing final test item
    train_losses = [epoch.get("train") for epoch in loss if "train" in epoch]
    test_losses = [epoch.get("test") for epoch in loss if "test" in epoch]

    # Create sequential arrays for the X-axes (Epochs)
    epochs_train = np.arange(1, len(train_losses) + 1)
    epochs_test = np.arange(1, len(test_losses) + 1)

    # 2. Build the Matplotlib figure
    plt.figure(figsize=(8, 5))

    # Plot both lines on the single plot canvas
    plt.plot(
        epochs_train,
        train_losses,
        label="Train Loss",
        color="blue",
        marker="o",
        linewidth=2,
    )
    plt.plot(
        epochs_test,
        test_losses,
        label="Test Loss",
        color="orange",
        marker="s",
        linewidth=2,
    )

    # Add titles, labels, and cosmetic elements
    plt.title("Model Training Loss vs. Test Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    # Display the final plot
    plt.tight_layout()
