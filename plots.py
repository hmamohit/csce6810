import matplotlib.pyplot as plt

def plot_training_curves(history, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train MSE")
    axes[0].plot(history["val_loss"],   label="Val MSE")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_mae"], label="Train MAE")
    axes[1].plot(history["val_mae"],   label="Val MAE")
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved → {save_path}")


def plot_predictions(preds, targets, save_path="predictions.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mn, mx = targets.min(), targets.max()
    axes[0].scatter(targets, preds, alpha=0.4, s=15)
    axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_xlabel("True Value")
    axes[0].set_ylabel("Predicted Value")
    axes[0].set_title("Predicted vs True")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residuals = preds - targets
    axes[1].hist(residuals, bins=40, color="steelblue",
                 edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual (Predicted − True)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Prediction plots saved → {save_path}")