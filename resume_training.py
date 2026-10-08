import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.train_ai_3d import train_custom_3d_ai_model, TRAINING_STATE

def plot_training_history():
    history = TRAINING_STATE.get("history", [])
    if not history:
        print("No training history available to plot.")
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss", marker='o')
    plt.plot(epochs, val_loss, label="Validation Loss", marker='o')
    
    plt.title("Model Training Loss (Accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE + Chamfer + Laplacian + Normal)")
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    output_path = "data/exports/training_graph.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Training graph saved to {output_path}")

if __name__ == "__main__":
    print("Resuming training from the last checkpoint...")
    # Target epochs set to 300. The checkpoint is currently at epoch 217.
    # It will automatically resume and train the difference.
    best_model = train_custom_3d_ai_model(
        epochs=300, 
        batch_size=16, 
        resume=True, 
        patience=50
    )
    
    print("\nTraining complete! Best model saved at:", best_model)
    plot_training_history()
