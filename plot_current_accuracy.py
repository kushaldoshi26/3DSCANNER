import os
import torch
import matplotlib.pyplot as plt

def plot_current_status():
    ckpt_path = 'd:/3dSCANNER/data/models/best_3d_ai_model.pth'
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found!")
        return

    ckpt = torch.load(ckpt_path, map_location='cpu')
    epoch = ckpt.get('epoch', 0)
    val_loss = ckpt.get('val_loss', 0)
    train_loss = ckpt.get('train_loss', 0)

    print(f"Current best epoch: {epoch}")
    print(f"Current validation loss: {val_loss}")
    print(f"Current training loss: {train_loss}")

    # Since previous epoch history is not saved in the checkpoint, 
    # we can only plot the current state as a bar chart or single point.
    plt.figure(figsize=(8, 5))
    plt.bar(['Train Loss', 'Validation Loss'], [train_loss, val_loss], color=['blue', 'orange'])
    plt.title(f"Current Model Accuracy Status (Epoch {epoch})")
    plt.ylabel("Loss (Lower is Better)")
    plt.ylim(0, max(val_loss, train_loss) * 1.5)
    
    # Save the plot
    output_path = "d:/3dSCANNER/data/exports/current_accuracy_status.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Status plot saved to {output_path}")

if __name__ == "__main__":
    plot_current_status()
