import os
import sys
import json
import lightning as L

# Add the 'src' directory to the Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SimpleCNN
from dataset import get_data_loaders

def train():
    # --- 1. Configuration and Hyperparameters ---
    # We use relative paths from the project root
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    batch_size = 32
    max_epochs = 10
    learning_rate = 0.001

    # Check if data directory is properly set up
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"Error: Could not find training data folder at '{train_dir}'.")
        print("Please read the README.md to understand how to structure your image data.")
        return

    # --- 2. Load Data ---
    print("Loading data...")
    train_loader, val_loader, classes = get_data_loaders(data_dir, batch_size)
    num_classes = len(classes)
    print(f"Classes found ({num_classes}): {classes}")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    # --- 3. Initialize Model ---
    model = SimpleCNN(num_classes=num_classes, learning_rate=learning_rate)

    # --- 4. Initialize Trainer ---
    # Lightning handles device placement automatically.
    trainer = L.Trainer(
        max_epochs=max_epochs,
        default_root_dir=model_dir, # Checkpoints and logs will be saved here
        accelerator="auto",         # Automatically choose GPU if available, else CPU
        devices="auto",
        log_every_n_steps=1,        # Added this to prevent the logging interval warning on small datasets
    )

    # --- 5. Train the Model ---
    print("Starting training with PyTorch Lightning...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("\nTraining complete!")
    print("Checkpoints are saved in the models/lightning_logs directory.")
    print("Class names are saved in models/classes.json.")

if __name__ == "__main__":
    train()

