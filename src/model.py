import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

class SimpleCNN(L.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super(SimpleCNN, self).__init__()
        self.save_hyperparameters() # Saves num_classes and learning_rate to self.hparams
        
        # 1. First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        # 2. Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3. Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # 4. Fully Connected Layers
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Pass data through convolution -> ReLU activation function -> pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the 3D tensor into a 1D vector for the fully connected layers
        x = x.view(x.size(0), -1) 
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        # Log metrics to the progress bar and TensorBoard.
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

