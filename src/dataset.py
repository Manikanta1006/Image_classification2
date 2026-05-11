import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32):
    """
    Sets up the dataset loaders for training and validation.
    """
    
    # 1. Define Image Transformations
    # We resize images to 224x224 to keep input dimensions consistent.
    # We also convert them to PyTorch Tensors and normalize the pixel values.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # 2. Load the Training Dataset
    # ImageFolder automatically infers the classes based on the folder names.
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Load the Validation Dataset (Optional)
    val_loader = None
    if os.path.exists(val_dir):
        val_dataset = ImageFolder(root=val_dir, transform=transform)
        if val_dataset.classes != train_dataset.classes:
            raise ValueError(
                "Validation class folders must match training class folders. "
                f"Train classes: {train_dataset.classes}; "
                f"Validation classes: {val_dataset.classes}"
            )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset.classes
