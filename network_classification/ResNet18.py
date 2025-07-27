#####################################################
# ResNet18 Model Training
# groups are A and B
# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#####################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from tqdm import tqdm


cudnn.benchmark = True
plt.ion()  # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Convert images to PyTorch tensors (multi-dimensional arrays)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# Load the dataset
data_dir = "split_data"  ###################################################################### TODO: change
# Create a dictionary with two datasets: one for training ("train") and one for validation ("val")
# Each dataset loads images from the corresponding folder and applies the appropriate transformations
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}
# Create a dictionary with two DataLoaders: one for training and one for validation
# Each DataLoader handles batching, shuffling (for training), and parallel data loading
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=4, shuffle=True, num_workers=0
    )
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes

# Automatically sets the device to GPU if available (CUDA), or falls back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize lists to store training and validation losses and accuracies
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    Trains a PyTorch model using a given optimizer, loss function, and learning rate scheduler.

    Args:
        model (torch.nn.Module): The neural network model to train.
        criterion (torch.nn.modules.loss._Loss): The loss function used to evaluate prediction error (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimization algorithm that updates model weights (e.g., SGD, Adam).
        scheduler (torch.optim.lr_scheduler._LRScheduler): Adjusts the learning rate during training (e.g., StepLR).
        num_epochs (int, optional): Number of full training cycles (epochs). Default is 25.

    Returns:
        torch.nn.Module: The trained model with the best validation accuracy.
    """
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(
                    dataloaders[phase],
                    desc=f"{phase.upper()} Epoch {epoch}",
                    leave=False,
                ):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                # Store the loss and accuracy for plotting later
                if phase == "train":
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc.item())
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc.item())

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

        # Plot training and validation losses and accuracies
        # Plot loss
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"training_progress_LR0.0001_gamma{exp_lr_scheduler.gamma}.png")
    return model


### change only last layer - freeze all other layers - feature extraction
# model_conv = Convolutional feature extractor

# Load a pre-trained ResNet-18 model with weights trained on ImageNet
model_conv = torchvision.models.resnet18(weights="IMAGENET1K_V1")
# Freeze all the layers so that their weights are not updated during training
# Only the final fully-connected layer (fc) will be trained
for param in model_conv.parameters():
    param.requires_grad = False

# Get the number of input features to the final fully-connected layer
num_ftrs = model_conv.fc.in_features
# Replace the final layer with a new one that has 2 output classes (e.g., class A and class B)
model_conv.fc = nn.Linear(num_ftrs, 2)

# Move the model to the appropriate device (GPU if available, else CPU)
model_conv = model_conv.to(device)

# Define the loss function: CrossEntropyLoss is standard for classification tasks
criterion = nn.CrossEntropyLoss()

# Define the optimizer – here we only pass the parameters of the final layer (fc)
# since all other layers are frozen and don't need to be updated
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.0001, momentum=0.9)

# Define a learning rate scheduler that decays the learning rate by a factor of 0.5 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.5)


if __name__ == "__main__":
    # Train the model using the defined parameters
    # This will train only the final layer (fc) while keeping all other layers frozen
    model_conv = train_model(
        model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25
    )

    # Save the trained model parameters to a file
    # This allows us to load the model later without retraining
    torch.save(model_conv.state_dict(), "model_conv_no_reg_A_vs_B.pth")


#### change all the layers - fine-tuning the whole model
# # Load a pre-trained ResNet-18 model with weights trained on ImageNet
# # This gives us a strong starting point instead of training from scratch
# model_ft = models.resnet18(weights="IMAGENET1K_V1")   # model_ft = Fine-Tuned Model

# # Get the number of input features to the final (fully connected) layer
# num_ftrs = model_ft.fc.in_features

# # Replace the final layer with a new one that has 2 output classes (for example: class A and class B)
# # If you have more than 2 classes, you can use: nn.Linear(num_ftrs, len(class_names))
# model_ft.fc = nn.Linear(num_ftrs, 2)

# # Move the model to the appropriate device (GPU if available, otherwise CPU)
# model_ft = model_ft.to(device)

# # Define the loss function: CrossEntropyLoss is standard for classification tasks
# criterion = nn.CrossEntropyLoss()

# # Define the optimizer: here we're using SGD with a small learning rate and momentum
# # This will update all model parameters during training
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# # Define a learning rate scheduler:
# # Every 7 epochs, reduce the learning rate by a factor of 0.1
# # This helps the model learn quickly at first and then fine-tune gently
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
