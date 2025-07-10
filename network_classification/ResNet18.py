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

cudnn.benchmark = True
plt.ion()  # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.RandomHorizontalFlip(),
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
data_dir = "data/hymenoptera_data"  ###################################################################### to change
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
        image_datasets[x], batch_size=4, shuffle=True, num_workers=4
    )
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes

# We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


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

        for epoch in range(num_epochs):
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
                for inputs, labels in dataloaders[phase]:
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
    return model


# Load a pre-trained ResNet-18 model with weights trained on ImageNet
# This gives us a strong starting point instead of training from scratch
model_ft = models.resnet18(weights="IMAGENET1K_V1")

# Get the number of input features to the final (fully connected) layer
num_ftrs = model_ft.fc.in_features

# Replace the final layer with a new one that has 2 output classes (for example: class A and class B)
# If you have more than 2 classes, you can use: nn.Linear(num_ftrs, len(class_names))
model_ft.fc = nn.Linear(num_ftrs, 2)

# Move the model to the appropriate device (GPU if available, otherwise CPU)
model_ft = model_ft.to(device)

# Define the loss function: CrossEntropyLoss is standard for classification tasks
criterion = nn.CrossEntropyLoss()

# Define the optimizer: here we're using SGD with a small learning rate and momentum
# This will update all model parameters during training
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Define a learning rate scheduler:
# Every 7 epochs, reduce the learning rate by a factor of 0.1
# This helps the model learn quickly at first and then fine-tune gently
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# Train and evaluate
model_ft = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25
)
