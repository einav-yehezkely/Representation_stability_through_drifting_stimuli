#####################################################
# VGG-16 Model Training
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
            # transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.Resize((128, 128)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ToTensor(),  # Convert images to PyTorch tensors (multi-dimensional arrays)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            # transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# Load the dataset
data_dir = "split_data"


# Create a dictionary with two datasets: one for training ("train") and one for validation ("val")
# Each dataset loads images from the corresponding folder and applies the appropriate transformations
def get_dataloaders(data_dir="split_data", batch_size=128):
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    return dataloaders, dataset_sizes, class_names


# Automatically sets the device to GPU if available (CUDA), or falls back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize lists to store training and validation losses and accuracies
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
reeval_train_losses = []
reeval_train_accuracies = []


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate loss and accuracy over a whole dataloader in eval mode."""
    model.eval()
    running_loss, running_corrects, n = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        n += inputs.size(0)
    return running_loss / max(1, n), running_corrects / max(1, n)


def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25
):
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

        # Evaluate the model on the training and validation sets before training
        init_train_loss, init_train_acc = evaluate(
            model, dataloaders["train"], criterion, device
        )
        init_val_loss, init_val_acc = evaluate(
            model, dataloaders["val"], criterion, device
        )

        reeval_train_losses.append(init_train_loss)
        reeval_train_accuracies.append(init_train_acc)

        val_losses.append(init_val_loss)
        val_accuracies.append(init_val_acc)

        print(
            f"[INIT] Train: loss={init_train_loss:.4f}, acc={init_train_acc:.4f} | "
            f"Val: loss={init_val_loss:.4f}, acc={init_val_acc:.4f}"
        )

        for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                data_loading_time = 0.0
                forward_backward_time = 0.0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    t0 = time.time()

                    t1 = time.time()
                    data_loading_time += t1 - t0

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    t2 = time.time()
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
                    t3 = time.time()
                    forward_backward_time += t3 - t2

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                print(
                    f"[{phase}] Data loading time: {data_loading_time:.2f}s, "
                    f"Model compute time: {forward_backward_time:.2f}s"
                )
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

            ### Re-evaluate the training set after each epoch - Recompute train loss in eval mode for fair comparison
            # Set the model to evaluation mode to ensure consistent behavior
            model.eval()
            # Initialize accumulators for loss and correct predictions on the training set
            running_loss = 0.0
            running_corrects = 0
            # Disable gradient tracking to reduce memory usage and speed up computations
            with torch.no_grad():
                # Iterate over the entire training set
                for inputs, labels in dataloaders["train"]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Forward pass: compute model outputs and predictions
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # Accumulate loss and correct predictions
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            # Calculate the average loss and accuracy over the training set
            true_train_loss = running_loss / dataset_sizes["train"]
            true_train_acc = running_corrects.double() / dataset_sizes["train"]
            # Store the re-evaluated training loss and accuracy for plotting
            reeval_train_losses.append(true_train_loss)
            reeval_train_accuracies.append(true_train_acc.item())

            print(
                f"[RECALC] Train Loss (eval mode): {true_train_loss:.4f}, Acc: {true_train_acc:.4f}"
            )

        time_elapsed = time.time() - since
        print(
            f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

        # Plot training and validation losses and accuracies
        # Plot loss
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(reeval_train_losses, label="Train Loss")
        # plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(reeval_train_accuracies, label="Train Accuracy")
        # plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"training_progress_reg1.png")
    return model


### change only last layer - freeze all other layers - feature extraction
# model_conv = Convolutional feature extractor
def create_model_and_optim_feature_extraction():
    # Use a very lightweight backbone that's fast on CPU
    model = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
    )

    # Replace the final classification layer to output 2 classes (A/B)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, 2)

    # Freeze all parameters by default (no gradients, no updates)
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze ONLY the head
    for p in model.classifier[-1].parameters():
        p.requires_grad = True

    # Move model to the selected device (CPU in your setup)
    model = model.to(device)

    # Loss with mild label smoothing helps stabilize training on small datasets / 2 classes
    criterion = nn.CrossEntropyLoss()

    # Train only the head
    optimizer = torch.optim.Adam(model.classifier[-1].parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    return model, criterion, optimizer, scheduler


##################################################################################################


### change all the layers - fine-tuning the whole model
def create_model_and_optim():
    model = torchvision.models.VGG(weights="IMAGENET1K_V1")

    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, 2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    return model, criterion, optimizer, scheduler


if __name__ == "__main__":
    print(f"Using device: {device}")
    dataloaders, dataset_sizes, class_names = get_dataloaders(batch_size=128)
    model_ft, criterion, optimizer_ft, exp_lr_scheduler = (
        create_model_and_optim_feature_extraction()
    )
    # Train the model using the defined parameters
    # This will train only the final layer (fc) while keeping all other layers frozen
    model_ft = train_model(
        model_ft,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=25,
    )

    # Save the trained model parameters to a file
    # This allows us to load the model later without retraining
    torch.save(model_ft.state_dict(), "model_ft_no_reg_135.pth")
