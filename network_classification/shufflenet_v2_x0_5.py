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
            transforms.RandomHorizontalFlip(),
            # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
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
            num_workers=2,
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
            epoch_start_time = time.time()
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 10)

            # Each epoch has a training and validation phase

            # -- Train Phase --
            model.train()  # Set model to training mode
            running_loss = 0.0
            running_corrects = 0

            data_loading_time = 0.0
            forward_backward_time = 0.0

            t0 = time.time()
            # Iterate over data.
            for inputs, labels in dataloaders["train"]:
                t1 = time.time()
                data_loading_time += t1 - t0

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                t2 = time.time()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                # backward + optimize
                loss.backward()
                optimizer.step()

                t3 = time.time()
                forward_backward_time += t3 - t2

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

                t0 = time.time()

            scheduler.step()

            epoch_loss = running_loss / dataset_sizes["train"]
            epoch_acc = running_corrects / dataset_sizes["train"]
            train_losses.append(epoch_loss)
            train_accuracies.append(float(epoch_acc))

            print(
                f"[Train] Data loading time: {data_loading_time:.2f}s, "
                f"Model compute time: {forward_backward_time:.2f}s"
            )

            # -- Validation Phase --
            val_loss, val_acc = evaluate(model, dataloaders["val"], criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            print(f"[val] Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_params_path)

            # ---- RECALC TRAIN (eval mode, fair comparison) ----
            true_train_loss, true_train_acc = evaluate(
                model, dataloaders["train"], criterion, device
            )
            reeval_train_losses.append(true_train_loss)
            reeval_train_accuracies.append(true_train_acc)
            print(
                f"[RECALC] Train Loss (eval mode): {true_train_loss:.4f}, "
                f"Acc: {true_train_acc:.4f}"
            )

            # -- Epoch total time --
            epoch_total_time = time.time() - epoch_start_time
            print(f"[Epoch {epoch+1}] Total time: {epoch_total_time:.2f}s")

        # -- End training --
        # Measure the total training time
        time_elapsed = time.time() - since
        print(
            f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

        # -- Plots --
        # Plot training and validation losses and accuracies
        # Plot loss
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(reeval_train_losses, label="Train Loss")
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
        plt.savefig(f"training_progress_no_reg.png")
    return model


### change only last layer - freeze all other layers - feature extraction
# model_conv = Convolutional feature extractor
def create_model_and_optim_feature_extraction():
    # Load a pre-trained ShuffleNet V2 model with weights trained on ImageNet
    model_conv = torchvision.models.shufflenet_v2_x0_5(weights="IMAGENET1K_V1")
    # Freeze all layers except the final fully-connected layer (fc) and stage4
    for param in model_conv.parameters():
        param.requires_grad = False
    for param in model_conv.stage4.parameters():
        param.requires_grad = True
    for param in model_conv.fc.parameters():
        param.requires_grad = True

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
    optimizer_conv = optim.Adam(
        list(model_conv.stage4.parameters()) + list(model_conv.fc.parameters()),
        lr=3e-4,
    )
    # optimizer_conv = optim.AdamW(model_conv.parameters(), lr=0.00005, weight_decay=0.01)

    # Define a learning rate scheduler that decays the learning rate by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.5)
    return model_conv, criterion, optimizer_conv, exp_lr_scheduler


##################################################################################################


### change all the layers - fine-tuning the whole model
# Load a pre-trained ResNet-18 model with weights trained on ImageNet
# This gives us a strong starting point instead of training from scratch
def create_model_and_optim():
    model_ft = models.shufflenet_v2_x0_5(
        weights="IMAGENET1K_V1"
    )  # model_ft = Fine-Tuned Model

    # Get the number of input features to the final (fully connected) layer
    num_ftrs = model_ft.fc.in_features

    # Here, we add dropout layers to prevent overfitting
    # The first dropout layer has a dropout probability of 0.5, and the second has 0.3
    # This means that during training, 50% and 30% of the neurons will be randomly set to zero
    # This helps the model generalize better by preventing it from relying too much on any single neuron
    # The final layer has 256 neurons followed by a ReLU activation function, and then another dropout layer
    # Finally, the last layer outputs 2 classes (A and B)
    model_ft.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 2),
    )

    # Move the model to the appropriate device (GPU if available, otherwise CPU)
    model_ft = model_ft.to(device)

    # Define the loss function: CrossEntropyLoss is standard for classification tasks
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer: here we're using Adam with a small learning rate
    # This will update all model parameters during training
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.01, weight_decay=0.01)

    # Define a learning rate scheduler:
    # Every 7 epochs, reduce the learning rate by a factor of 0.1
    # This helps the model learn quickly at first and then fine-tune gently
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.5)

    return model_ft, criterion, optimizer_ft, exp_lr_scheduler


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
        num_epochs=15,
    )

    # Save the trained model parameters to a file
    # This allows us to load the model later without retraining
    torch.save(model_ft.state_dict(), "model_ft_no_reg_45.pth")
