#####################################################
# ShuffleNet Model Training
# groups are A and B
# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#####################################################

import torch  # for tensor operations
import torch.nn as nn  # for neural network modules
import torch.optim as optim  # for optimization algorithms
from torch.optim import lr_scheduler  # for learning rate scheduling
import torch.backends.cudnn as cudnn  # for optimized GPU performance
import torchvision  # for computer vision tasks
from torchvision import (
    datasets,  # allows loading custom folders of images using ImageFolder.
    models,  # Contains many pre-trained deep learning models
    transforms,  # Includes tools for preprocessing and augmenting images
)
import matplotlib.pyplot as plt  # for plotting graphs
import time  # for measuring time - training duration
import os  # for file and directory operations
from tempfile import TemporaryDirectory  # for creating temporary directories
from tqdm import tqdm  # for progress bars

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class FilenameDataset(Dataset):
    """
    Dataset for loading images by filename. loads images from a specified directory using a list of filenames and their corresponding labels. Applies transformations to the images as specified.
    """
    
    def __init__(self, filenames, labels, image_dir, transform):
        self.filenames = filenames
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Set up CUDA for GPU acceleration if available
cudnn.benchmark = True
plt.ion()  # Enables interactive plotting mode for real-time graph updates

# Training data uses augmentation for better generalization
# Validation data uses only resizing and normalization
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224
            # transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally (data augmentation)
            # transforms.RandomPerspective(
            #     distortion_scale=0.5, p=0.5
            # ),  # Random perspective transformation
            transforms.ToTensor(),  # Convert images to PyTorch tensors (multi-dimensional arrays)
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # Standard ImageNet normalization - mean and std for each color channel
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# Load the dataset
data_dir = "split_data"  # Folder structure: split_data/train/A, split_data/train/B, split_data/val/A, split_data/val/B


# Create a dictionary with two datasets: one for training ("train") and one for validation ("val")
# Each dataset loads images from the corresponding folder and applies the appropriate transformations
def get_dataloaders(data_dir="split_data", batch_size=50):
    """
    Load datasets for training and validation using ImageFolder and create DataLoaders.
    """
    # Load datasets and apply the appropriate transform (train or val)
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    # Wrap each dataset in a DataLoader to provide batches, shuffling, and parallel loading
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,  # Randomize sample order each epoch
            num_workers=2,  # Load data in parallel (2 worker threads)
            persistent_workers=True,  # Keep workers alive between epochs for efficiency
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {
        x: len(image_datasets[x]) for x in ["train", "val"]
    }  # Get the size of each dataset
    class_names = image_datasets[
        "train"
    ].classes  # Get the class names (subfolder names)
    return dataloaders, dataset_sizes, class_names

def get_dataloaders_from_lists(filenames, labels, image_dir="female_faces", batch_size=25):
    """
    Load datasets for training and validation from lists of filenames and labels, and create DataLoaders.
    This function splits the provided filenames and labels into training and validation sets, creates custom datasets using the FilenameDataset class, and wraps them in DataLoaders for batch processing.
    """
    if len(filenames) < 4 or min(labels.count(0), labels.count(1)) < 2:
        train_files = filenames
        train_labels = labels

        val_files = filenames
        val_labels = labels
    else:
        train_files, val_files, train_labels, val_labels = train_test_split(
            filenames,
            labels,
            train_size=0.8,
            random_state=42,
            stratify=labels,
        )

    image_datasets = {
        "train": FilenameDataset(train_files, train_labels, image_dir, data_transforms["train"]),
        "val": FilenameDataset(val_files, val_labels, image_dir, data_transforms["val"]),
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            persistent_workers=False,
        )
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = ["A", "B"]

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


@torch.no_grad()  # Disable gradient calculation for evaluation (saves memory and computations)
def evaluate(model, dataloader, criterion, device):
    """
    Evaluate a trained model on a given dataset without updating its parameters.
    The function runs the model in evaluation mode, disables gradient computation,
    iterates over all batches in the provided DataLoader, and computes:
    - the average loss over all samples
    - the classification accuracy over all samples
    """
    model.eval()  # Set model to evaluation mode
    running_loss, running_corrects, n = 0.0, 0, 0  # Initialize metrics
    for inputs, labels in dataloader:  # Iterate over batches
        inputs, labels = inputs.to(device), labels.to(
            device
        )  # Move data to the appropriate device

        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels.float())
        preds = (outputs >= 0).long()
        # outputs = model(inputs)  # Forward pass
        # loss = criterion(outputs, labels)  # Compute loss
        # _, preds = torch.max(outputs, 1)  # Get predicted classes
        # outputs = model(inputs).squeeze(1)
        # probs = torch.sigmoid(outputs)
        # loss = criterion(probs, labels.float())
        # preds = (probs >= 0.5).long()

        running_loss += loss.item() * inputs.size(0)  # Accumulate loss
        running_corrects += (
            (preds == labels).sum().item()
        )  # Accumulate correct predictions
        n += inputs.size(0)  # Accumulate number of samples
    return running_loss / max(1, n), running_corrects / max(
        1, n
    )  # Return average loss and accuracy


def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
    plots=True,
):
    """
    Train a PyTorch model using supervised learning and track performance over epochs.

    The function performs the following steps:
    - Evaluates initial train and validation performance before training.
    - Trains the model for a fixed number of epochs.
    - At each epoch:
        * Updates model parameters using the training set.
        * Evaluates performance on the validation set.
        * Re-evaluates training performance in evaluation mode.
        * Tracks loss and accuracy.
        * Saves the model with the best validation accuracy.
    - Restores the best-performing model at the end of training.
    - Optionally generates and saves loss/accuracy plots.

    Args:
        model (torch.nn.Module): The neural network model to train.
        dataloaders (dict[str, torch.utils.data.DataLoader]):
            Dictionary containing DataLoaders for each phase.
            Expected keys are "train" and "val".
        dataset_sizes (dict[str, int]):
            Dictionary containing the number of samples in each dataset split.
            Expected keys are "train" and "val".
            Used to compute average loss and accuracy.
        criterion (torch.nn.modules.loss._Loss): The loss function used to evaluate prediction error (e.g.,     CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimization algorithm that updates model weights (e.g., SGD, Adam).
        scheduler (torch.optim.lr_scheduler._LRScheduler): Adjusts the learning rate during training.
        num_epochs (int, optional): Number of full training cycles (epochs). Default is 25.
        plots (bool, optional): Whether to generate and save training/validation loss and accuracy plots. Default is True.

    Returns:
        torch.nn.Module: The trained model with the best validation accuracy.
    """
    since = time.time()  # Record the start time of training

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(
            tempdir, "best_model_params.pt"
        )  # Path to save best model parameters

        torch.save(
            model.state_dict(), best_model_params_path
        )  # Save initial model parameters
        best_acc = 0.0  # Initialize best accuracy

        # Evaluate the model on the training and validation sets before training
        init_train_loss, init_train_acc = evaluate(
            model, dataloaders["train"], criterion, device
        )
        init_val_loss, init_val_acc = evaluate(
            model, dataloaders["val"], criterion, device
        )

        # Store initial losses and accuracies
        reeval_train_losses.append(init_train_loss)
        reeval_train_accuracies.append(init_train_acc)
        val_losses.append(init_val_loss)
        val_accuracies.append(init_val_acc)

        print(
            f"[INIT] Train: loss={init_train_loss:.4f}, acc={init_train_acc:.4f} | "
            f"Val: loss={init_val_loss:.4f}, acc={init_val_acc:.4f}"
        )

        # Main training loop
        for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
            epoch_start_time = time.time()  # Start time for the epoch
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 10)

            # Each epoch has a training and validation phase

            # -- Train Phase --
            model.train()  # Set model to training mode
            running_loss = (
                0.0  # Initialize running loss - for accumulating loss over batches
            )
            running_corrects = 0  # Initialize running correct predictions - for accumulating correct predictions

            # Timing variables - to measure data loading and model computation times
            data_loading_time = 0.0
            forward_backward_time = 0.0

            t0 = time.time()  # Start time for data loading
            # Iterate over data.
            for inputs, labels in dataloaders["train"]:
                t1 = time.time()  # End time for data loading
                data_loading_time += t1 - t0  # Accumulate data loading time

                inputs = inputs.to(
                    device
                )  # Move the input images to the selected device (GPU if available, otherwise CPU).
                labels = labels.to(
                    device
                )  # Move the labels to the selected device (GPU if available, otherwise CPU).

                # zero the parameter gradients - in order to prevent accumulation from previous batches
                optimizer.zero_grad()

                t2 = time.time()  # Start time for model computation
                ## forward ##

                # outputs = model(
                #     inputs
                # )  # Forward pass: compute predicted outputs by passing inputs to the model
                # _, preds = torch.max(
                #     outputs, 1
                # )  # Get the predicted class with the highest score
                # loss = criterion(
                #     outputs, labels
                # )  # Compute the loss between predicted outputs and true labels
                # outputs = model(inputs).squeeze(1)
                # probs = torch.sigmoid(outputs)
                # preds = (probs >= 0.5).long()
                # loss = criterion(probs, labels.float())
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels.float())
                preds = (outputs >= 0).long()

                ## backward + optimize ##
                loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # Update model parameters based on computed gradients

                t3 = time.time()  # End time for model computation
                forward_backward_time += t3 - t2  # Accumulate model computation time

                # statistics
                running_loss += loss.item() * inputs.size(0)  # Accumulate loss
                running_corrects += (
                    (preds == labels).sum().item()
                )  # Accumulate correct predictions

                t0 = time.time()  # Start time for next data loading

            scheduler.step()  # Adjust learning rate according to the scheduler

            epoch_loss = (
                running_loss / dataset_sizes["train"]
            )  # Compute average loss for the epoch
            epoch_acc = (
                running_corrects / dataset_sizes["train"]
            )  # Compute accuracy for the epoch
            train_losses.append(epoch_loss)  # Store training loss
            train_accuracies.append(float(epoch_acc))  # Store training accuracy

            print(
                f"[Train] Data loading time: {data_loading_time:.2f}s, "
                f"Model compute time: {forward_backward_time:.2f}s"
            )

            # -- Validation Phase --
            val_loss, val_acc = evaluate(
                model, dataloaders["val"], criterion, device
            )  # Evaluate on validation set
            val_losses.append(val_loss)  # Store validation loss
            val_accuracies.append(val_acc)  # Store validation accuracy
            print(f"[val] Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc > best_acc:  # If this is the best validation accuracy so far
                best_acc = val_acc  # Update best accuracy
                torch.save(
                    model.state_dict(), best_model_params_path
                )  # Save the model parameters

            # ---- RECALC TRAIN (eval mode, fair comparison) ----
            # Re-evaluate training performance in evaluation mode
            true_train_loss, true_train_acc = evaluate(
                model, dataloaders["train"], criterion, device
            )  # Evaluate on training set
            reeval_train_losses.append(
                true_train_loss
            )  # Store re-evaluated training loss
            reeval_train_accuracies.append(
                true_train_acc
            )  # Store re-evaluated training accuracy
            print(
                f"[RECALC] Train Loss (eval mode): {true_train_loss:.4f}, "
                f"Acc: {true_train_acc:.4f}"
            )

            # -- Epoch total time --
            epoch_total_time = (
                time.time() - epoch_start_time
            )  # Total time for the epoch
            print(f"[Epoch {epoch+1}] Total time: {epoch_total_time:.2f}s")

        # -- End training --
        # Measure the total training time
        time_elapsed = time.time() - since
        print(
            f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        # model.load_state_dict(torch.load(best_model_params_path, weights_only=True)) 
        # so now that we commented out the model.load_state_dict line, the model will be left with the parameters from the last epoch, not the best epoch.

        # -- Plots --
        if plots == True:
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
            plt.savefig(f"training_progress_CE.png")
    return model


def train_model_fast_for_self_training(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=1,
):
    """
    A faster version of the train_model function that only trains for a few epochs without evaluating on the validation set or recalculating training performance in evaluation mode. This is useful for quickly generating predictions for self-training or pseudo-labeling without the overhead of tracking performance metrics or saving the best model.
    The evaluation and tracking of training/validation performance is skipped to save time, making this function suitable for scenarios where the primary goal is to quickly update the model parameters rather than monitor training progress or select the best model based on validation accuracy.
    """
    
    since = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        running_corrects = 0
        data_loading_time = 0.0
        forward_backward_time = 0.0

        t0 = time.time()

        for inputs, labels in dataloaders["train"]:
            t1 = time.time()
            data_loading_time += t1 - t0

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            t2 = time.time()

            # outputs = model(inputs).squeeze(1)
            # probs = torch.sigmoid(outputs)
            # preds = (probs >= 0.5).long()
            # loss = criterion(probs, labels.float())

            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # _, preds = torch.max(outputs, 1)

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels.float())
            preds = (outputs >= 0).long()

            loss.backward()
            optimizer.step()

            t3 = time.time()
            forward_backward_time += t3 - t2

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()

            t0 = time.time()

        scheduler.step()

        epoch_loss = running_loss / dataset_sizes["train"]
        epoch_acc = running_corrects / dataset_sizes["train"]

        epoch_total_time = time.time() - epoch_start_time

        print(
            f"[Train fast] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
            f"Data: {data_loading_time:.2f}s | "
            f"Compute: {forward_backward_time:.2f}s | "
            f"Total: {epoch_total_time:.2f}s"
        )

    print(f"Fast training completed in {time.time() - since:.2f}s")

    return model





### change all the layers - fine-tuning the whole model
# Load a pre-trained ShuffleNet V2 (x0.5) model with weights trained on ImageNet
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
    # Finally, the last layer outputs logits for the two classes (A and B)
    # model_ft.fc = nn.Sequential(
    #     nn.Dropout(p=0.5),
    #     nn.Linear(num_ftrs, 256),
    #     nn.ReLU(),
    #     nn.Dropout(p=0.3),
    #     nn.Linear(256, 2),
    # )

    model_ft.fc = nn.Linear(num_ftrs, 1, bias=False)

    # Move the model to the appropriate device (GPU if available, otherwise CPU)
    model_ft = model_ft.to(device)

    # Define the loss function: CrossEntropyLoss is standard for classification tasks
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()  # For binary classification (A vs B)

    # Define the optimizer: here we're using AdamW with a small learning rate
    # This will update all model parameters during training
    # AdamW helps prevent overfitting through weight decay (L2 regularization)
    # weight_decay=0.01 adds a penalty to large weights in the loss function to encourage smaller weights and better generalization
    optimizer_ft = optim.AdamW(
        model_ft.parameters(), lr=0.005, weight_decay=0.1
    )

    # Define a learning rate scheduler:
    # Every 5 epochs, reduce the learning rate by a factor of 0.1
    # This helps the model learn quickly at first and then fine-tune gently
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    return model_ft, criterion, optimizer_ft, exp_lr_scheduler


if __name__ == "__main__":
    print(f"Using device: {device}")
    dataloaders, dataset_sizes, class_names = get_dataloaders(
        batch_size=50
    )  # Load data
    model_ft, criterion, optimizer_ft, exp_lr_scheduler = (
        create_model_and_optim()
    )  # Create model and optimizer
    # Train the model using the defined parameters - full fine-tuning
    model_ft = train_model(
        model_ft,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=10,
    )

    # Save the trained model parameters to a file
    # This allows us to load the model later without retraining
    torch.save(model_ft.state_dict(), "model_ft_0_BCE_no_bias.pth")

    
    print("\nChecking saved model...\n")

    train_loss, train_acc = evaluate(
        model_ft,
        dataloaders["train"],
        criterion,
        device,
    )

    val_loss, val_acc = evaluate(
        model_ft,
        dataloaders["val"],
        criterion,
        device,
    )

    print(f"TRAIN accuracy = {train_acc*100:.2f}%")
    print(f"VAL accuracy   = {val_acc*100:.2f}%")