############################################################
# FAST ARCFACE RESNET50 512D + MLP CLASSIFIER
#
# filename
#    ↓
# saved ArcFace ResNet50 512D embedding
#    ↓
# MLP:
# 512 -> 64 -> 2
#    ↓
# A / B
#
# No images are opened during training.
# No face detection or ArcFace forward pass is performed.
# Because all ResNet50 embeddings are precomputed and saved,
# classifier training is very fast.
############################################################

import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import copy

import torch.nn.functional as F

import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# PATHS
# ============================================================

EMBEDDINGS_CSV = "female_arcface_embeddings.csv"

DATA_DIR = "split_data"


# ============================================================
# LOAD ARCFACE RESNET50 EMBEDDINGS
#
# CSV format:
#
# column 0     = filename
# columns 1-512 = ArcFace ResNet50 embedding
#
# CSV has NO header.
# ============================================================

print()
print("=" * 70)
print("LOADING ARCFACE RESNET50 EMBEDDINGS")
print("=" * 70)


embeddings_df = pd.read_csv(
    EMBEDDINGS_CSV,
    header=None,
)


print(
    "CSV shape:",
    embeddings_df.shape
)


# ============================================================
# VALIDATE CSV
# ============================================================

expected_columns = 513


if embeddings_df.shape[1] != expected_columns:

    raise RuntimeError(
        f"Expected 513 columns "
        f"(1 filename + 512 ArcFace ResNet50 values), "
        f"but found {embeddings_df.shape[1]}"
    )


print(
    "Number of images:",
    len(embeddings_df)
)

print(
    "Embedding dimension:",
    embeddings_df.shape[1] - 1
)


# ============================================================
# BUILD FAST LOOKUP
#
# filename -> 512D numpy vector
# ============================================================

embedding_lookup = {}


for _, row in embeddings_df.iterrows():

    filename = str(
        row.iloc[0]
    ).strip()


    basename = os.path.basename(
        filename
    )


    vector = row.iloc[
        1:513
    ].to_numpy(
        dtype=np.float32
    )

    # L2-anormlize the ArcFace embedding
    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm


    embedding_lookup[
        filename
    ] = vector

    embedding_lookup[
        basename
    ] = vector


print(
    "Embeddings loaded:",
    len(embeddings_df)
)

print(
    "Example filename:",
    embeddings_df.iloc[0, 0]
)

example_name = os.path.basename(
    str(
        embeddings_df.iloc[0, 0]
    )
)

print(
    "Example embedding shape:",
    embedding_lookup[
        example_name
    ].shape
)

print("=" * 70)
print()


# ============================================================
# LOOKUP FUNCTION
# ============================================================

def get_embedding_from_filename(
    filename,
):

    filename = str(
        filename
    ).strip()


    basename = os.path.basename(
        filename
    )


    if filename in embedding_lookup:

        return embedding_lookup[
            filename
        ]


    if basename in embedding_lookup:

        return embedding_lookup[
            basename
        ]


    raise KeyError(
        f"Could not find ArcFace embedding for image: "
        f"{filename}"
    )


# ============================================================
# DATASET FROM FILE LIST
#
# Used later by rotation / self-training scripts.
#
# IMPORTANT:
# It receives filenames but returns ArcFace ResNet50 512D vectors.
# ============================================================

class FilenameDataset(Dataset):

    def __init__(
        self,
        filenames,
        labels,
        image_dir="female_faces",
    ):

        self.filenames = list(
            filenames
        )


        self.labels = [
            int(x)
            for x in labels
        ]


        # Kept only for compatibility.
        #
        # Images are NOT opened.
        self.image_dir = image_dir


    def __len__(
        self,
    ):

        return len(
            self.filenames
        )


    def __getitem__(
        self,
        idx,
    ):

        filename = self.filenames[
            idx
        ]


        embedding = get_embedding_from_filename(
            filename
        )


        embedding = torch.tensor(
            embedding,
            dtype=torch.float32,
        )


        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long,
        )


        return (
            embedding,
            label,
        )


# ============================================================
# DATASET FOR:
#
# split_data/
#     train/
#         A/
#         B/
#     val/
#         A/
#         B/
#
# The images themselves are NOT opened.
#
# Only their filenames are read from the folders.
# ============================================================

class FolderEmbeddingDataset(Dataset):
    """
    Dataset that reads images from folders A and B, but returns
    their precomputed ArcFace ResNet50 embeddings instead of the images themselves.
    """

    def __init__(
        self,
        root_dir,
    ):

        self.samples = []


        self.class_names = [
            "A",
            "B",
        ]


        missing = []


        for label, class_name in enumerate(
            self.class_names
        ):

            class_folder = os.path.join(
                root_dir,
                class_name
            )


            if not os.path.isdir(
                class_folder
            ):

                raise FileNotFoundError(
                    f"Could not find folder: "
                    f"{class_folder}"
                )


            for filename in os.listdir(
                class_folder
            ):

                if not filename.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png",
                    )
                ):

                    continue


                basename = os.path.basename(
                    filename
                )


                if basename not in embedding_lookup:

                    missing.append(
                        filename
                    )

                    continue


                self.samples.append(
                    (
                        filename,
                        label,
                    )
                )


        if missing:

            raise RuntimeError(
                f"{len(missing)} images in {root_dir} "
                f"do not exist in {EMBEDDINGS_CSV}.\n"
                f"Examples: {missing[:10]}"
            )


    def __len__(
        self,
    ):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        idx,
    ):

        filename, label = self.samples[
            idx
        ]


        embedding = get_embedding_from_filename(
            filename
        )


        embedding = torch.tensor(
            embedding,
            dtype=torch.float32,
        )


        label = torch.tensor(
            label,
            dtype=torch.long,
        )


        return (
            embedding,
            label,
        )


# ============================================================
# INITIAL TRAIN / VALIDATION DATALOADERS
# ============================================================

def get_dataloaders(
    data_dir="split_data",
    batch_size=50,
):

    datasets = {

        "train":
            FolderEmbeddingDataset(
                os.path.join(
                    data_dir,
                    "train"
                )
            ),

        "val":
            FolderEmbeddingDataset(
                os.path.join(
                    data_dir,
                    "val"
                )
            ),
    }


    dataloaders = {

        phase:
            DataLoader(
                datasets[phase],
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )

        for phase in [
            "train",
            "val",
        ]
    }


    dataset_sizes = {

        phase:
            len(
                datasets[phase]
            )

        for phase in [
            "train",
            "val",
        ]
    }


    class_names = [
        "A",
        "B",
    ]


    print()

    print(
        "Training images:",
        dataset_sizes["train"]
    )

    print(
        "Validation images:",
        dataset_sizes["val"]
    )


    return (
        dataloaders,
        dataset_sizes,
        class_names,
    )


# ============================================================
# DATALOADERS FROM IMAGE LISTS
# ============================================================

def get_dataloaders_from_lists(
    filenames,
    labels,
    image_dir="female_faces",
    batch_size=25,
):

    filenames = list(
        filenames
    )


    labels = [
        int(x)
        for x in labels
    ]


    # --------------------------------------------------------
    # Verify all images have ArcFace ResNet50 embeddings
    # --------------------------------------------------------

    missing = [
        filename
        for filename in filenames
        if os.path.basename(
            str(filename)
        ) not in embedding_lookup
    ]


    if missing:

        raise RuntimeError(
            f"{len(missing)} filenames do not have ArcFace ResNet50 "
            f"embeddings.\n"
            f"Examples: {missing[:10]}"
        )


    # --------------------------------------------------------
    # Small datasets cannot be split into train/val.
    # --------------------------------------------------------

    if (
        len(filenames) < 4
        or labels.count(0) < 2
        or labels.count(1) < 2
    ):

        train_files = filenames
        train_labels = labels

        val_files = filenames
        val_labels = labels


    else:

        (
            train_files,
            val_files,
            train_labels,
            val_labels,
        ) = train_test_split(

            filenames,
            labels,

            train_size=0.8,

            random_state=42,

            stratify=labels,
        )


    datasets = {

        "train":
            FilenameDataset(
                train_files,
                train_labels,
                image_dir,
            ),

        "val":
            FilenameDataset(
                val_files,
                val_labels,
                image_dir,
            ),
    }


    dataloaders = {

        phase:
            DataLoader(
                datasets[phase],
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )

        for phase in [
            "train",
            "val",
        ]
    }


    dataset_sizes = {

        phase:
            len(
                datasets[phase]
            )

        for phase in [
            "train",
            "val",
        ]
    }


    class_names = [
        "A",
        "B",
    ]


    return (
        dataloaders,
        dataset_sizes,
        class_names,
    )


# ============================================================
# ARCFACE RESNET50 MLP CLASSIFIER
#
# IMPORTANT:
#
# Input = 512D ArcFace ResNet50 representation
#
# PCA coordinates are NEVER provided to the model.
# ============================================================

class ArcFaceClassifier(nn.Module):

    def __init__(
        self,
    ):

        super().__init__()


        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                2,
            ),
        )

        # self.classifier = nn.Linear(512, 2) # Simple perceptron for binary classification

        # self.classifier = nn.Sequential(
        #     nn.Linear(512, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 2),
        # )

        # self.classifier = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 2),
        # )


    def forward(
        self,
        embeddings,
    ):

        return self.classifier(
            embeddings
        )


    @torch.no_grad()
    def get_embedding(
        self,
        embeddings,
    ):
        return embeddings

# ============================================================
# CREATE MODEL + LOSS + OPTIMIZER
# ============================================================

def create_model_and_optim():

    model_ft = ArcFaceClassifier()


    model_ft = model_ft.to(
        device
    )



    # criterion = nn.MSELoss(reduction='sum')
    criterion = nn.CrossEntropyLoss()


    # optimizer_ft = optim.AdamW(

    #     model_ft.classifier.parameters(),

    #     lr=0.001,

    #     weight_decay=0.0,
    # )

    # optimizer_ft = optim.AdamW(
    #     model_ft.parameters(),
    #     lr=0.001,
    #     weight_decay=0.01,
    # )

    optimizer_ft = optim.AdamW(
        model_ft.parameters(),
        lr=0.1,
        weight_decay=1,
    )


    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer_ft,

        step_size=10,

        gamma=0.1,
    )


    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    print(
        "Input: saved ArcFace ResNet50 embeddings"
    )

    print(
        "Input dimension: 512"
    )

    print(
        "Classifier: 512 -> 64 -> ReLU -> 2"
    )

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model_ft.parameters()
            if p.requires_grad
        )
    )

    print("=" * 70)
    print()


    return (
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
    )


# ============================================================
# METRICS
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

reeval_train_losses = []
reeval_train_accuracies = []


# ============================================================
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()


    running_loss = 0.0
    running_corrects = 0
    n = 0


    for embeddings, labels in dataloader:

        embeddings = embeddings.to(
            device
        )


        labels = labels.to(
            device
        )


        outputs = model(
            embeddings
        )


        loss = criterion(
            outputs,
            labels
        )

        # labels_one_hot = F.one_hot(
        #     labels,
        #     num_classes=2
        # ).float()

        # loss = criterion(
        #     outputs,
        #     labels_one_hot
        # )


        _, preds = torch.max(
            outputs,
            1
        )


        batch_size = embeddings.size(
            0
        )


        # running_loss += (
        #     loss.item()
        #     * batch_size
        # )

        running_loss += loss.item()


        running_corrects += (
            preds == labels
        ).sum().item()


        n += batch_size


    return (
        running_loss / max(1, n),
        running_corrects / max(1, n),
    )


# ============================================================
# STANDARD SUPERVISED TRAINING
# ============================================================

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

    since = time.time()


    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0


    # --------------------------------------------------------
    # Initial evaluation
    # --------------------------------------------------------

    init_train_loss, init_train_acc = evaluate(
        model,
        dataloaders["train"],
        criterion,
        device,
    )


    init_val_loss, init_val_acc = evaluate(
        model,
        dataloaders["val"],
        criterion,
        device,
    )


    reeval_train_losses.append(
        init_train_loss
    )

    reeval_train_accuracies.append(
        init_train_acc
    )

    val_losses.append(
        init_val_loss
    )

    val_accuracies.append(
        init_val_acc
    )


    print()

    print(
        f"[INIT] "
        f"Train loss={init_train_loss:.4f}, "
        f"acc={init_train_acc:.4f} | "
        f"Val loss={init_val_loss:.4f}, "
        f"acc={init_val_acc:.4f}"
    )


    # --------------------------------------------------------
    # Epochs
    # --------------------------------------------------------

    for epoch in range(
        num_epochs
    ):

        epoch_start = time.time()


        print()

        print(
            f"Epoch {epoch + 1}/{num_epochs}"
        )

        print(
            "-" * 60
        )


        model.train()


        running_loss = 0.0
        running_corrects = 0


        for embeddings, labels in dataloaders[
            "train"
        ]:

            embeddings = embeddings.to(
                device
            )


            labels = labels.to(
                device
            )


            optimizer.zero_grad()


            outputs = model(
                embeddings
            )


            loss = criterion(
                outputs,
                labels
            )

            # labels_one_hot = F.one_hot(
            #     labels,
            #     num_classes=2
            # ).float()

            # loss = criterion(
            #     outputs,
            #     labels_one_hot
            # )

            _, preds = torch.max(
                outputs,
                1
            )


            loss.backward()

            optimizer.step()


            batch_size = embeddings.size(
                0
            )


            # running_loss += (
            #     loss.item()
            #     * batch_size
            # )

            running_loss += loss.item()


            running_corrects += (
                preds == labels
            ).sum().item()


        if scheduler is not None:

            scheduler.step()


        epoch_loss = (
            running_loss
            /
            dataset_sizes["train"]
        )


        epoch_acc = (
            running_corrects
            /
            dataset_sizes["train"]
        )


        train_losses.append(
            epoch_loss
        )

        train_accuracies.append(
            epoch_acc
        )


        print(
            f"[Train] "
            f"Loss={epoch_loss:.4f} "
            f"Acc={epoch_acc:.4f}"
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_acc = evaluate(
            model,
            dataloaders["val"],
            criterion,
            device,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())

            print(
                f"*** NEW BEST MODEL: "
                f"epoch={best_epoch}, "
                f"val_acc={best_val_acc:.4f}"
            )


        val_losses.append(
            val_loss
        )

        val_accuracies.append(
            val_acc
        )


        print(
            f"[Val] "
            f"Loss={val_loss:.4f} "
            f"Acc={val_acc:.4f}"
        )


        # ----------------------------------------------------
        # Recalculate train
        # ----------------------------------------------------

        true_train_loss, true_train_acc = evaluate(
            model,
            dataloaders["train"],
            criterion,
            device,
        )


        reeval_train_losses.append(
            true_train_loss
        )

        reeval_train_accuracies.append(
            true_train_acc
        )


        print(
            f"[RECALC] "
            f"Train Loss={true_train_loss:.4f}, "
            f"Acc={true_train_acc:.4f}"
        )


        print(
            f"[Epoch {epoch + 1}] "
            f"Total time: "
            f"{time.time() - epoch_start:.4f}s"
        )


    elapsed = time.time() - since


    print()

    print(
        f"Training completed in "
        f"{elapsed:.2f}s"
    )


    # ========================================================
    # PLOTS
    # ========================================================

    if plots:

        plt.figure(
            figsize=(10, 4)
        )


        plt.subplot(
            1,
            2,
            1
        )


        plt.plot(
            reeval_train_losses,
            label="Train Loss"
        )


        plt.plot(
            val_losses,
            label="Val Loss"
        )


        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Loss"
        )

        plt.title(
            "Loss over Epochs"
        )

        plt.legend()


        plt.subplot(
            1,
            2,
            2
        )


        plt.plot(
            reeval_train_accuracies,
            label="Train Accuracy"
        )


        plt.plot(
            val_accuracies,
            label="Val Accuracy"
        )


        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Accuracy"
        )

        plt.title(
            "Accuracy over Epochs"
        )

        plt.legend()


        plt.tight_layout()


        plt.savefig(
            "training_progress_arcface_resnet50_M.png",
            dpi=200,
        )


        plt.close()

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    print()
    print("=" * 70)
    print(
        f"BEST VALIDATION ACCURACY: "
        f"{best_val_acc * 100:.2f}% "
        f"(epoch {best_epoch})"
    )
    print("=" * 70)

    return model


# ============================================================
# FAST TRAINING FOR SELF-TRAINING
#
# Same function name as previous versions.
# ============================================================

def train_model_fast_for_self_training(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=1,
):

    since = time.time()


    for epoch in range(
        num_epochs
    ):

        epoch_start = time.time()


        model.train()


        running_loss = 0.0
        running_corrects = 0


        for embeddings, labels in dataloaders[
            "train"
        ]:

            embeddings = embeddings.to(
                device
            )


            labels = labels.to(
                device
            )


            optimizer.zero_grad()


            outputs = model(
                embeddings
            )


            # loss = criterion(
            #     outputs,
            #     labels
            # )

            labels_one_hot = F.one_hot(
                labels,
                num_classes=2
            ).float()

            loss = criterion(
                outputs,
                labels_one_hot
            )


            _, preds = torch.max(
                outputs,
                1
            )


            loss.backward()

            optimizer.step()


            batch_size = embeddings.size(
                0
            )


            # running_loss += (
            #     loss.item()
            #     * batch_size
            # )

            running_loss += loss.item()


            running_corrects += (
                preds == labels
            ).sum().item()


        if scheduler is not None:

            scheduler.step()


        epoch_loss = (
            running_loss
            /
            dataset_sizes["train"]
        )


        epoch_acc = (
            running_corrects
            /
            dataset_sizes["train"]
        )


        print(
            f"[Train fast] "
            f"Loss: {epoch_loss:.4f} "
            f"Acc: {epoch_acc:.4f} | "
            f"Total: "
            f"{time.time() - epoch_start:.4f}s"
        )


    print(
        f"Fast training completed in "
        f"{time.time() - since:.4f}s"
    )


    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        f"Using device: {device}"
    )


    # --------------------------------------------------------
    # Load A/B initial supervised dataset.
    #
    # IMPORTANT:
    #
    # A/B folders were created using PCA.
    #
    # But the model itself receives ONLY ArcFace ResNet50 embeddings.
    # --------------------------------------------------------

    (
        dataloaders,
        dataset_sizes,
        class_names,
    ) = get_dataloaders(
        data_dir="split_data",
        batch_size=50,
    )


    # --------------------------------------------------------
    # Create MLP
    # --------------------------------------------------------

    (
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
    ) = create_model_and_optim()


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model_ft = train_model(
        model_ft,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=100,
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    MODEL_PATH = (
        "model_ft_0_ARCFACE_RESNET50_C.pth"
    )


    torch.save(
        model_ft.state_dict(),
        MODEL_PATH,
    )


    print()

    print(
        "Saved model:",
        MODEL_PATH
    )


    # --------------------------------------------------------
    # FINAL CHECK
    # --------------------------------------------------------

    print()
    print(
        "Checking saved model..."
    )
    print()


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


    print(
        f"TRAIN accuracy = "
        f"{train_acc * 100:.2f}%"
    )


    print(
        f"VAL accuracy   = "
        f"{val_acc * 100:.2f}%"
    )


    print()
    print("=" * 70)

    print(
        "Model input: 512D ArcFace ResNet50 embedding"
    )

    print(
        "Classifier: 512 -> 256 -> ReLU -> "
        "Dropout -> 128 -> ReLU -> 2"
    )

    print("=" * 70)