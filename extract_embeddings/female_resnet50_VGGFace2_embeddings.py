#################################################################
# Representation of faces using ResNet50 trained on VGGFace2
#
# ResNet50 trained FROM SCRATCH on VGGFace2.
# Each image is resized to 224x224 and represented
# by a 2048D feature vector.
#################################################################

import os
import pickle

import torch
import pandas as pd

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import resnet as ResNet

import numpy as np


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32
NUM_WORKERS = 0

OUTPUT_CSV = "female_resnet50_vggface2_embeddings.csv"

script_dir = os.path.dirname(
    os.path.abspath(__file__)
)

img_dir = os.path.abspath(
    os.path.join(
        script_dir,
        "..",
        "female_faces"
    )
)

WEIGHTS_PATH = os.path.join(
    script_dir,
    "resnet50_scratch_weight.pkl"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# PREPROCESSING
# ============================================================

# ============================================================
# PREPROCESSING
# Exact preprocessing used by VGGFace2-pytorch
# ============================================================

MEAN_BGR = torch.tensor(
    [91.4953, 103.8827, 131.0912],
    dtype=torch.float32
).view(3, 1, 1)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])


# ============================================================
# LOAD RESNET50 VGGFACE2
# ============================================================

print()
print("Loading ResNet50...")
print("Weights:")
print(WEIGHTS_PATH)

model = ResNet.resnet50(
    num_classes=8631,
    include_top=False
)


with open(
    WEIGHTS_PATH,
    "rb"
) as f:

    raw_weights = pickle.loads(
        f.read(),
        encoding="latin1"
    )


weights = {
    key: torch.from_numpy(value)
    for key, value in raw_weights.items()
}


model.load_state_dict(
    weights
)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ============================================================
# FIND IMAGES
# ============================================================

image_files = sorted([
    f
    for f in os.listdir(img_dir)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
])


print()
print("Image directory:")
print(img_dir)

print()
print(
    "Images found:",
    len(image_files)
)

print(
    "First images:",
    image_files[:5]
)


# ============================================================
# RESUME FROM EXISTING CSV
# ============================================================

already_processed = set()


if os.path.exists(OUTPUT_CSV):

    print()
    print("Existing output file found.")

    try:

        existing_df = pd.read_csv(
            OUTPUT_CSV,
            header=None,
            usecols=[0]
        )

        already_processed = set(
            existing_df.iloc[:, 0].astype(str)
        )

        print(
            "Already processed:",
            len(already_processed)
        )

    except Exception as e:

        print(
            "Could not read existing CSV:",
            e
        )


image_files = [
    f
    for f in image_files
    if f not in already_processed
]


print(
    "Images remaining:",
    len(image_files)
)


# ============================================================
# DATASET
# ============================================================

class FaceDataset(Dataset):

    def __init__(
        self,
        image_dir,
        filenames,
        transform
    ):

        self.image_dir = image_dir
        self.filenames = filenames
        self.transform = transform


    def __len__(self):

        return len(
            self.filenames
        )


    def __getitem__(
        self,
        idx
    ):

        fname = self.filenames[idx]

        img_path = os.path.join(
            self.image_dir,
            fname
        )


        img = Image.open(
            img_path
        ).convert("RGB")


        img = self.transform(img)

        # PIL RGB -> tensor, WITHOUT division by 255
        img = np.array(
            img,
            dtype=np.float32
        )

        # RGB -> BGR
        img = img[:, :, ::-1].copy()

        # HWC -> CHW
        img = torch.from_numpy(
            img
        ).permute(
            2, 0, 1
        )

        # subtract VGGFace2 BGR mean
        img = img - MEAN_BGR


        return fname, img


# ============================================================
# DATALOADER
# ============================================================

dataset = FaceDataset(
    image_dir=img_dir,
    filenames=image_files,
    transform=transform
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# EXTRACTION
# ============================================================

print()
print("=" * 60)
print("STARTING EMBEDDING EXTRACTION")
print("=" * 60)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Number of batches:",
    len(loader)
)

print()


total_processed = 0


with torch.inference_mode():

    for filenames, faces in tqdm(
        loader,
        desc="Extracting VGGFace2 ResNet50 embeddings"
    ):

        # ----------------------------------------------------
        # SEND BATCH TO DEVICE
        # ----------------------------------------------------

        faces = faces.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # RESNET50 FORWARD PASS
        # ----------------------------------------------------

        embeddings = model(
            faces
        )


        # ----------------------------------------------------
        # FLATTEN
        #
        # Expected:
        # [batch_size, 2048]
        # ----------------------------------------------------

        embeddings = embeddings.flatten(
            start_dim=1
        )


        # ----------------------------------------------------
        # DEBUG FIRST BATCH
        # ----------------------------------------------------

        if total_processed == 0:

            print()
            print(
                "First batch input shape:",
                tuple(faces.shape)
            )

            print(
                "First batch embedding shape:",
                tuple(embeddings.shape)
            )

            print()


        # ----------------------------------------------------
        # MOVE TO CPU
        # ----------------------------------------------------

        embeddings = (
            embeddings
            .detach()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # CREATE DATAFRAME FOR THIS BATCH
        # ----------------------------------------------------

        batch_results = []

        for fname, emb in zip(
            filenames,
            embeddings
        ):

            batch_results.append(
                [fname] + emb.tolist()
            )


        batch_df = pd.DataFrame(
            batch_results
        )


        # ----------------------------------------------------
        # SAVE THIS BATCH IMMEDIATELY
        #
        # This means that if the program stops,
        # completed batches are already saved.
        # ----------------------------------------------------

        batch_df.to_csv(
            OUTPUT_CSV,
            mode="a",
            index=False,
            header=False
        )


        total_processed += len(
            filenames
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("FINISHED")
print("=" * 60)

print(
    "New embeddings created:",
    total_processed
)

print(
    "Previously processed:",
    len(already_processed)
)

print(
    "Total:",
    total_processed
    + len(already_processed)
)

print(
    "Saved to:",
    OUTPUT_CSV
)