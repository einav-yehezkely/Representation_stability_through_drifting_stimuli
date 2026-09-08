#################################################################
# Representation of aligned face images using FaceNet
#
# InceptionResnetV1 pretrained on VGGFace2.
# Input images are already cropped and aligned.
# Each image is represented by a 512D embedding.
# 05/09/2026
#################################################################

import os

import torch
import pandas as pd

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from facenet_pytorch import InceptionResnetV1


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32
NUM_WORKERS = 0

OUTPUT_CSV = "female_facenet_vggface2_embeddings.csv"


script_dir = os.path.dirname(
    os.path.abspath(__file__)
)

img_dir = os.path.abspath(
    os.path.join(
        script_dir,
        "..",
        "..",
        "female_faces"
    )
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
#
# FaceNet / facenet-pytorch expects roughly [-1, 1] input.
# ============================================================

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])


# ============================================================
# LOAD FACENET
# ============================================================

print()
print("Loading InceptionResnetV1 pretrained on VGGFace2...")


model = InceptionResnetV1(
    pretrained="vggface2",
    classify=False
)

model = model.eval().to(device)


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
print("Looking inside:", img_dir)
print("Found", len(image_files), "image files")
print("Example filenames:", image_files[:5])


# ============================================================
# RESUME
#
# If the output CSV already exists, skip images that were
# already processed.
# ============================================================

already_processed = set()


if os.path.exists(OUTPUT_CSV):

    print()
    print("Existing output CSV found.")

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


        img = self.transform(
            img
        )


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
print("STARTING FACENET EMBEDDING EXTRACTION")
print("=" * 60)

print("Batch size:", BATCH_SIZE)
print("Number of batches:", len(loader))

print()


total_processed = 0


with torch.inference_mode():

    for filenames, faces in tqdm(
        loader,
        desc="Extracting FaceNet VGGFace2 embeddings"
    ):

        # ----------------------------------------------------
        # SEND BATCH TO DEVICE
        #
        # [B, 3, 160, 160]
        # ----------------------------------------------------

        faces = faces.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # FACENET FORWARD PASS
        #
        # Expected:
        # [B, 512]
        # ----------------------------------------------------

        embeddings = model(
            faces
        )


        # ----------------------------------------------------
        # CHECK FIRST BATCH
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
        # SAVE BATCH
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