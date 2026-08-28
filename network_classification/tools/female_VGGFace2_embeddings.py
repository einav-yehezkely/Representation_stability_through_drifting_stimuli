#################################################################
# FAST VGGFace2 EMBEDDINGS EXTRACTION
#
# InceptionResnetV1 pretrained on VGGFace2
# Batch processing + periodic checkpoints
# 26/8/26
#################################################################

import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32
NUM_WORKERS = 0

CHECKPOINT_EVERY = 5000

OUTPUT_CSV = "female_vggface2_embeddings.csv"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# MODELS
# ============================================================

mtcnn = MTCNN(
    image_size=160,
    margin=0,
    device=device,
)

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=False
).eval().to(device)


# ============================================================
# IMAGE DIRECTORY
# ============================================================

script_dir = os.path.dirname(__file__)

img_dir = os.path.abspath(
    os.path.join(
        script_dir,
        "..",
        "..",
        "female_faces"
    )
)

image_files = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

image_files.sort()

print("Looking inside:", img_dir)
print("Found", len(image_files), "image files")
print("Example filenames:", image_files[:5])


# ============================================================
# DATASET
# ============================================================

class FaceImageDataset(Dataset):

    def __init__(
        self,
        image_dir,
        filenames,
    ):

        self.image_dir = image_dir
        self.filenames = filenames


    def __len__(self):

        return len(
            self.filenames
        )


    def __getitem__(
        self,
        idx,
    ):

        fname = self.filenames[idx]

        img_path = os.path.join(
            self.image_dir,
            fname
        )

        try:

            img = Image.open(
                img_path
            ).convert("RGB")

            return fname, img

        except Exception as e:

            print(
                f"Failed to open {fname}: {e}"
            )

            return fname, None


# ============================================================
# CUSTOM COLLATE
#
# PIL images cannot be stacked automatically.
# ============================================================

def collate_fn(batch):

    filenames = []
    images = []

    for fname, img in batch:

        if img is not None:

            filenames.append(
                fname
            )

            images.append(
                img
            )

    return filenames, images


# ============================================================
# DATA LOADER
# ============================================================

dataset = FaceImageDataset(
    img_dir,
    image_files,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
)


# ============================================================
# RESUME SUPPORT
# ============================================================

results = []

already_done = set()


if os.path.exists(
    OUTPUT_CSV
):

    print()
    print(
        "Existing output file found."
    )

    old_df = pd.read_csv(
        OUTPUT_CSV,
        header=None,
    )

    results = old_df.values.tolist()

    already_done = set(
        old_df.iloc[:, 0]
        .astype(str)
        .tolist()
    )

    print(
        "Already processed:",
        len(already_done)
    )


# ============================================================
# EXTRACTION
# ============================================================

processed_since_save = 0
failed = 0


progress_bar = tqdm(
    total=len(image_files),
    desc="Extracting embeddings"
)


# Account for already completed images
progress_bar.update(
    len(already_done)
)


for filenames, images in loader:

    # --------------------------------------------------------
    # Skip already processed files
    # --------------------------------------------------------

    filtered_filenames = []
    filtered_images = []

    for fname, img in zip(
        filenames,
        images
    ):

        if fname in already_done:
            continue

        filtered_filenames.append(
            fname
        )

        filtered_images.append(
            img
        )


    if len(filtered_images) == 0:

        continue


    try:

        # ====================================================
        # MTCNN BATCH
        #
        # Returns:
        # list / tensor of aligned 160x160 faces
        # ====================================================

        faces = mtcnn(
            filtered_images
        )


        if faces is None:

            failed += len(
                filtered_images
            )

            progress_bar.update(
                len(filtered_images)
            )

            continue


        # ----------------------------------------------------
        # Sometimes MTCNN returns list-like output.
        # Convert valid faces into a batch.
        # ----------------------------------------------------

        valid_faces = []
        valid_filenames = []


        if isinstance(
            faces,
            torch.Tensor
        ):

            # Normal batch case
            if faces.ndim == 4:

                valid_faces = faces

                valid_filenames = filtered_filenames


            # Single image edge case
            elif faces.ndim == 3:

                valid_faces = faces.unsqueeze(
                    0
                )

                valid_filenames = [
                    filtered_filenames[0]
                ]


        else:

            # Handle list output
            temp_faces = []

            for fname, face in zip(
                filtered_filenames,
                faces
            ):

                if face is None:

                    failed += 1

                    continue

                temp_faces.append(
                    face
                )

                valid_filenames.append(
                    fname
                )


            if len(temp_faces) > 0:

                valid_faces = torch.stack(
                    temp_faces
                )


        # ====================================================
        # NO VALID FACES
        # ====================================================

        if len(valid_filenames) == 0:

            progress_bar.update(
                len(filtered_images)
            )

            continue


        # ====================================================
        # MOVE TO DEVICE
        # ====================================================

        valid_faces = valid_faces.to(
            device
        )


        # ====================================================
        # EMBEDDING BATCH
        # ====================================================

        with torch.no_grad():

            embeddings = model(
                valid_faces
            )


        embeddings = (
            embeddings
            .cpu()
            .numpy()
        )


        # ====================================================
        # STORE
        # ====================================================

        for fname, emb in zip(
            valid_filenames,
            embeddings
        ):

            results.append(
                [fname] + emb.tolist()
            )

            already_done.add(
                fname
            )


        processed_since_save += len(
            valid_filenames
        )


        progress_bar.update(
            len(filtered_images)
        )


        # ====================================================
        # CHECKPOINT
        # ====================================================

        if (
            processed_since_save
            >= CHECKPOINT_EVERY
        ):

            df = pd.DataFrame(
                results
            )

            df.to_csv(
                OUTPUT_CSV,
                index=False,
                header=False
            )

            print()
            print(
                f"Checkpoint saved: "
                f"{len(results)} embeddings"
            )

            processed_since_save = 0


    except Exception as e:

        print()
        print(
            "Batch error:",
            e
        )

        # ----------------------------------------------------
        # FALLBACK:
        # process problematic batch one image at a time
        # ----------------------------------------------------

        for fname, img in zip(
            filtered_filenames,
            filtered_images
        ):

            try:

                face = mtcnn(
                    img
                )


                if face is None:

                    failed += 1
                    continue


                face = face.unsqueeze(
                    0
                ).to(
                    device
                )


                with torch.no_grad():

                    emb = model(
                        face
                    ).squeeze(
                        0
                    )


                emb = (
                    emb
                    .cpu()
                    .numpy()
                )


                results.append(
                    [fname] + emb.tolist()
                )

                already_done.add(
                    fname
                )


            except Exception as single_error:

                print(
                    f"Error processing "
                    f"{fname}: "
                    f"{single_error}"
                )

                failed += 1


        progress_bar.update(
            len(filtered_images)
        )


progress_bar.close()


# ============================================================
# FINAL SAVE
# ============================================================

df = pd.DataFrame(
    results
)

df.to_csv(
    OUTPUT_CSV,
    index=False,
    header=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)

print(
    "Embeddings saved:",
    len(results)
)

print(
    "Failed:",
    failed
)

print(
    "Output:",
    OUTPUT_CSV
)

print("=" * 70)