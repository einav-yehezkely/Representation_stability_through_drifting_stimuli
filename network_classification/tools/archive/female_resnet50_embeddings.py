#################################################################
# Representation of faces using ArcFace ResNet50
# The model is pretrained for face recognition on WebFace600K.
# Each image is processed and its 512D embedding is saved to CSV.
# 24/8/26
#################################################################

import os
import cv2
import pandas as pd
from tqdm import tqdm
from insightface.app import FaceAnalysis


# ---------------------------------------------------------
# Initialize ArcFace / ResNet50 face-recognition model
# buffalo_l contains:
#   - face detector
#   - face alignment
#   - ResNet50 recognition model\
# Load the InsightFace buffalo_l model pack.
# Its face-recognition model is w600k_r50:
# a ResNet50 trained for face recognition on WebFace600K.
# We use this ResNet50 to extract 512D face embeddings.
# ---------------------------------------------------------
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

# ---------------------------------------------------------
# Directory containing female face images
# ---------------------------------------------------------
# script_dir = os.path.dirname(__file__)

# img_dir = os.path.abspath(
#     os.path.join(
#         script_dir,
#         "..",
#         "..",
#         "female_faces"
#     )
# )
script_dir = os.path.dirname(__file__)

img_dir = os.path.join(
    script_dir,
    "female_faces"
)

image_files = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Looking inside:", img_dir)
print("Found", len(image_files), "image files")
print("Example filenames:", image_files[:5])

# ---------------------------------------------------------
# Extract embeddings
# ---------------------------------------------------------

results = []
failed = 0

for fname in tqdm(image_files):

    try:

        img_path = os.path.join(img_dir, fname)

        # Open image
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to open {fname}")
            failed += 1
            continue

        # Detect faces
        faces = app.get(img)

        if len(faces) == 0:
            print(f"No face detected in {fname}")
            failed += 1
            continue

        # -------------------------------------------------
        # If more than one face is detected,
        # use the largest face
        # -------------------------------------------------

        face = max(
            faces,
            key=lambda f:
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
        )

        # ArcFace / ResNet50 embedding
        emb = face.embedding

        print(
            f"Embedding shape for {fname}:",
            emb.shape
        )

        # Save filename + embedding
        results.append(
            [fname] + emb.tolist()
        )

    except Exception as e:

        print(
            f"Error processing {fname}: {e}"
        )

        failed += 1


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("Faces detected:", len(results))
print("Faces not detected:", failed)

# ---------------------------------------------------------
# Save embeddings to CSV
# ---------------------------------------------------------

df = pd.DataFrame(results)

df.to_csv(
    "female_arcface_embeddings.csv",
    index=False,
    header=False
)

print(
    "Saved embeddings to:",
    "female_arcface_embeddings.csv"
)