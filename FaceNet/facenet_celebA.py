#################################################################
# Representation of the face in 128 dimensions vector using FaceNet
# This code uses the MTCNN model to detect faces in images and the InceptionResnetV1 model to extract 128D embeddings.
# The code processes images from the CelebA dataset, extracts face embeddings, and saves them to a CSV file.
# 04/05/2025
#################################################################
import os
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize the face detector (MTCNN) and the pre-trained FaceNet model (InceptionResnetV1)
mtcnn = MTCNN(image_size=160, margin=0)
model = InceptionResnetV1(pretrained="casia-webface", classify=False).eval()

# Directory containing face images (e.g., CelebA dataset)
script_dir = os.path.dirname(__file__)
img_dir = os.path.join(script_dir, "celebA_7")
image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]

print("Looking inside:", img_dir)
print("Found", len(image_files), "image files")
print("Example filenames:", image_files[:5])

# List to store [filename, embedding_vector...]
results = []
failed = 0
# Iterate over each image file
for fname in tqdm(image_files):
    try:
        # Load image and convert to RGB (just in case)
        img_path = os.path.join(img_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {fname}: {e}")
            continue

        # Detect and align the face
        face = mtcnn(img)
        if face is None:
            print(f"No face detected in {fname}")
            failed += 1
            continue
        if face is not None:
            # Add batch dimension: from [3, 160, 160] to [1, 3, 160, 160]
            face = face.unsqueeze(0)

            # Forward pass through FaceNet to get 128D embedding
            with torch.no_grad():
                emb = model(face).squeeze().numpy()
                print(f"Embedding shape for {fname}: {emb.shape}")

            # Save filename and embedding vector
            results.append([fname] + emb.tolist())

    except Exception as e:
        print(f"Error processing {fname}: {e}")

print("Faces detected:", len(results))
print("Faces not detected:", failed)

# Convert to DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv("1_celeba_facenet_embeddings.csv", index=False, header=False)
