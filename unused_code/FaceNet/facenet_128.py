import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize MTCNN (face detector) and FaceNet (resnet) model
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained="vggface2", classify=False).eval()


# Function to extract embedding
def get_embedding(image_path):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to open image: {e}")
        return None

    face = mtcnn(img)
    if face is None:
        print(f"⚠️ No face detected in image: {image_path}")
        return None

    face = face.unsqueeze(0)  # Add batch dimension

    try:
        with torch.no_grad():
            embedding = resnet(face).squeeze()
        return embedding.numpy()
    except RuntimeError as e:
        print(f"❌ Runtime error during embedding: {e}")
        return None


# Example usage
script_dir = os.path.dirname(__file__)
image_path = os.path.join(script_dir, "celebA_1000", "161979.jpg")
embedding_vector = get_embedding(image_path)

if embedding_vector is not None:
    print(f"✅ Embedding vector created, shape: {embedding_vector.shape}")
    print("First 5 values:", embedding_vector[:5])
