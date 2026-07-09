import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models


MODEL_PATH = "model_ft_0_MSE_20_percent.pth"
PCA_CSV = "pca_top2_filtered_female_20_percent.csv"
IMAGE_DIR = "female_faces"

BATCH_SIZE = 64
OUTPUT_IMAGE = "pca_predictions_scatter.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path="model_ft_0_MSE_20_percent.pth"):
    model = models.shufflenet_v2_x0_5(weights=None)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
    )

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


def classify_all_images_batched(model, df, batch_size=64):
    predictions = np.full(len(df), -1)

    batch_tensors = []
    batch_indices = []

    with torch.no_grad():
        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Loading and classifying images"
        ):
            image_path = os.path.join(IMAGE_DIR, row["filename"])

            if not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert("RGB")
            x = transform(image)

            batch_tensors.append(x)
            batch_indices.append(idx)

            if len(batch_tensors) == batch_size:
                run_batch(model, batch_tensors, batch_indices, predictions)
                batch_tensors = []
                batch_indices = []

        if len(batch_tensors) > 0:
            run_batch(model, batch_tensors, batch_indices, predictions)

    return predictions


def run_batch(model, batch_tensors, batch_indices, predictions):
    batch = torch.stack(batch_tensors).to(DEVICE)

    outputs = model(batch).squeeze(1)
    probs_b = torch.sigmoid(outputs)

    preds = (probs_b >= 0.5).long().cpu().numpy()

    for idx, pred in zip(batch_indices, preds):
        predictions[idx] = pred


def plot_predictions(df, preds):
    valid = preds != -1

    df = df[valid].copy()
    preds = preds[valid]

    df_A = df[preds == 0]
    df_B = df[preds == 1]

    plt.figure(figsize=(10, 10))

    plt.scatter(
        df_A["x"], df_A["y"],
        color="blue",
        s=10,
        alpha=0.7,
        label="Predicted A",
    )

    plt.scatter(
        df_B["x"], df_B["y"],
        color="red",
        s=10,
        alpha=0.7,
        label="Predicted B",
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Model predictions over PCA space")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_IMAGE, dpi=300)
    plt.show()

    print(f"\nSaved: {OUTPUT_IMAGE}")
    print(f"Predicted A: {len(df_A)}")
    print(f"Predicted B: {len(df_B)}")
    print(f"Missing images skipped: {(preds == -1).sum()}")


def main():
    print(f"Using device: {DEVICE}")

    df = pd.read_csv(
        PCA_CSV,
        header=None,
        names=["filename", "x", "y"]
    )

    print(f"Loaded {len(df)} PCA points")

    model = load_model(MODEL_PATH)

    preds = classify_all_images_batched(
        model,
        df,
        batch_size=BATCH_SIZE,
    )

    plot_predictions(df, preds)

    print("Done.")


if __name__ == "__main__":
    main()