import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms


# ============================================================
# Configuration
# ============================================================

PCA_CSV = "pca_top2_filtered_female.csv"
IMAGE_DIR = "female_faces"
MODEL_PATH = "model_ft_0_BCE_no_bias.pth"

OUTPUT_DIR = "output_feature_trajectory"

NUM_ITERATIONS = 720
ROTATION_DEGS = 0.5
TARGET_RADIUS = 0.45

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Image preprocessing
# ============================================================

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# Load PCA data
# ============================================================

def load_pca_data(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["filename", "PC1", "PC2"]

    filenames = df["filename"].to_numpy()

    points = df[["PC1", "PC2"]].to_numpy(
        dtype=np.float32
    )

    return df, filenames, points


# ============================================================
# Load frozen ShuffleNet feature extractor
# ============================================================

def load_feature_extractor(model_path):
    model = models.shufflenet_v2_x0_5(weights=None)

    feature_dim = model.fc.in_features

    # The saved checkpoint contains a one-output classifier.
    model.fc = nn.Linear(
        feature_dim,
        1,
        bias=False,
    )

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
    )

    model.load_state_dict(state_dict)

    # Remove the classifier. The model will now return its
    # visual feature vector.
    model.fc = nn.Identity()

    model = model.to(DEVICE)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, feature_dim


# ============================================================
# Geometry
# ============================================================

def rotate_target_point(angle_deg, radius):
    angle_rad = np.deg2rad(angle_deg)

    return np.array(
        [
            radius * np.cos(angle_rad),
            radius * np.sin(angle_rad),
        ],
        dtype=np.float32,
    )


def select_nearest_image(
    target_point,
    all_points,
    all_filenames,
):
    distances = np.linalg.norm(
        all_points - target_point,
        axis=1,
    )

    index = int(np.argmin(distances))

    return {
        "index": index,
        "filename": all_filenames[index],
        "point": all_points[index].copy(),
        "selection_distance": float(distances[index]),
    }


def circular_angle_difference(angle_a, angle_b):
    return (
        angle_a - angle_b + 180.0
    ) % 360.0 - 180.0


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_image_feature(model, filename):
    image_path = os.path.join(
        IMAGE_DIR,
        filename,
    )

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")
    image_tensor = IMAGE_TRANSFORM(image)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    feature = model(image_tensor)

    feature = (
        feature.squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return feature


# ============================================================
# Main experiment
# ============================================================

def run_feature_trajectory_experiment():
    _, filenames, pca_points = load_pca_data(PCA_CSV)

    feature_extractor, feature_dim = (
        load_feature_extractor(MODEL_PATH)
    )

    print(f"Using device: {DEVICE}")
    print(f"Feature dimension: {feature_dim}")

    feature_vectors = []
    records = []

    previous_filename = None
    feature_cache = {}

    for iteration in range(NUM_ITERATIONS):
        target_angle = (
            iteration * ROTATION_DEGS
        ) % 360.0

        target_point = rotate_target_point(
            angle_deg=target_angle,
            radius=TARGET_RADIUS,
        )

        selected = select_nearest_image(
            target_point=target_point,
            all_points=pca_points,
            all_filenames=filenames,
        )

        filename = selected["filename"]

        # Avoid extracting the same image repeatedly.
        if filename not in feature_cache:
            feature_cache[filename] = extract_image_feature(
                feature_extractor,
                filename,
            )

        feature = feature_cache[filename].copy()

        repeated_from_previous = (
            previous_filename == filename
            if previous_filename is not None
            else False
        )

        previous_filename = filename

        point = selected["point"]

        actual_image_angle = (
            np.degrees(
                np.arctan2(
                    point[1],
                    point[0],
                )
            )
            % 360.0
        )

        records.append(
            {
                "iteration": iteration,
                "target_angle_deg": target_angle,
                "image_angle_deg": actual_image_angle,
                "angular_selection_error_deg": (
                    circular_angle_difference(
                        actual_image_angle,
                        target_angle,
                    )
                ),
                "filename": filename,
                "repeated_from_previous": (
                    repeated_from_previous
                ),
                "selection_distance": (
                    selected["selection_distance"]
                ),
                "PC1": float(point[0]),
                "PC2": float(point[1]),
            }
        )

        feature_vectors.append(feature)

        if iteration % 20 == 0:
            print(
                f"Iteration {iteration:4d} | "
                f"target={target_angle:7.2f}° | "
                f"image={actual_image_angle:7.2f}° | "
                f"file={filename}"
            )

    feature_matrix = np.stack(
        feature_vectors,
        axis=0,
    )

    records_df = pd.DataFrame(records)

    analyze_feature_trajectory(
        feature_matrix=feature_matrix,
        records_df=records_df,
    )


# ============================================================
# Analysis
# ============================================================

def analyze_feature_trajectory(
    feature_matrix,
    records_df,
):
    # --------------------------------------------------------
    # Center features
    # --------------------------------------------------------

    feature_mean = feature_matrix.mean(
        axis=0,
        keepdims=True,
    )

    centered_features = feature_matrix - feature_mean

    feature_norms = np.linalg.norm(
        centered_features,
        axis=1,
        keepdims=True,
    )

    normalized_features = (
        centered_features
        / np.maximum(feature_norms, 1e-8)
    )

    # --------------------------------------------------------
    # Distances between consecutive feature vectors
    # --------------------------------------------------------

    euclidean_step_distances = np.zeros(
        len(feature_matrix),
        dtype=np.float32,
    )

    cosine_step_similarities = np.ones(
        len(feature_matrix),
        dtype=np.float32,
    )

    for i in range(1, len(feature_matrix)):
        euclidean_step_distances[i] = np.linalg.norm(
            normalized_features[i]
            - normalized_features[i - 1]
        )

        cosine_step_similarities[i] = float(
            np.dot(
                normalized_features[i],
                normalized_features[i - 1],
            )
        )

    records_df["feature_step_distance"] = (
        euclidean_step_distances
    )

    records_df["consecutive_cosine_similarity"] = (
        cosine_step_similarities
    )

    # --------------------------------------------------------
    # Compare each feature to the initial feature
    # --------------------------------------------------------

    initial_feature = normalized_features[0]

    similarity_to_initial = (
        normalized_features @ initial_feature
    )

    records_df["cosine_similarity_to_initial"] = (
        similarity_to_initial
    )

    # --------------------------------------------------------
    # Reduce feature trajectory to 2D for visualization
    # --------------------------------------------------------

    trajectory_pca = PCA(n_components=2)

    feature_trajectory_2d = (
        trajectory_pca.fit_transform(centered_features)
    )

    records_df["feature_trajectory_PC1"] = (
        feature_trajectory_2d[:, 0]
    )

    records_df["feature_trajectory_PC2"] = (
        feature_trajectory_2d[:, 1]
    )

    explained_variance = (
        trajectory_pca.explained_variance_ratio_
    )

    records_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_trajectory.csv",
        ),
        index=False,
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "feature_vectors.npy",
        ),
        feature_matrix,
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "feature_mean.npy",
        ),
        feature_mean,
    )

    print(
        "\nFeature-trajectory PCA explained variance:",
        explained_variance,
    )

    print(
        "Mean consecutive cosine similarity:",
        cosine_step_similarities[1:].mean(),
    )

    print(
        "Minimum consecutive cosine similarity:",
        cosine_step_similarities[1:].min(),
    )

    print(
        "Mean consecutive feature distance:",
        euclidean_step_distances[1:].mean(),
    )

    create_plots(
        records_df=records_df,
        explained_variance=explained_variance,
    )


# ============================================================
# Plots
# ============================================================

def create_plots(records_df, explained_variance):
    iterations = records_df["iteration"].to_numpy()
    target_angles = records_df[
        "target_angle_deg"
    ].to_numpy()

    trajectory_x = records_df[
        "feature_trajectory_PC1"
    ].to_numpy()

    trajectory_y = records_df[
        "feature_trajectory_PC2"
    ].to_numpy()

    # --------------------------------------------------------
    # 1. Feature trajectory in reduced 2D space
    # --------------------------------------------------------

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        trajectory_x,
        trajectory_y,
        c=target_angles,
        s=18,
    )

    plt.plot(
        trajectory_x,
        trajectory_y,
        linewidth=0.8,
        alpha=0.5,
    )

    plt.scatter(
        trajectory_x[0],
        trajectory_y[0],
        s=120,
        marker="o",
        label="Start",
    )

    plt.scatter(
        trajectory_x[-1],
        trajectory_y[-1],
        s=120,
        marker="x",
        label="End",
    )

    plt.colorbar(
        scatter,
        label="Target angle in FaceNet PCA",
    )

    plt.xlabel(
        f"Feature trajectory PC1 "
        f"({explained_variance[0] * 100:.1f}%)"
    )

    plt.ylabel(
        f"Feature trajectory PC2 "
        f"({explained_variance[1] * 100:.1f}%)"
    )

    plt.title(
        "Trajectory of Selected Images in ShuffleNet Feature Space"
    )

    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_trajectory_2d.png",
        ),
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # 2. Consecutive cosine similarity
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        iterations,
        records_df[
            "consecutive_cosine_similarity"
        ],
    )

    plt.xlabel("Iteration")
    plt.ylabel(
        "Cosine similarity to previous feature"
    )

    plt.title(
        "Similarity Between Consecutive Selected Images"
    )

    plt.ylim(-1.0, 1.05)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "consecutive_cosine_similarity.png",
        ),
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # 3. Consecutive feature distance
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        iterations,
        records_df["feature_step_distance"],
    )

    plt.xlabel("Iteration")
    plt.ylabel(
        "Distance from previous normalized feature"
    )

    plt.title(
        "Step Size in ShuffleNet Feature Space"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_step_distance.png",
        ),
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # 4. Similarity to the initial feature
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        target_angles,
        records_df[
            "cosine_similarity_to_initial"
        ],
    )

    plt.xlabel("Rotating target angle")
    plt.ylabel(
        "Cosine similarity to first feature"
    )

    plt.title(
        "Feature Similarity to the Initial Image"
    )

    plt.xlim(
        target_angles.min(),
        target_angles.max(),
    )

    plt.ylim(-1.0, 1.05)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "similarity_to_initial.png",
        ),
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # 5. Selected image angle versus target angle
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        iterations,
        records_df["target_angle_deg"],
        label="Target angle",
    )

    plt.plot(
        iterations,
        records_df["image_angle_deg"],
        label="Selected image angle",
        alpha=0.8,
    )

    plt.xlabel("Iteration")
    plt.ylabel("Angle")
    plt.title(
        "Rotating Target and Selected PCA Image"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "selected_image_angles.png",
        ),
        dpi=300,
    )

    plt.close()


if __name__ == "__main__":
    run_feature_trajectory_experiment()