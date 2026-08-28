import os   # for file and directory operations
import torch    # for PyTorch operations
import pandas as pd # for data manipulation and CSV handling
import matplotlib.pyplot as plt # for plotting
import numpy as np  # for numerical operations
import shutil   # for file copying
from sklearn.model_selection import train_test_split    # for splitting datasets - used in split_and_copy_images
import torch.nn as nn   # loss functions and neural network layers
from tqdm import tqdm   # for progress bars
from arcface_embeddings_training import (
    ArcFaceClassifier,
    train_model,
    get_dataloaders_from_lists,
    train_model_fast_for_self_training
)
from matplotlib.patches import Circle   # for drawing circles in plots
import time # for time tracking
import torch.optim as optim # for optimization algorithms - AdamW
from torch.optim import lr_scheduler    # for learning rate scheduling
import sys  # for system-specific parameters and functions

BASE_DIR = "tmp_CE"
os.makedirs(BASE_DIR, exist_ok=True)

OUTPUT_DIR = "output_CE"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD PCA DATA AND COMPUTE ANGLES OF ALL IMAGES
# ============================================================
PCA_DF = pd.read_csv("pca_top2_filtered_female_1.csv", header=None)
PCA_DF.columns = ["filename", "x", "y"]
PCA_DF["angle_deg"] = np.degrees(np.arctan2(PCA_DF["y"], PCA_DF["x"])) % 360
ANGLE_MAP = dict(zip(PCA_DF["filename"], PCA_DF["angle_deg"]))

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# LOAD PRECOMPUTED ARCFACE EMBEDDINGS
# INSTEAD OF COMPUTING THEM AGAIN
# ============================================================
ARCFACE_EMBEDDINGS_CSV = "female_arcface_embeddings.csv"

# Load the ArcFace embeddings CSV into a DataFrame
_arcface_df = pd.read_csv(
    ARCFACE_EMBEDDINGS_CSV,
    header=None,
)

if _arcface_df.shape[1] != 513:
    raise RuntimeError(
        f"Expected 513 columns "
        f"(filename + 512 ArcFace ResNet50 dimensions), "
        f"found {_arcface_df.shape[1]}"
    )

ARCFACE_LOOKUP = {}

for _, row in _arcface_df.iterrows():

    filename = str(row.iloc[0]).strip()
    basename = os.path.basename(filename)

    embedding = row.iloc[1:513].to_numpy(
        dtype=np.float32
    )

    # L2-normalize the ArcFace embedding
    norm = np.linalg.norm(embedding)

    if norm > 0:
        embedding = embedding / norm

    ARCFACE_LOOKUP[filename] = embedding
    ARCFACE_LOOKUP[basename] = embedding


def get_arcface_embedding(filename):

    filename = str(filename).strip()
    basename = os.path.basename(filename)

    if basename not in ARCFACE_LOOKUP:
        raise KeyError(
            f"ArcFace ResNet50 embedding not found for: {filename}"
        )

    return torch.tensor(
        ARCFACE_LOOKUP[basename],
        dtype=torch.float32,
    )

class Tee:
    """ 
    class to duplicate stdout and stderr to a log file. Useful for logging console output to a file while still displaying it in the terminal.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def inside_tmp(*paths):
    """Return a path inside the BASE_DIR (tmp)."""
    return os.path.join(BASE_DIR, *paths)


def inside_output(*paths):
    """return a path inside the OUTPUT_DIR (output)."""
    return os.path.join(OUTPUT_DIR, *paths)

# ===========================================================
# SETUP PLOTS DIRECTORIES AND LOGGING
# ===========================================================
SCATTER_DIR = inside_output("scatter_frames")
LINEAR_DIR = inside_output("linear_frames")

os.makedirs(SCATTER_DIR, exist_ok=True)
os.makedirs(LINEAR_DIR, exist_ok=True)

log_file = open(inside_output("output.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

def load_top2_filtered(csv_path="pca_top2_filtered_female_1.csv"):
    """
    Load 2D PCA coordinates of filtered images from a CSV file.

    Assumes the format: image_name, x, y

    The images in this file have been filtered such that they are not only close in the top 2 PCA components,
    but also have low variance in the remaining PCA dimensions (i.e., their radius in the residual components is small).
    This ensures that proximity in 2D reflects real similarity in the full FaceNet space.
    """
    df = pd.read_csv(csv_path, header=None)
    names = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    y = df.iloc[:, 2].values
    points = np.stack((x, y), axis=1)
    return names, points


def create_base_and_opposite_points(angle, csv_path="pca_top2_filtered_female_1.csv"):
    """
    Given a target angle, find the base point in PCA space that is closest to that angle,
    and compute its opposite point (180 degrees away).
    """
    # Load data
    names, points = load_top2_filtered(csv_path)

    # Compute angles (in radians) of each point from the origin
    angles = np.arctan2(points[:, 1], points[:, 0])

    # Convert angles from radians to degrees, now in range [-180, 180]
    angles_deg = np.degrees(angles)
    # Shift all angles to be in the range [0, 360)
    angles_deg = (angles_deg + 360) % 360

    radii = np.linalg.norm(points, axis=1)

    # Define the target angle in degrees
    target_angle = angle

    target_radius = 0.45

    # Compute a combined error metric that considers both angle and radius differences for finding the closest point to the target angle and radius
    angle_error = np.abs(angles_deg - target_angle)
    radius_error = np.abs(radii - target_radius)
    combined_error = angle_error + radius_error * 100

    # Find the index of the point whose angle is closest to the target angle
    base_idx = np.argmin(combined_error)
    # Retrieve the actual 2D PCA coordinates of the selected base point
    base_point = points[base_idx]

    opposite_point = -base_point

    return base_point, opposite_point


def rotate_vector(v, angle_deg):
    """
    Rotate a 2D vector counter clockwise by angle_deg (in degrees) around the origin.

    Parameters:
    - v: numpy array of shape (2,), the vector to rotate
    - angle_deg: float, the rotation angle in degrees

    Returns:
    - rotated vector (2D numpy array)
    """
    angle_rad = np.deg2rad(angle_deg)
    R = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return R @ v


def collect_nearest_images(
    center_point,
    all_points,
    all_names,
    output_dir,
    k=500,
    image_source_dir="female_faces",
):
    """
    Find the k nearest images to center_point and save their filenames.
    Also saves a CSV file with the selected filenames.

    Parameters:
        center_point: np.array of shape (2,)
        all_points: np.array of shape (N, 2)
        all_names: list or array of N image filenames
        output_dir: path to the folder where results will be saved
        k: number of images to select
        image_source_dir: directory where source images are located
    Returns:
        nearest_indices: indices of the selected images in all_points/all_names
    """

    # Compute distances
    dists = np.linalg.norm(all_points - center_point, axis=1)

    # Use np.argpartition for efficiency, then sort the selected indices
    nearest_indices = np.argpartition(dists, k)[:k]
    nearest_indices = nearest_indices[np.argsort(dists[nearest_indices])]

    selected_names = []

    for idx in nearest_indices:
        name = all_names[idx]
        selected_names.append(name)

    # Save filenames to CSV
    csv_name = f"filenames_{os.path.basename(output_dir)}.csv"
    csv_path = inside_tmp(csv_name)
    pd.DataFrame(selected_names, columns=["filename"]).to_csv(csv_path, index=False)
    print(f"Saved {len(selected_names)} image names to {csv_path}")

    return nearest_indices


def merge_clusters():
    """
    Merge the two CSV files filenames_A.csv and filenames_B.csv into filenames_merged.csv in order to retrain the model on both clusters.
    The merged CSV will contain all filenames from both clusters, shuffled randomly.
    """
    # Load both CSVs
    df_a = pd.read_csv(inside_tmp("filenames_A.csv"))
    df_b = pd.read_csv(inside_tmp("filenames_B.csv"))

    # Concatenate the DataFrames
    df_merged = pd.concat([df_a, df_b], ignore_index=True)

    # Shuffle rows
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to new CSV
    df_merged.to_csv(inside_tmp("filenames_merged.csv"), index=False)


def load_model(
    model_path="model_ft_0_ARCFACE_RESNET50.pth"
):
    model = ArcFaceClassifier()

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def safe_read_filenames(csv_path):
    """
    Safely read the 'filename' column from a CSV file. If the file does not exist, is empty, or does not contain the 'filename' column, return an empty pandas Series.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return pd.Series(dtype="object")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.Series(dtype="object")

    if "filename" not in df.columns:
        return pd.Series(dtype="object")

    return df["filename"]


def classify_images(model, csv_path, clusters=False):
    """
    Classify images listed in a CSV file using the provided model.
    parameters:
        model: The trained PyTorch model for classification.
        csv_path: Path to the CSV file containing image filenames.
        clusters: If True, save additional information about the predictions for clustering analysis.
    returns:
        If clusters is True, returns a list of dictionaries containing filename, predicted probabilities, and predicted class for each image.
        If clusters is False, saves two CSV files: predicted_as_A.csv and predicted_as_B.csv containing the filenames classified as A and B respectively.
    """
    model.eval()

    df = pd.read_csv(csv_path)

    predicted_A = []
    predicted_B = []
    # training_records will store the filename, predicted probabilities, and predicted class for each image if clusters is True
    training_records = []

    # Use torch.no_grad() to disable gradient calculations for efficiency since we are in evaluation mode
    with torch.no_grad():
        for _, row in df.iterrows():

            filename = str(
                row["filename"]
            )

            try:
                embedding = get_arcface_embedding(
                    filename
                )
            except KeyError as e:
                print(
                    f"Warning: {e}. Skipping."
                )
                continue

            # 2D tensor of shape [1, 512] for a single image embedding needed for the model input
            input_tensor = embedding.unsqueeze(
                0
            ).to(device)

            output = model(
                input_tensor
            )

            # Compute softmax probabilities for the two classes (A and B)
            probs = torch.softmax(
                output,
                dim=1
            )

            prob_a = probs[
                0,
                0
            ].item()

            prob_b = probs[
                0,
                1
            ].item()

            # Get the predicted class (0 for A, 1 for B) by taking the argmax of the output logits
            pred = output.argmax(
                dim=1
            ).item()

            if pred == 0:
                predicted_A.append(
                    row
                )
            else:
                predicted_B.append(
                    row
                )

            if clusters:
                training_records.append(
                    {
                        "filename":
                            row["filename"],

                        "prob_A":
                            prob_a,

                        "prob_B":
                            prob_b,

                        "pred":
                            "A"
                            if pred == 0
                            else "B",
                    }
                )
    # base_columns will be used to ensure that the output DataFrames have the same columns as the input CSV, even if some predictions are empty
    base_columns = df.columns.tolist()

    df_A = pd.DataFrame(
        predicted_A
    )

    df_B = pd.DataFrame(
        predicted_B
    )

    if df_A.empty:
        df_A = pd.DataFrame(
            columns=base_columns
        )
    else:
        df_A = df_A.reindex(
            columns=base_columns
        )

    if df_B.empty:
        df_B = pd.DataFrame(
            columns=base_columns
        )
    else:
        df_B = df_B.reindex(
            columns=base_columns
        )

    if clusters:

        df_A.to_csv(
            inside_tmp(
                "cluster_predicted_as_A.csv"
            ),
            index=False
        )

        df_B.to_csv(
            inside_tmp(
                "cluster_predicted_as_B.csv"
            ),
            index=False
        )

        return training_records

    else:

        df_A.to_csv(
            inside_tmp(
                "predicted_as_A.csv"
            ),
            index=False
        )

        df_B.to_csv(
            inside_tmp(
                "predicted_as_B.csv"
            ),
            index=False
        )


def classify_images_batched(
    model,
    csv_path,
    clusters=False,
    batch_size=50,
):
    """
    Classify images listed in a CSV file using the provided model. 
    Batches the classification for efficiency.

    Keyword arguments:
        model -- The trained PyTorch model for classification.
        csv_path -- Path to the CSV file containing image filenames.
        clusters -- If True, save additional information about the predictions for clustering analysis.
        batch_size -- The number of images to process in each batch.

    Return: 
        If clusters is True, returns a list of dictionaries containing filename, predicted probabilities, and predicted class for each image.
        If clusters is False, saves two CSV files: predicted_as_A.csv and predicted_as_B.csv containing the filenames classified as A and B respectively.
    """
    
    model.eval()

    df = pd.read_csv(csv_path)

    predicted_A = []
    predicted_B = []
    training_records = []

    batch_embeddings = []
    batch_rows = []

    with torch.no_grad():

        for _, row in df.iterrows():

            filename = str(
                row["filename"]
            )

            try:
                embedding = get_arcface_embedding(
                    filename
                )
            except KeyError as e:
                print(
                    f"Warning: {e}. Skipping."
                )
                continue

            batch_embeddings.append(
                embedding
            )

            batch_rows.append(
                row
            )
            # Once the batch size is reached, process the current batch of embeddings and rows. This will classify them and update the predicted_A, predicted_B, and training_records lists.
            if len(batch_embeddings) == batch_size:
                process_classification_batch(
                    model,
                    batch_embeddings,
                    batch_rows,
                    predicted_A,
                    predicted_B,
                    training_records,
                    clusters,
                )
                # Reset the batch lists for the next batch of images
                batch_embeddings = []
                batch_rows = []

        # After the loop, if there are any remaining embeddings that didn't fill a complete batch, process them as well.
        if batch_embeddings:

            process_classification_batch(
                model,
                batch_embeddings,
                batch_rows,
                predicted_A,
                predicted_B,
                training_records,
                clusters,
            )


    base_columns = df.columns.tolist()

    df_A = pd.DataFrame(
        predicted_A
    )

    df_B = pd.DataFrame(
        predicted_B
    )


    if df_A.empty:
        df_A = pd.DataFrame(
            columns=base_columns
        )
    else:
        df_A = df_A.reindex(
            columns=base_columns
        )


    if df_B.empty:
        df_B = pd.DataFrame(
            columns=base_columns
        )
    else:
        df_B = df_B.reindex(
            columns=base_columns
        )


    if clusters:

        df_A.to_csv(
            inside_tmp(
                "cluster_predicted_as_A.csv"
            ),
            index=False,
        )

        df_B.to_csv(
            inside_tmp(
                "cluster_predicted_as_B.csv"
            ),
            index=False,
        )

        return training_records

    else:

        df_A.to_csv(
            inside_tmp(
                "predicted_as_A.csv"
            ),
            index=False,
        )

        df_B.to_csv(
            inside_tmp(
                "predicted_as_B.csv"
            ),
            index=False,
        )

def process_classification_batch(
    model,
    batch_embeddings,
    batch_rows,
    predicted_A,
    predicted_B,
    training_records,
    clusters,
):
    """
    Process a batch of image embeddings and classify them.
    """
    
    # Shape:
    # [batch_size, 512]
    batch = torch.stack(
        batch_embeddings
    ).to(device)


    outputs = model(
        batch
    )


    probs = torch.softmax(
        outputs,
        dim=1
    )

    preds = outputs.argmax(
        dim=1
    )


    for row, prob, pred in zip(
        batch_rows,
        probs,
        preds,
    ):

        prob_a = prob[0].item()
        prob_b = prob[1].item()

        pred = pred.item()


        if pred == 0:
            predicted_A.append(
                row
            )
        else:
            predicted_B.append(
                row
            )


        if clusters:

            training_records.append(
                {
                    "filename":
                        row["filename"],

                    "prob_A":
                        prob_a,

                    "prob_B":
                        prob_b,

                    "pred":
                        "A"
                        if pred == 0
                        else "B",
                }
            )

def split_and_copy_images(
    csv_path,
    label,
    image_source_dir="female_faces",
    train_ratio=0.8,
    root_dir=inside_tmp("split_data"),
):
    """
    Split images listed in a CSV file into training and validation sets, and copy them to separate directories.
    """
    # Load the CSV with predicted filenames
    df = pd.read_csv(csv_path)
    filenames = df["filename"].tolist()

    # Split into train and val sets
    train_files, val_files = train_test_split(
        filenames, train_size=train_ratio, random_state=42
    )

    # Define output directories
    for subset, files in [("train", train_files), ("val", val_files)]:
        target_dir = os.path.join(root_dir, subset, label)
        os.makedirs(target_dir, exist_ok=True)

        for name in tqdm(files, desc=f"Copying {subset}/{label}"):
            src = os.path.join(image_source_dir, name)
            dst = os.path.join(target_dir, name)
            try:
                shutil.copy2(src, dst)
            except FileNotFoundError:
                print(f"Warning: {src} not found.")


def generate_rotation_sequence(
    base_point,
    all_points,
    all_names,
    num_steps=180,
    start_angle=0,
    rotation_range=180,
    used_indices=None,
):
    """
    Rotate base_point around the origin in num_steps steps (in degrees)
    and find the closest point from all_points at each step.

    Parameters:
    - base_point: 2D numpy array representing the starting point
    - all_points: numpy array of shape (N, 2), 2D positions of all images
    - all_names: list or array of N image names corresponding to all_points
    - num_steps: number of rotation steps
    - start_angle: starting angle in degrees (default 0)
    - rotation_range: range of rotation in degrees - meaning the total rotation will be from start_angle to start_angle + rotation_range
    - used_indices: set of indices to avoid reusing images across runs

    Returns:
    - List of tuples: (step_index, angle_in_degrees, closest_image_name)
    """
    results = []
    if used_indices is None:
        used_indices = set()

    for i in range(num_steps):
        # Calculate the current angle for this step
        angle_deg = (start_angle + (rotation_range * i / num_steps)) % 360
        rotated = rotate_vector(base_point, angle_deg)
        true_angle = np.degrees(np.arctan2(rotated[1], rotated[0])) % 360
        dists = np.linalg.norm(all_points - rotated, axis=1)

        # Mark already used indices with infinite distance
        for idx in used_indices:
            dists[idx] = np.inf  # Ignore already used indices

        # Find the index of the closest point to the rotated position
        idx_closest = np.argmin(dists)
        used_indices.add(idx_closest)
        results.append((i, true_angle, all_names[idx_closest]))
    return results, used_indices


def create_prediction_scatter(angle, frame_id, save_dir=SCATTER_DIR):
    """
    Create a scatter plot showing model predictions over 2D PCA space.
    Saves the result as an image in the specified folder.

    Args:
        angle (float): The angle used for training the model.
        frame_id (int): Frame number for the filename.
        save_dir (str): Directory to save the image.
    """
    opposite_angle = (angle + 180) % 360
    os.makedirs(save_dir, exist_ok=True)

    # Load PCA data
    df = pd.read_csv("pca_top2_filtered_female_1.csv", header=None)
    df.columns = ["name", "x", "y"]

    # Load predictions
    pred_a = safe_read_filenames(inside_tmp("predicted_as_A.csv"))
    pred_b = safe_read_filenames(inside_tmp("predicted_as_B.csv"))

    predicted_cluster_a = safe_read_filenames(inside_tmp("cluster_predicted_as_A.csv"))
    predicted_cluster_b = safe_read_filenames(inside_tmp("cluster_predicted_as_B.csv"))

    # Filter the DataFrame to only include points that were predicted as A or B
    df_a = df[df["name"].isin(pred_a)]
    df_b = df[df["name"].isin(pred_b)]
    df_predicted_cluster_a = df[df["name"].isin(predicted_cluster_a)]
    df_predicted_cluster_b = df[df["name"].isin(predicted_cluster_b)]

    plt.figure(figsize=(10, 10))
    plt.scatter(df["x"], df["y"], s=5, alpha=0.3, color="gray", label="All Vectors")
    plt.scatter(
        df_predicted_cluster_a["x"],
        df_predicted_cluster_a["y"],
        s=9,
        alpha=0.8,
        color="lightblue",
        label="Trained A - predicted",
    )
    plt.scatter(
        df_predicted_cluster_b["x"],
        df_predicted_cluster_b["y"],
        s=9,
        alpha=0.7,
        color="pink",
        label="Trained B - predicted",
    )
    plt.scatter(
        df_a["x"], df_a["y"], s=10, alpha=0.7, color="blue", label="Predicted A"
    )
    plt.scatter(df_b["x"], df_b["y"], s=10, alpha=0.7, color="red", label="Predicted B")

    # Add circle and lines for reference
    radius = max(np.sqrt(df["x"] ** 2 + df["y"] ** 2)) * 1.05
    circle = Circle(
        (0, 0), radius, fill=False, color="black", linestyle="--", alpha=0.5
    )
    plt.gca().add_patch(circle)
    for angle_circ in range(0, 360, 20):
        rad = np.deg2rad(angle_circ)
        x = radius * np.cos(rad)
        y = radius * np.sin(rad)
        plt.plot([0, x], [0, y], color="gray", linewidth=0.5, alpha=0.5)
        plt.text(x * 1.05, y * 1.05, f"{angle_circ}°", ha="center", va="center")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axhline(y=0, color="black", linewidth=1)
    plt.axvline(x=0, color="black", linewidth=1)
    plt.title(
        f"images predicted as A/B, trained on {angle:.1f}° and {opposite_angle:.1f}° clusters"
    )
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, f"scatter_frame_{frame_id:03d}.png")
    plt.savefig(path, dpi=300)
    print(f"\nScatter Frame {frame_id} completed.\n")
    plt.close()


def create_linear_graph(angle, frame_id, save_dir=LINEAR_DIR):
    """
    Creates a linear graph showing the prediction percentages across different angles.
    Parameters:
        angle (float): The angle used for training the model.
        frame_id (int): Frame number for the filename.
        save_dir (str): Directory to save the image.
    """
    opposite_angle = (angle + 180) % 360
    os.makedirs(save_dir, exist_ok=True)

    pred_a = pd.DataFrame(
        {"filename": safe_read_filenames(inside_tmp("predicted_as_A.csv"))}
    )
    pred_b = pd.DataFrame(
        {"filename": safe_read_filenames(inside_tmp("predicted_as_B.csv"))}
    )

    pred_a["pred"] = "A"
    pred_b["pred"] = "B"

    df = pd.concat([pred_a, pred_b], ignore_index=True)

    # use precomputed PCA angles
    df["angle_deg"] = df["filename"].map(ANGLE_MAP)
    df = df.dropna(subset=["angle_deg"])

    window_size = 20
    results = []

    # Calculate the percentage of images predicted as A and B for each angle window.
    for step_angle in range(0, 360, 1):
        end = (step_angle + window_size) % 360

        if step_angle < end:
            window_data = df[(df["angle_deg"] >= step_angle) & (df["angle_deg"] < end)]
        else:
            window_data = df[(df["angle_deg"] >= step_angle) | (df["angle_deg"] < end)]

        total = len(window_data)

        if total > 0:
            count_a = (window_data["pred"] == "A").sum()
            percent_a = count_a * 100 / total
            percent_b = 100 - percent_a
        else:
            percent_a = 0
            percent_b = 0

        center_angle = (step_angle + window_size / 2) % 360
        results.append((center_angle, percent_a, percent_b))

    df_results = pd.DataFrame(results, columns=["angle", "percent_A", "percent_B"])

    angle0 = df_results[df_results["angle"] == 0]
    angle360 = angle0.copy()
    angle360["angle"] = 360
    df_results = pd.concat([df_results, angle360], ignore_index=True)
    df_results = df_results.sort_values(by="angle")

    plt.figure(figsize=(12, 6))
    plt.plot(
        df_results["angle"], df_results["percent_A"], label="Predicted A", color="blue"
    )
    plt.plot(
        df_results["angle"], df_results["percent_B"], label="Predicted B", color="red"
    )

    plt.xlabel("Angle")
    plt.ylabel("%")
    plt.title(
        f"% images predicted as A/B, trained on {angle:.1f}° and {opposite_angle:.1f}° clusters, {window_size}° slices"
    )

    plt.axhline(y=0, color="black", linewidth=1)
    plt.axvline(x=0, color="black", linewidth=1)
    plt.axvline(x=angle, color="blue", linewidth=1, linestyle="--")
    plt.axvline(x=opposite_angle, color="red", linewidth=1, linestyle="--")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 100)
    plt.xlim(0, 360)

    path = os.path.join(save_dir, f"linear_frame_{frame_id:03d}.png")
    plt.savefig(path, dpi=300)
    print(f"\nLinear Frame {frame_id} completed.\n")
    plt.close()


def take_k_closest_to_angle(filenames, center_angle_deg, k, pca_df=PCA_DF):
    """
    filenames: iterable of image filenames
    center_angle_deg: angle in [0,360)
    returns: list of k filenames closest in ANGLE to center_angle_deg (circular distance)
    Not using Euclidean distance in PCA space, but rather the angular distance in degrees.
    """
    df = pca_df[pca_df["filename"].isin(filenames)].copy()
    if df.empty:
        return []

    delta = (df["angle_deg"] - center_angle_deg).abs()
    df["ang_dist"] = np.minimum(delta, 360 - delta)

    return df.sort_values("ang_dist").head(k)["filename"].tolist()


def percent_predicted_as_filenames(
    model,
    filenames,
    target_pred,
    image_source_dir="female_faces",
):
    """
    Compute the percentage of images in filenames that are predicted as target_pred by the model.
    """

    model.eval()

    count_target = 0
    total = 0


    with torch.no_grad():

        for fname in filenames:

            try:

                embedding = get_arcface_embedding(
                    fname
                )

            except KeyError:

                continue

            x = embedding.unsqueeze(
                0
            ).to(device)


            output = model(
                x
            )


            pred = output.argmax(
                dim=1
            ).item()


            count_target += int(
                pred == target_pred
            )

            total += 1


    percent = (
        100 * count_target / total
        if total > 0
        else 0
    )


    return (
        percent,
        total,
    )

def percent_predicted_as_filenames_batched(
    model,
    filenames,
    target_pred,
    image_source_dir="female_faces",
    batch_size=50,
):
    """
    Compute the percentage of images in filenames that are predicted as target_pred by the model, processing in batches for efficiency.
    """

    model.eval()

    count_target = 0
    total = 0

    batch_embeddings = []


    with torch.no_grad():

        for fname in filenames:

            try:

                embedding = get_arcface_embedding(
                    fname
                )

            except KeyError:

                continue


            batch_embeddings.append(
                embedding
            )


            if len(batch_embeddings) == batch_size:

                batch = torch.stack(
                    batch_embeddings
                ).to(device)


                outputs = model(
                    batch
                )


                preds = outputs.argmax(
                    dim=1
                )


                count_target += (
                    preds == target_pred
                ).sum().item()


                total += len(
                    batch_embeddings
                )


                batch_embeddings = []


        if batch_embeddings:

            batch = torch.stack(
                batch_embeddings
            ).to(device)


            outputs = model(
                batch
            )


            preds = outputs.argmax(
                dim=1
            )


            count_target += (
                preds == target_pred
            ).sum().item()


            total += len(
                batch_embeddings
            )


    percent = (
        100 * count_target / total
        if total > 0
        else 0
    )


    return (
        percent,
        total,
    )

def compute_cluster_concentration(
    angle, iteration, cluster_concentration=None, k_eval=100
):
    """
    Evaluate the current trained classifier after clustering-based training.

    Measures:
      - Among the k_eval images closest to the current A angle,
        what percentage is classified as A by the final classifier.
      - Among the k_eval images closest to the opposite B angle,
        what percentage is classified as B by the final classifier.
    """
    if cluster_concentration is None:
        cluster_concentration = []

    a_csv = inside_tmp("filenames_A.csv")
    b_csv = inside_tmp("filenames_B.csv")

    if not (os.path.exists(a_csv) and os.path.exists(b_csv)):
        print("Warning: filenames_A/B.csv not found. Skipping.")
        return cluster_concentration

    dfA = pd.read_csv(a_csv)
    dfB = pd.read_csv(b_csv)

    opposite_angle = (angle + 180) % 360

    # pick only k_eval closest-by-angle within each cluster. Because the clusters are not perfectly circular, we don't want to take all images in the cluster, just the ones closest to the training angle.
    eval_A_filenames = take_k_closest_to_angle(dfA["filename"].tolist(), angle, k_eval)
    eval_B_filenames = take_k_closest_to_angle(
        dfB["filename"].tolist(), opposite_angle, k_eval
    )

    # Evaluate the percentage of images predicted as A in cluster A and as B in cluster B using the trained model.
    pct_A_in_A, nA = percent_predicted_as_filenames_batched(
        self_training_model, eval_A_filenames, target_pred=0
    )

    pct_B_in_B, nB = percent_predicted_as_filenames_batched(
        self_training_model, eval_B_filenames, target_pred=1
    )

    cluster_concentration.append(
        {
            "iteration": iteration,
            "angle": angle,
            "A_percent_in_A_cluster": pct_A_in_A,
            "B_percent_in_B_cluster": pct_B_in_B,
            "nA": nA,
            "nB": nB,
            "k_eval": k_eval,
            "opposite_angle": opposite_angle,
        }
    )

    print(
        f"Iter {iteration}: classifier predicts A for {pct_A_in_A:.1f}% "
        f"of {k_eval} images near {angle:.1f}° (n={nA}), "
        f"and predicts B for {pct_B_in_B:.1f}% "
        f"of {k_eval} images near {opposite_angle:.1f}° (n={nB})"
    )

    return cluster_concentration


def plot_cluster_concentration(
    cluster_concentration,
    save_path="cluster_concentration_over_rotations.png",
    type_of_learning="Unsupervised Learning",
):
    """
    Plot:
      - % predicted as A in cluster A in every rotation step.
      - % predicted as B in cluster B in every rotation step.

    X-axis stays in rotation order, but tick labels show: base_angle/opposite_angle
    """
    df_seq = pd.DataFrame(cluster_concentration)

    x = np.arange(len(df_seq))

    plt.figure(figsize=(12, 6))

    plt.plot(
        x,
        df_seq["A_percent_in_A_cluster"],
        label="% predicted A in cluster centered at base angle",
        color="blue",
        linewidth=2,
    )

    plt.plot(
        x,
        df_seq["B_percent_in_B_cluster"],
        label="% predicted B in cluster centered at opposite angle",
        color="red",
        linewidth=2,
    )

    angle_labels = [
        f"{row['angle']:.2f}°/{((row['angle'] + 180) % 360):.2f}°"
        for _, row in df_seq.iterrows()
    ]

    step = max(1, len(x) // 12)
    plt.xticks(x[::step], angle_labels[::step], rotation=45)

    plt.xlabel("Base angle / Opposite angle")
    plt.ylabel("% of images classified as cluster label")

    plt.title(
        f"At Each Rotation Step: How Many of the 100 Closest Images\n"
        f"Are Classified as the Cluster’s Intended Label\n"
        f"-- {type_of_learning} --"
    )

    plt.ylim(0, 100)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_label_change_csvs(training_log_path="training_log.csv"):
    """
    Analysis of the training log to find images that changed cluster labels between consecutive iterations. Saves two CSV files:
    1. changed_images_between_consecutive_iterations.csv: Contains details of images that changed labels
    2. changed_images_summary_per_iteration.csv: Summary of the number of images that changed labels per iteration
    """
    df = pd.read_csv(training_log_path)

    df = df.sort_values(["filename", "iteration"]).reset_index(drop=True)

    # Convert cluster labels to binary values:
    # A -> 0, B -> 1
    df["pred_binary"] = (df["cluster_label"] == "B").astype(int)

    df["prev_iteration"] = df.groupby("filename")["iteration"].shift(1)
    df["prev_cluster_label"] = df.groupby("filename")["cluster_label"].shift(1)
    df["prev_pred_binary"] = df.groupby("filename")["pred_binary"].shift(1)
    df["prev_cluster_angle"] = df.groupby("filename")["cluster_angle"].shift(1)

    df["is_consecutive"] = df["iteration"] == df["prev_iteration"] + 1

    changed = df[
        df["prev_iteration"].notna()
        & df["is_consecutive"]
        & (df["pred_binary"] != df["prev_pred_binary"])
    ].copy()

    changed["from_label"] = changed["prev_cluster_label"]
    changed["to_label"] = changed["cluster_label"]
    changed["from_iteration"] = changed["prev_iteration"].astype(int)
    changed["to_iteration"] = changed["iteration"].astype(int)

    changed_csv = changed[
        [
            "filename",
            "image_angle",
            "from_iteration",
            "to_iteration",
            "from_label",
            "to_label",
            "prev_cluster_angle",
            "cluster_angle",
        ]
    ].rename(
        columns={
            "prev_cluster_angle": "from_cluster_angle",
            "cluster_angle": "to_cluster_angle",
        }
    )

    changed_csv.to_csv(
        inside_output("changed_images_between_consecutive_iterations.csv"),
        index=False,
    )

    summary = (
        changed_csv.groupby("to_iteration")["filename"]
        .count()
        .reset_index(name="num_changed_images")
    )

    summary.to_csv(
        inside_output("changed_images_summary_per_iteration.csv"),
        index=False,
    )

    return changed_csv, summary


def estimate_model_angle_from_predictions():
    """
    Estimate the model's decision boundary angle based on the predictions of images classified as A and B.
    Returns:
        model_angle (float): Estimated angle of the model's decision boundary in degrees.ד
    """
    pred_a = safe_read_filenames(inside_tmp("predicted_as_A.csv"))
    pred_b = safe_read_filenames(inside_tmp("predicted_as_B.csv"))

    df_a = pd.DataFrame({"filename": pred_a, "pred": 0})
    df_b = pd.DataFrame({"filename": pred_b, "pred": 1})
    df = pd.concat([df_a, df_b], ignore_index=True)

    # Map filenames to their corresponding angles using the precomputed ANGLE_MAP and sort by angle.
    df["angle_deg"] = df["filename"].map(ANGLE_MAP)
    df = df.dropna(subset=["angle_deg"]).sort_values("angle_deg")

    if len(df) == 0:
        return np.nan

    preds = df["pred"].values
    angles = df["angle_deg"].values

    # Find the indices where the predictions change from A to B or B to A. This indicates a potential decision boundary.
    changes = np.where(preds[:-1] != preds[1:])[0]

    if len(changes) == 0:
        return np.nan

    # Take the first change as the decision boundary.
    idx = changes[0]
    # Compute the angle of the decision boundary as the average of the two angles where the prediction changes.
    boundary_angle = (angles[idx] + angles[idx + 1]) / 2

    # The model's decision boundary is perpendicular to the line connecting the two clusters, so we subtract 90 degrees to get the model's angle.
    model_angle = (boundary_angle - 90) % 360

    return model_angle

if __name__ == "__main__":

    UNSUPERVISED = True  # Set to True for unsupervised self-training, False for supervised training
    ROTATION_DEGS = 0.1
    NUM_ITERATIONS = 3600 #10800 # 3 rounds of 360 degrees at 0.1 degree increments
    NUM_EPOCHS = 1
    PLOT_EVERY = 100
    NUM_OF_IMAGES_PER_CLUSTER = 10
    LR = 0.001
    WEIGHT_DECAY = 1
    K_EVAL = 100 # number of images to evaluate cluster concentration on

    names, points = load_top2_filtered("pca_top2_filtered_female_1.csv")
    base_point, opposite_point = create_base_and_opposite_points(0,csv_path="pca_top2_filtered_female_1.csv")
    self_training_model = load_model(
        model_path="model_ft_0_ARCFACE_RESNET50_M.pth"
    )
    self_training_model = self_training_model.to(device)

    optimizer_ft = optim.AdamW(
        self_training_model.classifier.parameters(), # only train the classifier layer
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    # Generate rotation sequence
    rotation_seq, _ = generate_rotation_sequence(
        base_point=base_point,
        all_points=points,
        all_names=names,
        num_steps=360,
        start_angle=0,
        rotation_range=360,
        used_indices=set(),  # ensure we don't reuse the same images to train on in the rotation sequence, so we get 360 unique images in the sequence (one per degree)
    )

    df_seq = pd.DataFrame(rotation_seq, columns=["step", "angle_deg", "filename"])
    df_seq.to_csv(inside_tmp("rotation_sequence_all.csv"), index=False)
    print("Generated rotation sequence with unique images at each step.\n")

    cluster_concentration = []
    training_log = []
    angle_tracking_log = []

    start = time.time()
    for i in range(
        NUM_ITERATIONS
    ): 
        total_angle_deg = i * ROTATION_DEGS # total rotation angle in degrees
        angle_deg = total_angle_deg % 360   # current angle in degrees for this iteration

        # Collect nearest images to the base and opposite points for training
        collect_nearest_images(
            base_point, points, names, output_dir=inside_tmp("A"), k=NUM_OF_IMAGES_PER_CLUSTER
        )
        collect_nearest_images(
            opposite_point, points, names, output_dir=inside_tmp("B"), k=NUM_OF_IMAGES_PER_CLUSTER
        )
        # now we have two directories: A and B with NUM_OF_IMAGES_PER_CLUSTER images each from opposite clusters

        if UNSUPERVISED:
            merge_clusters()    # merge A and B into a single CSV for classification
            t = time.time()
            training_records = classify_images_batched(
                self_training_model,
                inside_tmp("filenames_merged.csv"),
                clusters=True,
                batch_size=50,
            )   # classify images in the merged cluster and return training records for logging

            print("Classified images in clusters A and B.")
            print(f"classify_images_batched time: {time.time() - t:.2f}s")
            
            for rec in training_records:
                image_angle = ANGLE_MAP.get(rec["filename"], np.nan)

                training_log.append(
                    {
                        "iteration": i,
                        "cluster_angle": angle_deg,
                        "filename": rec["filename"],
                        "image_angle": image_angle,
                        "cluster_label": rec["pred"],
                        "prob_A": rec["prob_A"],
                    }
                )

            # All images that the model predicted as A or B are saved in cluster_predicted_as_A.csv and cluster_predicted_as_B.csv respectively. We read those files to get the filenames for training.
            df_A = pd.read_csv(inside_tmp("cluster_predicted_as_A.csv"))
            df_B = pd.read_csv(inside_tmp("cluster_predicted_as_B.csv"))
        else:
            # In supervised mode, use the original cluster membership as the true labels:
            # images selected near the A point are labeled A, and images selected near
            # the opposite B point are labeled B. No pseudo-labeling by the model is performed.
            df_A = pd.read_csv(inside_tmp("filenames_A.csv"))
            df_B = pd.read_csv(inside_tmp("filenames_B.csv"))

        print(f"Pseudo-label split: A={len(df_A)}, B={len(df_B)}")
        # Combine the filenames and labels for training
        filenames = df_A["filename"].tolist() + df_B["filename"].tolist()
        labels = [0] * len(df_A) + [1] * len(df_B)

        # Create dataloaders for training the model on the selected images
        dataloaders, dataset_sizes, class_names = get_dataloaders_from_lists(
            filenames,
            labels,
            image_dir="female_faces",
            batch_size=100,
        )
        ################
        criterion = nn.CrossEntropyLoss()
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=5, gamma=1
        )  # right now no LR decay
        ################
        if UNSUPERVISED == True:
            self_training_model = train_model_fast_for_self_training(
                self_training_model,
                dataloaders,
                dataset_sizes,
                criterion,
                optimizer_ft,
                exp_lr_scheduler,
                num_epochs=NUM_EPOCHS,
            )
        else:
            self_training_model = train_model(
                self_training_model,
                dataloaders,
                dataset_sizes,
                criterion,
                optimizer_ft,
                exp_lr_scheduler,
                num_epochs=NUM_EPOCHS,
                plots = False
            )

        # now we have a trained model - self trained on it's own predictions

        # Plotting and evaluation every PLOT_EVERY iterations
        if i % PLOT_EVERY == 0:
            t = time.time()

            classify_images_batched(
                self_training_model,
                csv_path=inside_tmp("rotation_sequence_all.csv"),
                clusters=False,
                batch_size=50,
            )

            print("Classified rotation sequence.")
            print(f"rotation sequence classification time: {time.time()-t:.2f}s")

            model_angle = estimate_model_angle_from_predictions()

            angle_tracking_log.append({
                "iteration": i,
                "example_angle": total_angle_deg,
                "model_angle": model_angle,
            })
            

            if UNSUPERVISED:
                create_prediction_scatter(angle=angle_deg, frame_id=i)

            create_linear_graph(angle=angle_deg, frame_id=i)


        t = time.time()

        cluster_concentration = compute_cluster_concentration(
            angle=angle_deg,
            iteration=i,
            cluster_concentration=cluster_concentration,
            k_eval=K_EVAL
        )

        print(f"Concentration time: {time.time()-t:.2f}s")
        
        # rotate base_point and opposite_point by ROTATION_DEGS degrees for the next iteration
        base_point = rotate_vector(base_point, angle_deg=ROTATION_DEGS)  
        opposite_point = rotate_vector(opposite_point, angle_deg=ROTATION_DEGS) 
        # delete csv files
        csv_files_to_delete = [
            inside_tmp("filenames_A.csv"),
            inside_tmp("filenames_B.csv"),
            inside_tmp("predicted_as_A.csv"),
            inside_tmp("predicted_as_B.csv"),
        ]

        if UNSUPERVISED:
            csv_files_to_delete += [
                inside_tmp("filenames_merged.csv"),
                inside_tmp("cluster_predicted_as_A.csv"),
                inside_tmp("cluster_predicted_as_B.csv"),
            ]

        for fname in csv_files_to_delete:
            if os.path.exists(fname):
                try:
                    os.remove(fname)
                except Exception as e:
                    print(f"Could not delete {fname}: {e}")
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
    print("Total time (minutes): ", (end - start) / 60, "minutes")
    plot_cluster_concentration(
        cluster_concentration,
        save_path=inside_output("cluster_concentration_over_rotations.png"),
        type_of_learning=(
            "Unsupervised Learning" if UNSUPERVISED else "Supervised Learning"
        ),
    )

    pd.DataFrame(training_log).to_csv(inside_output("training_log.csv"), index=False)

    changed_csv, summary = save_label_change_csvs(inside_output("training_log.csv"))

    df_angles = pd.DataFrame(angle_tracking_log)

    # Save raw angle measurements
    df_angles.to_csv(
        inside_output("angle_tracking_log.csv"),
        index=False
    )

    # Align the model angle with the continuously rotating example angle.
    # The model's decision-boundary orientation has a 180° ambiguity, meaning that
    # angles separated by 180° represent the same boundary orientation.
    # In addition, the measured model angle is restricted to [0°, 360°), while the
    # example angle continues increasing across multiple rotations (e.g., 360°, 720°).
    # Therefore, for each measurement, choose the equivalent model angle
    # (model_angle + k*180°) that is closest to the current example angle.
    # This does not modify the model or its predictions; it only creates a continuous
    # representation of the model angle that can be meaningfully compared with the
    # rotating examples across multiple full rotations.
    example_angles = df_angles["example_angle"].values
    model_angles = df_angles["model_angle"].values

    model_aligned = []

    for example_angle, model_angle in zip(
        example_angles,
        model_angles
    ):

        if np.isnan(model_angle):
            model_aligned.append(np.nan)
            continue

        k = round(
            (example_angle - model_angle) / 180
        )

        best_angle = model_angle + 180 * k

        model_aligned.append(best_angle)

    df_angles["model_angle_aligned"] = model_aligned

    # Save also the aligned angles
    df_angles.to_csv(
        inside_output("angle_tracking_log.csv"),
        index=False
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    example_plot = df_angles["example_angle"] % 360
    model_plot = df_angles["model_angle_aligned"] % 360

    plt.figure(figsize=(10, 5))

    plt.plot(
        df_angles["iteration"],
        example_plot,
        label="examples",
        linewidth=2
    )

    plt.plot(
        df_angles["iteration"],
        model_plot,
        label="weights / model",
        linewidth=2
    )

    plt.xlabel("iteration")
    plt.ylabel("angle")

    plt.ylim(0, 360)

    plt.title(
        f"rotation tracking, "
        f"step={ROTATION_DEGS} degs/iteration"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        inside_output("angle_tracking_graph.png"),
        dpi=300
    )

    plt.close()


    torch.save(
        self_training_model.state_dict(), inside_output("model_self_trained.pth")
    )

    log_file.close()
