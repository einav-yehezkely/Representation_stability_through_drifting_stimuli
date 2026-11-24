from collections import Counter
from pathlib import Path
from shutil import copy2
from tempfile import mkdtemp

from deepface import DeepFace

SOURCE = Path(
    r"C:\Users\einav\clone_here\Representation_stability_through_drifting_stimuli"
    r"\composites\generated\candidates"
)
MIN_SCORE = 90.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LABEL_TO_FOLDER = {"Woman": "women", "Man": "men"}


def classify_image(path: Path) -> str:
    faces = DeepFace.analyze(
        img_path=str(path),
        actions=("gender",),
        detector_backend="opencv",
        enforce_detection=True,
        silent=True,
    )

    if len(faces) != 1:
        return "review"

    face = faces[0]
    label = face["dominant_gender"]
    if face["gender"][label] < MIN_SCORE:
        return "review"

    return LABEL_TO_FOLDER.get(label, "review")


def main() -> None:
    images = sorted(
        path for path in SOURCE.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise SystemExit(f"No images found in {SOURCE}")

    DeepFace.build_model("Gender", task="facial_attribute")

    output = Path(mkdtemp(prefix="candidates_sorted_", dir=SOURCE.parent))
    folders = ("women", "men", "review")
    for folder in folders:
        (output / folder).mkdir()

    counts = Counter()
    for path in images:
        try:
            folder = classify_image(path)
        except (ValueError, OSError) as error:
            folder = "review"
            print(f"{path.name}: {error}")

        copy2(path, output / folder / path.name)
        counts[folder] += 1
        print(f"{path.name} -> {folder}")

    print(f"\nOutput: {output}")
    for folder in folders:
        print(f"{folder}: {counts[folder]}")


if __name__ == "__main__":
    main()