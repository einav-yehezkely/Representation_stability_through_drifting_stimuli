import os
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms, models
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = models.shufflenet_v2_x0_5(pretrained=False)

# model = models.resnet18(pretrained=False)

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes: A and B
model.load_state_dict(torch.load("model_ft_no_reg_45.pth"))
model.eval()

# Load CSV
df = pd.read_csv("merged_sequences.csv")

# Image transforms
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Mapping labels
label_map = {"A": 0, "B": 1}

# Results tracking
angle_errors = defaultdict(list)
all_predictions = []
all_labels = []
predicted_A = []
predicted_B = []

for i, row in df.iterrows():
    image_path = os.path.join("female_faces", row["filename"])

    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found. Skipping.")
        continue

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        # prediction is the index with max probability
        pred = output.argmax(dim=1).item()

    # true_label = label_map[row["group"]]
    angle = row["angle_deg"]

    all_predictions.append(pred)
    # all_labels.append(true_label)

    # Track predictions
    if pred == 0:
        predicted_A.append(row)
    else:
        predicted_B.append(row)

        # if pred != true_label:
        angle_errors[angle].append(row["filename"])

# Save predicted CSVs
pd.DataFrame(predicted_A).to_csv("predicted_as_A.csv", index=False)
pd.DataFrame(predicted_B).to_csv("predicted_as_B.csv", index=False)


# Print accuracy
# correct = sum(p == t for p, t in zip(all_predictions, all_labels))
# accuracy = correct / len(all_labels)
# print(f"\nOverall Accuracy: {accuracy:.2%}")
