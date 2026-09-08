import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("output_CE/angle_tracking_log.csv")

# examples are already progressing continuously: 0,10,20,...,350
example_angle = df["example_angle"].values

model_raw = df["model_angle"].values

model_aligned = []

for ex_angle, model_angle in zip(example_angle, model_raw):

    # same boundary can be represented every 180 degrees
    candidates = [
        model_angle + 180 * k
        for k in range(-3, 5)
    ]

    best = min(
        candidates,
        key=lambda x: abs(x - ex_angle)
    )

    model_aligned.append(best)

model_aligned = np.array(model_aligned)

plt.figure(figsize=(10, 5))

plt.plot(
    df["iteration"],
    example_angle,
    label="examples",
    linewidth=2
)

plt.plot(
    df["iteration"],
    model_aligned,
    label="weights / model",
    linewidth=2
)

plt.xlabel("iteration")
plt.ylabel("angle")
plt.title("rotation tracking, step=0.1 degs/iteration")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "output_CE/angle_tracking_graph_aligned.png",
    dpi=300
)

plt.show()