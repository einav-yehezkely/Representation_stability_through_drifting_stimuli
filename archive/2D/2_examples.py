#################################################################
# Adaptation to rotating stimuli
# a system in which the input undergoes drift
# There are two examples
# 06/05/2025
#################################################################
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([1, 0])
x2 = np.array([-1, 0])
X = np.vstack([x1, x2])
tags = np.array([1, 0])


def learning(mat: np.array, tag: np.ndarray) -> np.ndarray:
    """
    num of cols = dimension = N
    num of rows = examples = P
    tag = desired classification. Length = P
    comment: I used a matrix with size of PxN and not NxP for convenience purposes only. Each row is an example and the number of columns is the dimension. In case of a given matrix with size NxP, we can use matrix.T to transpose it and the function will work.
    """
    P = mat.shape[0]
    N = mat.shape[1]
    weight = np.ones(N)
    updated = True
    max_iterations = 1000  # for non-linear data
    iterations = 0

    while updated and iterations < max_iterations:
        updated = False
        for i in range(P):
            classification = int(np.heaviside(np.dot(weight, mat[i]), 1))
            if classification != tag[i]:
                weight = weight + (2 * tag[i] - 1) * mat[i]
                updated = True
        iterations += 1
    return weight


def rotate_examples(X: np.ndarray, theta: float) -> np.ndarray:
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return X @ R.T


def separating_line(weight):
    """
    calculates and draws the separating line
    """
    vertical_vector = (-weight[1], weight[0])
    if vertical_vector[0] != 0:
        m = vertical_vector[1] / vertical_vector[0]
        c = 0  # y-intercept
        x_line = np.linspace(-10, 10, 100)
        y_line = m * x_line + c  # The equation of a straight line
    else:
        # if slope is vertical
        x_line = np.full(100, 0)
        y_line = np.linspace(-10, 10, 100)
    plt.plot(
        x_line,
        y_line,
        color="black",
        linestyle="--",
        linewidth=2,
    )


weight = learning(X, tags)
print("Weight vector:", weight)


plt.figure(figsize=(6, 6))
plt.scatter(X[tags == 1][:, 0], X[tags == 1][:, 1], color="blue", s=10)
plt.scatter(X[tags == 0][:, 0], X[tags == 0][:, 1], color="red", s=10)

plt.quiver(
    0,
    0,
    weight[0],
    weight[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="black",
    label="Weight vector",
    headwidth=3,  # arrow
    linewidth=1,
)

separating_line(weight)
plt.show()
