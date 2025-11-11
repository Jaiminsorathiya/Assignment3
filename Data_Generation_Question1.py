"""

-------------------------------------------------
In this script, I define and visualize the 4-class 3D Gaussian data distribution
that I used for training the MLP classifier and computing the Bayes-optimal rule.

The goal is to generate synthetic data drawn from known Gaussian class-conditionals
so I can later evaluate both theoretical and empirical classification errors.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# -------------------------------------------------------
# Step 1: Define the True Data Distribution (4 Gaussian Classes)
# -------------------------------------------------------

# I fix the random seed for reproducibility
np.random.seed(42)

# I define 4 class means in 3D space, placed so that classes moderately overlap
means = np.array([
    [0.0, 0.0, 0.0],
    [2.2, 2.2, 0.0],
    [2.2, 0.0, 2.2],
    [0.0, 2.2, 2.2]
])

# Each class has its own covariance matrix
covariances = [
    np.array([[1, 0.5, 0.3],
              [0.5, 1, 0.2],
              [0.3, 0.2, 1]]),
    np.array([[1, 0.2, 0.1],
              [0.2, 1, 0.3],
              [0.1, 0.3, 1]]),
    np.array([[1, 0.1, 0.2],
              [0.1, 1, 0.3],
              [0.2, 0.3, 1]]),
    np.array([[1, 0.2, 0.3],
              [0.2, 1, 0.1],
              [0.3, 0.1, 1]])
]

# Equal priors for all classes
priors = np.ones(4) / 4


# -------------------------------------------------------
# Step 2: Sampling Function
# -------------------------------------------------------
def sample_gaussian_classes(n_samples):
    """
    I sample n_samples points according to the 4-class Gaussian model.
    Each sample is drawn from one of the four Gaussians with equal prior.
    """
    labels = np.random.choice(4, size=n_samples, p=priors)
    X = np.zeros((n_samples, 3))
    for i, c in enumerate(labels):
        X[i] = np.random.multivariate_normal(means[c], covariances[c])
    return X, labels


# -------------------------------------------------------
# Step 3: Generate Datasets of Different Sizes
# -------------------------------------------------------
dataset_sizes = [100, 500, 1000, 5000, 10000]
datasets = {}

for N in dataset_sizes:
    X, y = sample_gaussian_classes(N)
    datasets[N] = (X, y)
    np.savez(f"gaussian_data_N{N}.npz", X=X, y=y)
    print(f"Saved dataset with N={N} samples")

# I also generate a large test set (used only for evaluation)
X_test, y_test = sample_gaussian_classes(100000)
np.savez("gaussian_test_data.npz", X=X_test, y=y_test)
print("Saved test dataset with N=100000 samples")


# -------------------------------------------------------
# Step 4: Visualize Data for N=1000
# -------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
X_vis, y_vis = datasets[1000]

# I plot each class in a different color
for c in range(4):
    ax.scatter(X_vis[y_vis == c, 0],
               X_vis[y_vis == c, 1],
               X_vis[y_vis == c, 2],
               color=colors[c],
               alpha=0.6,
               label=f'Class {c+1}')

# Mark class means
for idx, mu in enumerate(means):
    ax.scatter(mu[0], mu[1], mu[2], color='black', marker='x', s=80)
    ax.text(mu[0]+0.1, mu[1]+0.1, mu[2]+0.1, f'C{idx+1}', color='black')

ax.set_title("True 4-Class 3D Gaussian Data (N = 1000)", fontsize=13)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.legend()
plt.tight_layout()
plt.savefig("gaussian_data_visualization.png", dpi=300)
plt.show()
