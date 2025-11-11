"""

-------------------------------------------------
In this script, I define the true 4-component 2D Gaussian Mixture Model (GMM)
used for data generation. I then visualize how the distribution looks by plotting
sampled data for N = 10, 100, and 1000 points.

This visualization helps me understand how sample density affects how well
the GMM structure can be recovered in the model order selection experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# -------------------------------------------------------
# Step 1: Define the True 4-Component 2D Gaussian Mixture
# -------------------------------------------------------

# I fix the random seed for reproducibility
np.random.seed(42)

# I define four Gaussian components with different means
# Two components (at [0,0] and [3,3]) are intentionally placed close together
# to create overlap between their distributions.
means = np.array([
    [0, 0],
    [3, 3],
    [6, 0],
    [3, -3]
])

# Each component has its own covariance matrix to control shape and orientation
covariances = [
    np.array([[1.0, 0.3], [0.3, 0.8]]),
    np.array([[1.2, 0.4], [0.4, 1.0]]),
    np.array([[1.0, -0.2], [-0.2, 0.8]]),
    np.array([[1.0, 0.5], [0.5, 1.2]])
]

# I assign equal mixture weights to all components (uniform prior)
weights = np.array([0.25, 0.25, 0.25, 0.25])


# -------------------------------------------------------
# Step 2: Sampling Function for GMM
# -------------------------------------------------------
def sample_gmm(n_samples):
    """
    This function samples 'n_samples' points from the defined GMM.
    Each point is drawn from one of the four Gaussian components
    based on the specified mixture weights.
    """
    z = np.random.choice(len(weights), size=n_samples, p=weights)
    X = np.zeros((n_samples, 2))
    for i, k in enumerate(z):
        X[i] = np.random.multivariate_normal(means[k], covariances[k])
    return X, z


# -------------------------------------------------------
# Step 3: Helper Function for Covariance Ellipse
# -------------------------------------------------------
def plot_cov_ellipse(cov, mean, ax, color, label_id):
    """
    I use this helper function to draw the 1σ ellipse
    representing the covariance contour of each Gaussian.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height,
                      angle=theta, edgecolor=color, lw=2, fc='none')
    ax.add_artist(ellipse)
    # I annotate each Gaussian mean with its component number
    ax.text(mean[0] + 0.1, mean[1] + 0.1, f"C{label_id+1}", fontsize=10, color='black')


# -------------------------------------------------------
# Step 4: Generate and Visualize Datasets for N = 10, 100, 1000
# -------------------------------------------------------
dataset_sizes = [10, 100, 1000]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, N in enumerate(dataset_sizes):
    X, labels = sample_gmm(N)
    ax = axes[idx]

    # I plot samples for each component
    for k, (mu, cov) in enumerate(zip(means, covariances)):
        ax.scatter(X[labels == k, 0], X[labels == k, 1], s=25, alpha=0.5, color=colors[k])
        plot_cov_ellipse(cov, mu, ax, colors[k], k)
        ax.scatter(mu[0], mu[1], c='black', marker='x', s=80)

    # I set plot details
    ax.set_title(f"N = {N} Samples", fontsize=12)
    ax.set_xlabel("Feature 1")
    if idx == 0:
        ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 8)
    ax.set_ylim(-6, 6)

# -------------------------------------------------------
# Step 5: Final Figure Formatting and Save
# -------------------------------------------------------
fig.suptitle("True Data Generation from 4-Component 2D Gaussian Mixture Model", fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("true_gmm_data_generation.png", dpi=300)
plt.show()
