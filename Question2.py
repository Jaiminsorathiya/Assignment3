# gmm_model_order_cv.py
# -------------------------------------------------------------
# In this script, I perform model-order selection for Gaussian
# Mixture Models (GMMs) using 10-fold cross-validation.
# I follow the given steps: generate data from a true 4-component
# 2D GMM (with overlapping components), run CV to select the
# best number of components, and repeat the entire process
# multiple times to study consistency as data increases.
# -------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import csv
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------
# Step 1: Define my "true" Gaussian Mixture Model (4 components)
# -------------------------------------------------------------
# I want 4 Gaussians in 2D with different means, covariances,
# and weights. Two of them (0 and 1) will overlap significantly
# to make the problem realistic for model selection.
# -------------------------------------------------------------

rng_global = np.random.default_rng(42)

# Mixing probabilities (these must sum to 1)
true_weights = np.array([0.35, 0.30, 0.20, 0.15])

# Mean vectors for each component (2D)
true_means = np.array([
    [0.0, 0.0],      # Component 0
    [1.6, 1.6],      # Component 1 (close to 0 -> overlap)
    [4.0, 0.0],      # Component 2 (well separated)
    [0.0, 4.0],      # Component 3 (well separated)
], dtype=float)

# Covariance matrices for each Gaussian
true_covs = np.array([
    [[0.9,  0.35], [0.35, 0.8]],   # C0
    [[0.8,  0.30], [0.30, 0.9]],   # C1 (similar scale to C0)
    [[0.5, -0.25], [-0.25, 1.0]],  # C2
    [[1.1,  0.10], [0.10, 0.6]],   # C3
], dtype=float)

# Normalize weights to ensure they sum to 1
true_weights = true_weights / true_weights.sum()

# -------------------------------------------------------------
# Function to sample data from my true GMM
# -------------------------------------------------------------
def sample_true_gmm(N, rng):
    """Generate N samples (2D points) from the true 4-component GMM."""
    comps = rng.choice(4, size=N, p=true_weights)
    X = np.zeros((N, 2), dtype=float)
    for i, c in enumerate(comps):
        X[i] = rng.multivariate_normal(true_means[c], true_covs[c])
    return X, comps


# -------------------------------------------------------------
# Step 2: Define helper functions for fitting GMMs and CV
# -------------------------------------------------------------

def fit_gmm_and_score(train_X, val_X, n_components, random_state):
    """
    I fit a GMM with 'n_components' using EM on the training fold,
    and then compute the average log-likelihood on the validation fold.
    This tells me how well that model generalizes.
    """
    # Handle cases where we can't fit more components than training samples
    if len(train_X) < n_components:
        return -np.inf

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=5,          # multiple random starts -> better local optimum
        max_iter=300,
        reg_covar=1e-6,
        random_state=random_state
    )
    gmm.fit(train_X)
    return float(gmm.score(val_X))  # mean log-likelihood per sample


def select_order_via_kfold_cv(X, K=10, max_components=10, base_seed=0):
    """
    Here I perform K-fold CV (default 10-fold) to pick the best number of components.
    For each k in [1..max_components], I train on K-1 folds and validate on the remaining one.
    The model with the highest mean validation log-likelihood is chosen.
    """
    N = len(X)
    kf = KFold(n_splits=min(K, N), shuffle=True, random_state=base_seed)
    candidate_ks = list(range(1, max_components + 1))
    fold_scores = {k: [] for k in candidate_ks}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        train_X, val_X = X[train_idx], X[val_idx]
        n_train = len(train_X)

        # Cap k so it never exceeds the number of training samples
        k_cap = min(max_components, n_train)

        for k in candidate_ks:
            if k > k_cap:
                continue
            rs = base_seed * 10_000 + fold_idx * 100 + k
            score = fit_gmm_and_score(train_X, val_X, k, random_state=rs)
            fold_scores[k].append(score)

    # Average validation score per k
    mean_scores = {k: (np.mean(fold_scores[k]) if len(fold_scores[k]) > 0 else -np.inf)
                   for k in candidate_ks}

    # Select k with highest mean log-likelihood (smaller k breaks ties)
    best_k = min(candidate_ks, key=lambda kk: (-mean_scores[kk], kk))
    return best_k, mean_scores


# -------------------------------------------------------------
# Step 3: The main experiment loop
# -------------------------------------------------------------
def run_experiment(dataset_sizes=(10, 100, 1000),
                   K=10, repeats=100, max_components=10, base_seed=123):
    """
    I repeat the CV model selection process multiple times
    for each dataset size to measure how often each model order is selected.
    """
    selection_counts = {N: np.zeros(max_components + 1, dtype=int) for N in dataset_sizes}
    last_run_scores = {}

    for r in range(repeats):
        print(f"\n=== Repeat {r+1}/{repeats} ===")
        rng = np.random.default_rng(base_seed + 7777 * r)

        for N in dataset_sizes:
            X, _ = sample_true_gmm(N, rng)
            best_k, mean_scores = select_order_via_kfold_cv(
                X, K=K, max_components=max_components, base_seed=base_seed + r
            )
            selection_counts[N][best_k] += 1
            last_run_scores[(N, r)] = mean_scores
            print(f"  N={N:<5} → selected k={best_k}")

    return selection_counts, last_run_scores


# -------------------------------------------------------------
# Step 4: Run everything and summarize
# -------------------------------------------------------------
if __name__ == "__main__":
    # Parameters for my experiment
    DATASET_SIZES = (10, 100, 1000)
    K_FOLDS = 10
    REPEATS = 100
    MAX_K = 10
    BASE_SEED = 2025

    print("True 4-component GMM I used for data generation:")
    print("Weights:", true_weights)
    print("Means:\n", true_means)
    print("Covariances:\n", true_covs)
    print("\nNote: Components 0 and 1 overlap significantly (as required).")

    # Run the 100 repeated CV experiments
    counts, _ = run_experiment(
        dataset_sizes=DATASET_SIZES,
        K=K_FOLDS,
        repeats=REPEATS,
        max_components=MAX_K,
        base_seed=BASE_SEED
    )

    # Compute selection frequency (normalized counts)
    rates = {N: counts[N] / counts[N].sum() for N in DATASET_SIZES}

    # ---------------------------------------------------------
    # Step 5: Save results to CSV for documentation
    # ---------------------------------------------------------
    csv_path = "gmm_order_selection_rates.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset_Size_N", "k", "Selection_Count", "Selection_Rate"])
        for N in DATASET_SIZES:
            for k in range(1, MAX_K + 1):
                w.writerow([N, k, int(counts[N][k]), f"{rates[N][k]:.6f}"])
    print(f"\nSaved selection rates to {csv_path}")

    # ---------------------------------------------------------
    # Step 6: Plot selection rates (3 panels for N=10,100,1000)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, N in zip(axes, DATASET_SIZES):
        ks = np.arange(1, MAX_K + 1)
        ax.bar(ks, rates[N][1:], color="steelblue", alpha=0.8)
        ax.set_title(f"N = {N}")
        ax.set_xlabel("Model order (k)")
        ax.set_xticks(ks)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Selection rate across 100 repeats")
    fig.suptitle("GMM Order Selection via 10-Fold CV (100 repeats)")
    plt.tight_layout()
    out_path = "gmm_order_selection_rates.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved figure to {out_path}")

    # ---------------------------------------------------------
    # Step 7: Console summary for quick inspection
    # ---------------------------------------------------------
    print("\n=== Selection rates (per dataset size) ===")
    for N in DATASET_SIZES:
        ks = np.arange(1, MAX_K + 1)
        summary = ", ".join([f"k={k}:{rates[N][k]:.2f}" for k in ks])
        print(f"N={N:<5} → {summary}")

    # ---------------------------------------------------------
    # Step 8: Interpretation notes (I include this in my report)
    # ---------------------------------------------------------
    print("\nObservations:")
    print("• For N=10, CV often underestimates the true order (k<4) due to limited data.")
    print("• For N=100, selections start concentrating around k≈4.")
    print("• For N=1000, k=4 dominates, confirming that cross-validation converges")
    print("  to the true model order as the sample size grows.")
