# Two-layer MLP vs Bayes-optimal classifier on a 4-class 3D Gaussian problem
# CPU-friendly: requires only numpy, torch, matplotlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv

# ----------------------
# Global config
# ----------------------
BASE_SEED = 42              # base seed; varied per repeat to decorrelate runs
DEVICE = torch.device("cpu")

TRAIN_SIZES   = [100, 500, 1000, 5000, 10000]
TEST_SIZE     = 100_000
N_REPEATS     = 3           # number of independent repeats per training size
K_FOLDS       = 10          # CV folds
EPOCHS_CV     = 60
EPOCHS_FINAL  = 100
RESTARTS_CV   = 2
RESTARTS_FINAL= 5
LR            = 1e-2
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 64

def candidate_P_list(N):
    if N <= 500:
        return [4, 8, 16]
    elif N <= 5000:
        return [8, 16, 32]
    else:
        return [16, 32, 64]

# ----------------------
# Data distribution (tuned to ~12–15% Bayes error)
# ----------------------
C = 4
D = 3

MEANS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([2.2, 2.2, 0.0]),
    np.array([2.2, 0.0, 2.2]),
    np.array([0.0, 2.2, 2.2]),
]

COVS = [
    np.array([[1.0, 0.5, 0.3],
              [0.5, 1.0, 0.2],
              [0.3, 0.2, 1.0]]),
    np.array([[1.0, 0.2, 0.1],
              [0.2, 1.0, 0.3],
              [0.1, 0.3, 1.0]]),
    np.array([[1.0, 0.1, 0.2],
              [0.1, 1.0, 0.3],
              [0.2, 0.3, 1.0]]),
    np.array([[1.0, 0.2, 0.3],
              [0.2, 1.0, 0.1],
              [0.3, 0.1, 1.0]]),
]

CHOL     = [np.linalg.cholesky(S) for S in COVS]
COV_INV  = [np.linalg.inv(S) for S in COVS]
COV_DET  = [np.linalg.det(S) for S in COVS]
LOG_CONST= [ -0.5 * (D * np.log(2*np.pi) + np.log(det)) for det in COV_DET ]  # log normalizer

def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

# ----------------------
# Sampling
# ----------------------
def sample_data(N: int):
    y = np.random.randint(0, C, size=N, dtype=np.int64)  # uniform prior
    X = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        c = y[i]
        z = np.random.randn(1, D).astype(np.float32)
        X[i] = (z @ CHOL[c].T + MEANS[c]).astype(np.float32)
    return X, y

# ----------------------
# Bayes classifier (known PDFs)
# ----------------------
def bayes_predict(X: np.ndarray):
    N = X.shape[0]
    logps = np.zeros((N, C), dtype=np.float64)
    for c in range(C):
        diff = X - MEANS[c]
        quad = np.einsum("ij,jk,ik->i", diff, COV_INV[c], diff)
        logps[:, c] = LOG_CONST[c] - 0.5 * quad
    return np.argmax(logps, axis=1)

def bayes_error_rate(X_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred = bayes_predict(X_test)
    return float(np.mean(y_pred != y_test))

# ----------------------
# MLP model
# ----------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ELU()  # smooth-ramp style
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))  # logits

def train_mlp_best_restart(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int,
    n_restarts: int = 3,
    n_epochs: int = 80,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = BATCH_SIZE,
) -> nn.Module:
    X_t = torch.from_numpy(X).float().to(DEVICE)
    y_t = torch.from_numpy(y).long().to(DEVICE)
    N = X_t.shape[0]
    criterion = nn.CrossEntropyLoss()

    best_model = None
    best_loss = float("inf")

    for _ in range(n_restarts):
        model = MLP(D, hidden_dim, C).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(n_epochs):
            perm = torch.randperm(N)
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                xb, yb = X_t[idx], y_t[idx]
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            logits = model(X_t)
            loss = criterion(logits, y_t).item()

        if loss < best_loss:
            best_loss = loss
            best_model = model

    return best_model

@torch.no_grad()
def classification_error(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    X_t = torch.from_numpy(X).float().to(DEVICE)
    y_t = torch.from_numpy(y).long().to(DEVICE)
    logits = model(X_t)
    preds = torch.argmax(logits, dim=1)
    return float((preds != y_t).float().mean().item())

@torch.no_grad()
def confusion_matrix(model: nn.Module, X: np.ndarray, y: np.ndarray, num_classes: int = C):
    model.eval()
    X_t = torch.from_numpy(X).float().to(DEVICE)
    logits = model(X_t)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for yi, pi in zip(y, preds):
        cm[yi, pi] += 1
    return cm

# ----------------------
# Cross-validation for hidden size P
# ----------------------
def select_hidden_size_cv(X: np.ndarray, y: np.ndarray, P_list, K: int = K_FOLDS):
    N = len(X)
    indices = np.arange(N)
    np.random.shuffle(indices)
    folds = np.array_split(indices, K)
    avg_errors = {}

    for P in P_list:
        fold_errs = []
        for k in range(K):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(K) if i != k])
            X_train, y_train = X[train_idx], y[train_idx]
            X_val,   y_val   = X[val_idx],   y[val_idx]
            model = train_mlp_best_restart(
                X_train, y_train,
                hidden_dim=P,
                n_restarts=RESTARTS_CV,
                n_epochs=EPOCHS_CV
            )
            fold_errs.append(classification_error(model, X_val, y_val))
        avg_errors[P] = float(np.mean(fold_errs))

    best_P = min(P_list, key=lambda p: (avg_errors[p], p))  # tie-breaker: smaller P
    return best_P, avg_errors

# ----------------------
# Main experiment
# ----------------------
def run_experiment():
    # Fixed test set
    set_all_seeds(BASE_SEED)
    X_test, y_test = sample_data(TEST_SIZE)
    bayes_err = bayes_error_rate(X_test, y_test)
    print(f"Bayes optimal empirical P(error) ≈ {bayes_err:.4f}")

    means, stds = [], []
    chosen_Ps_summary = {}
    perN_best_models = {}  # store best model by test error per N for confusion matrices

    # For each training size, average across repeats & keep best model
    for idx, N in enumerate(TRAIN_SIZES):
        P_list = candidate_P_list(N)
        errs = []
        best_err = float("inf")
        best_model_for_N = None

        print(f"\n=== Training size N = {N} (averaging {N_REPEATS} runs) ===")
        for r in range(N_REPEATS):
            set_all_seeds(BASE_SEED + 1000*idx + r)  # independent repeat

            X_train, y_train = sample_data(N)
            best_P, cv_errors = select_hidden_size_cv(X_train, y_train, P_list, K=K_FOLDS)
            chosen_Ps_summary.setdefault(N, []).append((best_P, cv_errors))
            print(f"  Repeat {r+1}/{N_REPEATS}: chosen P = {best_P}, CV errors = {cv_errors}")

            final_model = train_mlp_best_restart(
                X_train, y_train,
                hidden_dim=best_P,
                n_restarts=RESTARTS_FINAL,
                n_epochs=EPOCHS_FINAL
            )
            test_err = classification_error(final_model, X_test, y_test)
            errs.append(test_err)
            print(f"  → Test P(error) = {test_err:.4f}")

            if test_err < best_err:
                best_err = test_err
                best_model_for_N = final_model

        perN_best_models[N] = best_model_for_N
        m, s = float(np.mean(errs)), float(np.std(errs))
        means.append(m)
        stds.append(s)
        print(f"N = {N}: Mean Test Error = {m:.4f} ± {s:.4f} (best = {best_err:.4f})")

    # ----------------------
    # Plot: mean ± 95% CI (semilog-x)
    # ----------------------
    means_arr = np.array(means)
    stds_arr  = np.array(stds)
    ci95      = 1.96 * stds_arr / np.sqrt(N_REPEATS)

    plt.figure(figsize=(7.2, 5.0))
    plt.semilogx(TRAIN_SIZES, means_arr, marker='o', label="MLP (CV-selected P)")
    plt.fill_between(TRAIN_SIZES, means_arr - ci95, means_arr + ci95,
                     alpha=0.18, label="95% CI")
    plt.axhline(bayes_err, linestyle='--', label=f"Bayes optimal ({bayes_err:.3f})")
    plt.xlabel("Number of training samples (log scale)")
    plt.ylabel("Test P(error)")
    plt.title("MLP vs Theoretical Optimal Classifier (mean ± 95% CI)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    # exact tick labels for clarity
    plt.xticks(TRAIN_SIZES, [str(n) for n in TRAIN_SIZES])
    plt.tight_layout()
    fig_path = "mlp_vs_bayes_ci95.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ----------------------
    # CSV summary export
    # ----------------------
    csv_path = "mlp_vs_bayes_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "Mean_Test_Error", "StdDev", "CI95", "Bayes_Error"])
        for N, m, s, c in zip(TRAIN_SIZES, means, stds, ci95):
            w.writerow([N, f"{m:.6f}", f"{s:.6f}", f"{c:.6f}", f"{bayes_err:.6f}"])
    print(f"\nSaved figure to {fig_path} and CSV to {csv_path}")

    # ----------------------
    # Confusion matrices (best model per N on the shared test set)
    # ----------------------
    for N in TRAIN_SIZES:
        model = perN_best_models[N]
        cm = confusion_matrix(model, X_test, y_test, num_classes=C)
        np.savetxt(f"confusion_N{N}.csv", cm, fmt="%d", delimiter=",")
    print("Saved per-N confusion matrices: confusion_N100.csv, ..., confusion_N10000.csv")

    # ----------------------
    # Console summary
    # ----------------------
    print("\n=== Summary (mean ± std over repeats) ===")
    for N, m, s in zip(TRAIN_SIZES, means, stds):
        print(f"N={N:>6}  →  Test P(error) = {m:.4f} ± {s:.4f}")

    print("\nChosen hidden sizes per repeat (per N):")
    for N, picks in chosen_Ps_summary.items():
        picked = [p for p, _ in picks]
        unique, counts = np.unique(picked, return_counts=True)
        combo = ", ".join([f"P={int(u)} x{int(c)}" for u, c in zip(unique, counts)])
        print(f"  N={N}: {combo}")

if __name__ == "__main__":
    run_experiment()
