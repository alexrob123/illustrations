"""
https://arxiv.org/abs/2405.01611
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

# Paper constants

N_GT, N, N_GMM = 100_000, 10_000, 1_000

DIM = 64
BIG_DIM = 2048

K = [4, max(1, int(np.sqrt(N)))]

CLF = [
    "gt",
    "ipr",
    "knn",
    "parzen",
    "coverage",
]
CLF_COLORS = {
    "gt": "#6e18bd",
    "ipr": "#1774b4",
    "knn": "#ff8820",
    "parzen": "#14a014",
    "coverage": "#d61b1c",
}

SHIFTS = [0, 1, 2, 3, 4]

N_GMM = 1_000
GMM_MEANS = np.array([0.0, -5.0, 3.0, 5.0])
GMM_WEIGHTS_P = np.array([0.3, 0.2, 0.5, 0.0])
GMM_WEIGHTS_Q = np.array([0.0, 0.5, 0.2, 0.3])


# Personal constants

LAMBDAS = np.tan(np.linspace(0, np.pi / 2, 1_000 + 1, endpoint=False)[1:])
SEED = 0

FONTSIZE = 12

# --------------------------------------------------------------------------------


def plot_thresholds_and_lambdas(thresholds, lambdas, log_lambdas=None):
    print(f"thresholds: {thresholds.min():.3f} to {thresholds.max():.3f}")
    print(f"lambdas: {lambdas.min():.3f} to {lambdas.max():.3f}")
    if log_lambdas is not None:
        print(f"log-lambdas: {log_lambdas.min():.3f} to {log_lambdas.max():.3f}")
    print("\n")

    plt.subplots(1, 2, figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.hist(thresholds, bins=50, color="#6f8ebf", label="Thresholds")

    if log_lambdas is not None:
        plt.hist(log_lambdas, bins=50, color="#cf7174", label="Log-Lambdas")
    plt.title(f"Number of thresholds: {len(thresholds)}")
    plt.ylabel("Frequency")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(lambdas, bins=25, color="#77b986", label="Lambdas")
    plt.title(f"Number of thresholds: {len(thresholds)}")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("outputs/thresholds_hist.png")


# --------------------------------------------------------------------------------
# Ground Truth
# --------------------------------------------------------------------------------


def log_likelihood_ratio_score(
    x: np.ndarray, mu_p: np.ndarray, mu_q: np.ndarray, sigma: float = 1.0
) -> np.ndarray:
    sq_p = np.sum((x - mu_p) ** 2, axis=1)
    sq_q = np.sum((x - mu_q) ** 2, axis=1)
    return (-0.5 / sigma**2) * (sq_q - sq_p)


def log_likelihood_ratio_score_gmm(
    x: np.ndarray,
    means: np.ndarray,
    weights_P: np.ndarray,
    weights_Q: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Bayes-optimal score log(p(z)/q(z)) for the GMM setup with shared means and isotropic
    unit variance.
    """
    dim = x.shape[1]

    # ||x - μ_l·1_d||² = ||x||² - 2·μ_l·Σx_i + d·μ_l² (vectorised over l)

    # x: (n, d)
    # x_norm: (n,) the squared norm of each x (||x||²)
    # x_sum: (n,) the sum of each x (Σ_1^d x_i)
    # means: (K,) the means of each component
    # exponand: (n, K) the exponand for each x and each mean

    x_norm = np.sum(x**2, axis=1)
    x_sum = np.sum(x, axis=1)

    norm = (
        x_norm[:, None]
        - 2.0 * means[None, :] * x_sum[:, None]
        + dim * means[None, :] ** 2
    )
    exponand = -(1 / 2) / sigma**2 * norm

    log_p = logsumexp(exponand, b=weights_P, axis=1)
    log_q = logsumexp(exponand, b=weights_Q, axis=1)

    return log_p - log_q


def gt_score(
    z: np.ndarray,
    mu_p: np.ndarray,
    mu_q: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Bayes-optimal score p(z)/q(z).
    """
    dim = z.shape[1]
    assert dim == DIM or dim == BIG_DIM

    p_z = multivariate_normal.pdf(z, mean=mu_p, cov=sigma * np.eye(dim))
    q_z = multivariate_normal.pdf(z, mean=mu_q, cov=sigma * np.eye(dim))

    return p_z / q_z


# --------------------------------------------------------------------------------
# Coverage
# (Naem 2020)
# https://arxiv.org/abs/2002.09797
# --------------------------------------------------------------------------------


def coverage_score(z: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    """
    For each probe z:
      - compute radius r_Y(z) = dist to k-th NN of z in Y
      - compute radius r_X(z) = dist to k-th NN of z in X
      - score = #{x∈X inside B^Y_kNN(z)} / #{y∈Y inside B^X_kNN(z)}
    """

    nn_X = NearestNeighbors(n_neighbors=k).fit(X)
    nn_Y = NearestNeighbors(n_neighbors=k).fit(Y)

    dist_z_to_X, _ = nn_X.kneighbors(z)
    dist_z_to_Y, _ = nn_Y.kneighbors(z)
    r_X = dist_z_to_X[:, -1]  # dist from z to k-th neighbor in X
    r_Y = dist_z_to_Y[:, -1]  # dist from z to k-th neighbor in Y

    # distance from every x∈X to every z, and every y∈Y to every z
    dist_X_to_z, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
    dist_Y_to_z, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

    # #{x∈X : dist(x,z) <= r_Y(z)}
    # #{y∈Y : dist(y,z) <= r_X(z)}
    n_X_in_Y_ball = (dist_X_to_z <= r_Y[:, None]).sum(axis=1).astype(float)
    n_Y_in_X_ball = (dist_Y_to_z <= r_X[:, None]).sum(axis=1).astype(float)

    return n_X_in_Y_ball / (n_Y_in_X_ball + 1e-12)


# --------------------------------------------------------------------------------
# Improved PR (adapative bandwidth Kernel Density Estimator)
# (Kynkaanniemi 2019)
# https://arxiv.org/abs/1904.06991
# --------------------------------------------------------------------------------


def ipr_score(z: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    """
    For each probe point z, compute p̂(z) and q̂(z) via manifold indicator,
    then return the log-ratio score (monotone in p̂/q̂, avoids division by zero).
    """

    nn_X = NearestNeighbors(n_neighbors=k).fit(X)
    nn_Y = NearestNeighbors(n_neighbors=k).fit(Y)

    distances_X, _ = nn_X.kneighbors(X)
    distances_Y, _ = nn_Y.kneighbors(Y)
    knn_radii_X = distances_X[:, -1]  # dist from x to k-th neighbour in X
    knn_radii_Y = distances_Y[:, -1]  # dist from y to k-th neighbour in Y

    # distances from each z to every x / y
    dist_z_to_X, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
    dist_z_to_Y, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

    p_hat = (dist_z_to_X <= knn_radii_X[None, :]).sum(axis=1).astype(float)
    q_hat = (dist_z_to_Y <= knn_radii_Y[None, :]).sum(axis=1).astype(float)

    return p_hat / (q_hat + 1e-12)


# def ipr_score(
#     z: np.ndarray,  # (|z|, d)
#     X: np.ndarray,  # (|X|, d)
#     Y: np.ndarray,  # (|Y|, d)
#     k: int,
#     n_jobs: int = 8,
# ) -> np.ndarray:
#     """
#     For each probe point z, compute p̂(z) and q̂(z) via manifold indicator,
#     then return the score (monotone in p̂/q̂, avoids division by zero).
#     """

#     nn_X = NearestNeighbors(
#         n_neighbors=k + 1,  # do not count itself
#         algorithm="ball_tree",  # tree indexing
#         n_jobs=n_jobs,
#     ).fit(X)
#     nn_Y = NearestNeighbors(
#         n_neighbors=k + 1,  # do not count itself
#         algorithm="ball_tree",  # tree indexing
#         n_jobs=n_jobs,
#     ).fit(Y)

#     distances_X, _ = nn_X.kneighbors(X)
#     distances_Y, _ = nn_Y.kneighbors(Y)
#     knn_radii_X = distances_X[:, -1]  # dist from x to k-th neighbour in X, (|X|,)
#     knn_radii_Y = distances_Y[:, -1]  # dist from y to k-th neighbour in Y, (|Y|,)

#     # distances from each z to every x / y
#     # dist_z_to_X, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
#     # dist_z_to_Y, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

#     # Replace kneighbors(n_neighbors=len(X)) with radius_neighbors for efficiency
#     # use global max radius for parallelism, then filter per point
#     dists_z_X, idx_z_X = nn_X.radius_neighbors(
#         z,
#         radius=knn_radii_X.max(),
#         return_distance=True,
#     )
#     dists_z_Y, idx_z_Y = nn_Y.radius_neighbors(
#         z,
#         radius=knn_radii_Y.max(),
#         return_distance=True,
#     )

#     # Count matches via per-point radius filtering
#     n_z = len(z)
#     p_hat = np.zeros(n_z, dtype=np.float64)  # (|z|,)
#     q_hat = np.zeros(n_z, dtype=np.float64)  # (|z|,)

#     for i in range(n_z):
#         # keep only x's where dist <= that x's own radius (the manifold indicator)
#         p_hat[i] = (dists_z_X[i] <= knn_radii_X[idx_z_X[i]]).sum()
#         q_hat[i] = (dists_z_Y[i] <= knn_radii_Y[idx_z_Y[i]]).sum()

#     return p_hat / (q_hat + 1e-12)  # (|z|,)


# --------------------------------------------------------------------------------
# KNN (KNN Classifier)
# (Park & Kim 2023)
# https://arxiv.org/abs/2309.01590
# --------------------------------------------------------------------------------


def knn_score(z: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    """
    For each probe z, find k nearest neighbours in X∪Y.
    Score = (# that come from X) - (# that come from Y).
    High score → neighbourhood is mostly real → z is real-like.
    """
    XY = np.vstack([X, Y])
    labels = np.array([0] * len(X) + [1] * len(Y))  # 0: X, 1: Y

    # for each z, count how many of its k nearest neighbours are real vs fake
    nn = NearestNeighbors(n_neighbors=k).fit(XY)
    _, indices = nn.kneighbors(z)  # (|z|, k)

    neighbour_labels = labels[indices]
    n_X = (neighbour_labels == 0).sum(axis=1).astype(float)
    n_Y = (neighbour_labels == 1).sum(axis=1).astype(float)

    return n_X / (n_Y + 1e-12)


# --------------------------------------------------------------------------------
# Parzen (fixed bandwidth Kernel Density Estimator)
# --------------------------------------------------------------------------------


def parzen_score(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Fixed-bandwidth Parzen classifier.
    Same as IPR but with a single global radius per dataset
    instead of per-sample adaptive radii.
    ρ_X = mean kNN radius over X
    ρ_Y = mean kNN radius over Y
    """

    nn_X = NearestNeighbors(n_neighbors=k).fit(X)
    nn_Y = NearestNeighbors(n_neighbors=k).fit(Y)

    distances_X, _ = nn_X.kneighbors(X)
    distances_Y, _ = nn_Y.kneighbors(Y)
    knn_radii_X = distances_X[:, -1]  # dist from x to k-th neighbour in X
    knn_radii_Y = distances_Y[:, -1]  # dist from y to k-th neighbour in Y

    rho_X = np.mean(knn_radii_X)  # single global bandwidth for P
    rho_Y = np.mean(knn_radii_Y)  # single global bandwidth for Q

    # distances from each z to every x / y
    dist_z_to_X, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
    dist_z_to_Y, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

    p_hat = (dist_z_to_X <= rho_X).sum(axis=1).astype(float)  # (|z|,)
    q_hat = (dist_z_to_Y <= rho_Y).sum(axis=1).astype(float)  # (|z|,)

    return p_hat / (q_hat + 1e-12)


# --------------------------------------------------------------------------------
# PR Curve
# --------------------------------------------------------------------------------


SCORE = {
    "cov": coverage_score,
    "ipr": ipr_score,
    "knn": knn_score,
    "kde": parzen_score,
}


def pr_curve(
    X: np.ndarray,
    Y: np.ndarray,
    lambdas: np.ndarray,
    clf: str,
    k: int,
):
    P_scores = SCORE[clf](X, X, Y, k)  # (N,)
    Q_scores = SCORE[clf](Y, X, Y, k)  # (N,)

    thresholds = 1.0 / lambdas  # (L,)
    fpr = (P_scores[:, None] < thresholds[None, :]).mean(axis=0)  # (L,)
    fnr = (Q_scores[:, None] >= thresholds[None, :]).mean(axis=0)  # (L,)

    risk = lambdas[:, None] * fpr[None, :] + fnr[None, :]  # (L, L)
    precisions = risk.min(axis=1)  # (L,)
    recalls = precisions / lambdas  # (L,)

    return precisions, recalls


# def gt_pr_curve(
#     X: np.ndarray,
#     Y: np.ndarray,
#     mu_p: np.ndarray,
#     mu_q: np.ndarray,
#     lambdas: np.ndarray,
# ):
#     # Bayes scores
#     P_log_scores = log_likelihood_ratio_score(X, mu_p, mu_q)
#     Q_log_scores = log_likelihood_ratio_score(Y, mu_p, mu_q)

#     thresholds = -np.log(lambdas)  # (L,)
#     fpr = (P_log_scores[:, None] >= thresholds[None, :]).mean(axis=0)  # (L,)
#     fnr = (Q_log_scores[:, None] < thresholds[None, :]).mean(axis=0)  # (L,)

#     risk = lambdas[:, None] * fpr[None, :] + fnr[None, :]  # (L, L)
#     precisions = risk.min(axis=1)  # (L,)
#     recalls = precisions / lambdas  # (L,)

#     return precisions, recalls


def gt_pr_curve(
    X: np.ndarray,
    Y: np.ndarray,
    mu_p: np.ndarray,
    mu_q: np.ndarray,
    lambdas: np.ndarray,
):
    # Bayes scores
    P_scores = gt_score(X, mu_p, mu_q)  # (N,)
    Q_scores = gt_score(Y, mu_p, mu_q)  # (N,)

    thresholds = 1.0 / lambdas  # (L,)
    fpr = (P_scores[:, None] < thresholds[None, :]).mean(axis=0)  # (L,)
    fnr = (Q_scores[:, None] >= thresholds[None, :]).mean(axis=0)  # (L,)

    risk = lambdas[:, None] * fpr[None, :] + fnr[None, :]  # (L, L)
    precisions = risk.min(axis=1)  # (L,)
    recalls = precisions / lambdas  # (L,)

    return precisions, recalls


def gt_pr_curve_for_gmm(
    X: np.ndarray,
    Y: np.ndarray,
    means: np.ndarray,
    weights_P: np.ndarray,
    weights_Q: np.ndarray,
    lambdas: np.ndarray,
):
    scores_for_P = log_likelihood_ratio_score_gmm(X, means, weights_P, weights_Q)
    scores_for_Q = log_likelihood_ratio_score_gmm(Y, means, weights_P, weights_Q)

    # precisions, recalls = [], []
    # for lam in lambdas:
    #     fpr = (scores_for_P < -np.log(lam)).mean()
    #     fnr = (scores_for_Q >= -np.log(lam)).mean()

    all_scores = np.concatenate([scores_for_P, scores_for_Q])
    thresholds = np.linspace(
        all_scores.max() + 0.1,
        all_scores.min() - 0.1,
        len(lambdas),
    )

    precisions, recalls = [], []
    for thresh in thresholds:
        fpr = (scores_for_P < thresh).mean()
        fnr = (scores_for_Q >= thresh).mean()

        lam = np.exp(-thresh)

        precision = lam * fpr + fnr
        recall = 1 / lam * precision

        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


# --------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------


def sample_gauss(
    mu_p: np.ndarray,
    mu_q: np.ndarray,
    dim: int,
    n: int,
    rng: np.random.Generator = None,
    return_params: bool = False,
) -> tuple:

    if rng is None:
        rng = np.random.default_rng(SEED)

    # Sample from P and Q
    dim = len(mu_p)
    X = rng.normal(loc=mu_p, scale=1.0, size=(n, dim))  # real
    Y = rng.normal(loc=mu_q, scale=1.0, size=(n, dim))  # fake

    return (X, Y, mu_p, mu_q) if return_params else (X, Y)


def sample_gauss_mix(
    means: list,
    weights_P: list,
    weights_Q: list,
    dim: int,
    n: int,
    rng: np.random.Generator = None,
    return_params: bool = False,
) -> tuple:

    if rng is None:
        rng = np.random.default_rng(SEED)

    def sample_mixture(n, weights):
        # 1. draw component assignments
        components = rng.choice(len(means), size=n, p=weights)
        # 2. sample from the chosen Gaussian for each point
        samples = np.array(
            [rng.normal(loc=means[c], scale=1.0, size=dim) for c in components]
        )
        return samples

    # Sample from P and Q
    X = sample_mixture(n, weights_P)  # real
    Y = sample_mixture(n, weights_Q)  # fake

    return (X, Y, means, weights_P, weights_Q) if return_params else (X, Y)


# --------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------


def run_xp(clf, n_runs, shift, dim, k, split=0):
    rng = np.random.default_rng(SEED)

    run_precisions = []

    for _ in tqdm(range(n_runs)):
        pr_curve_func = gt_pr_curve if clf == "gt" else pr_curve
        precisions, _ = pr_curve_func(
            *sample_gauss(
                mu_p=np.zeros(dim),
                mu_q=np.ones(dim) * shift / np.sqrt(dim),
                dim=dim,
                n=N_GT if clf == "gt" else N,
                rng=rng,
                return_params=True if clf == "gt" else False,
            ),
            lambdas=LAMBDAS,
            **({"clf": clf.lower(), "k": k} if clf != "gt" else {}),
        )

        run_precisions.append(precisions)

    mean_precision = np.mean(run_precisions, axis=0)
    std_precision = np.std(run_precisions, axis=0)

    return mean_precision, std_precision


def run_xp_with_gmm(clf, n_runs, k, split=0):
    rng = np.random.default_rng(SEED)

    run_precisions = []

    for _ in tqdm(range(n_runs)):
        pr_curve_func = gt_pr_curve_for_gmm if clf == "gt" else pr_curve
        precisions, recalls = pr_curve_func(
            *sample_gauss_mix(
                means=GMM_MEANS,
                weights_P=GMM_WEIGHTS_P,
                weights_Q=GMM_WEIGHTS_Q,
                dim=DIM,
                n=N_GT if clf == "gt" else N_GMM,
                rng=rng,
                return_params=True if clf == "gt" else False,
            ),
            lambdas=LAMBDAS,
            **({"clf": clf.lower(), "k": k} if clf != "gt" else {}),
        )

        run_precisions.append(precisions)

    mean_precision = np.mean(run_precisions, axis=0)
    std_precision = np.std(run_precisions, axis=0)

    return mean_precision, std_precision


# --------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------


def plot_fig_2(out=None):
    """
    FIG 2 - Comparing two shifted Gaussians.
    """

    print("-" * 60)
    print(plot_fig_2.__doc__)
    print("-" * 60)

    classifiers = ["gt", "ipr", "knn", "parzen", "coverage"] or CLF
    shifts = [1, 3] or SHIFTS
    n_runs = 10
    k_list = [4, max(1, int(np.sqrt(N)))] or K

    results = defaultdict(dict)
    for k in k_list:
        print(f"\nRunning with k={k}...")
        for shift in shifts:
            print(f"\tRunning with shift {shift}...")
            results[k][shift] = dict()
            for clf in classifiers:
                print(f"\t\tClassifier: {clf}")
                results[k][shift][clf] = run_xp(
                    clf=clf,
                    n_runs=n_runs,
                    shift=shift,
                    dim=DIM,
                    k=k,
                )

    # Plot.

    plt.subplots(1, len(k_list), sharey=True, figsize=(int(4 * len(shifts)), 6))
    for i_k, k in enumerate(k_list):
        plt.subplot(1, len(k_list), i_k + 1)
        for i_shift, shift in enumerate(shifts):
            for clf in classifiers:
                mean_precision, std_precision = results[k][shift][clf]
                precision, recall = mean_precision, mean_precision / LAMBDAS
                plt.plot(
                    recall,
                    precision,
                    color=CLF_COLORS[clf],
                    linestyle="--" if clf == "gt" else "-",
                    linewidth=1.5,
                    label=f"{clf.upper()}",
                )
                plt.fill_between(
                    recall,
                    mean_precision - std_precision,
                    mean_precision + std_precision,
                    color=CLF_COLORS[clf],
                    alpha=0.15,
                )
            if i_shift == 0:
                plt.legend(loc="upper right")
        if i_k == 0:
            plt.ylabel(r"Precision ($\alpha$)", fontsize=FONTSIZE)
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xlabel(r"Recall ($\beta$)", fontsize=FONTSIZE)
        plt.title(
            r"$k=\sqrt{{N}}$" if k == max(1, int(np.sqrt(N))) else rf"$k={k}$",
            fontsize=FONTSIZE,
        )

    plt.suptitle("without split", fontsize=FONTSIZE)
    plt.legend()
    plt.tight_layout()

    if out is None:
        plt.show()
    else:
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()


def plot_fig_3(out=None):
    """
    FIG 3 - Comparing two Gaussian Mixtures.
    """

    print("-" * 60)
    print(plot_fig_3.__doc__)
    print("-" * 60)

    classifiers = [] or CLF
    n_runs = 10

    results = dict()
    for clf in classifiers:
        print(f"Classifier: {clf}")
        results[clf] = run_xp_with_gmm(
            clf,
            n_runs=n_runs,
            k=K[1],
        )

    # Plot.

    plt.figure(figsize=(6, 6))
    for clf in classifiers:
        mean_precision, std_precision = results[clf]
        precision, recall = mean_precision, mean_precision / LAMBDAS
        plt.plot(
            recall,
            precision,
            color=CLF_COLORS[clf],
            linestyle="--" if clf == "gt" else "-",
            linewidth=1.5,
            label=f"{clf.upper()}",
        )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel(r"Precision ($\alpha$)", fontsize=FONTSIZE)
    plt.xlabel(r"Recall ($\beta$)", fontsize=FONTSIZE)
    plt.legend()
    plt.tight_layout()

    if out is None:
        plt.show()
    else:
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()


def plot_fig_4(out=None):
    """
    FIG 4 - PR curves in high dimension.
    """

    print("-" * 60)
    print(plot_fig_4.__doc__)
    print("-" * 60)

    classifiers = [] or CLF
    shifts = [1, 3] or SHIFTS
    n_runs = 10

    results = dict()
    for shift in shifts:
        print(f"Running shift {shift}...")
        results[shift] = dict()
        for clf in classifiers:
            print(f"\tClassifier: {clf}")
            results[shift][clf] = run_xp(
                clf=clf,
                n_runs=n_runs,
                shift=shift,
                dim=BIG_DIM,
                k=K[1],
            )

    # Plot.

    plt.figure(figsize=(8, 8))
    for i, shift in enumerate(shifts):
        for clf in classifiers:
            mean_precision, std_precision = results[shift][clf]
            precision, recall = mean_precision, mean_precision / LAMBDAS
            plt.plot(
                recall,
                precision,
                color=CLF_COLORS[clf],
                linestyle="--" if clf == "gt" else "-",
                linewidth=1.5,
                # only label the first shift to avoid duplicate legend entries
                label=f"{clf.upper()}" if i == 0 else None,
            )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlabel(r"Recall ($\beta$)", fontsize=FONTSIZE)
    plt.ylabel(r"Precision ($\alpha$)", fontsize=FONTSIZE)
    plt.title(rf"$\mu = {shift / np.sqrt(BIG_DIM):.2f}$")
    plt.legend(loc="upper right")
    plt.tight_layout()

    if out is None:
        plt.show()
    else:
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()


if __name__ == "__main__":
    plot_fig_2("outputs/Sykes2024-Fig2.png")
    # plot_fig_3("outputs/Sykes2024-Fig3.png")
    # plot_fig_4("outputs/Sykes2024-Fig4.png")
