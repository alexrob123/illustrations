"""
https://arxiv.org/abs/2405.01611
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances_chunked
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
    "cov",
    "ipr",
    "kde",
    "knn",
]
CLF_COLORS = {
    "gt": "#6e18bd",
    "cov": "#d61b1c",
    "ipr": "#1774b4",
    "kde": "#14a014",
    "knn": "#ff8820",
}

SHIFTS = [0, 1, 2, 3, 4]

N_GMM = 1_000
GMM_MEANS = np.array([0.0, -5.0, 3.0, 5.0])
GMM_WEIGHTS_P = np.array([0.3, 0.2, 0.5, 0.0])
GMM_WEIGHTS_Q = np.array([0.0, 0.5, 0.2, 0.3])


# Personal constants

# LAMBDAS = np.tan(np.linspace(0, np.pi / 2, 1_000 + 1, endpoint=False)[1:])
LAMBDAS = np.tan(np.linspace(0, np.pi / 2, 100 + 1, endpoint=True))[1:]
print(len(LAMBDAS))
# LAMBDAS[-1] = np.inf


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


# utils

WORKING_MEMORY = 512


def kth_nn_radius(
    query: np.ndarray,
    points: np.ndarray,
    k: int,
    exclude_self: bool = False,
    working_memory: int = WORKING_MEMORY,
) -> np.ndarray:
    """Radius of the ball at query[i] reaching its k-th NN among points. -> (|query|,)"""

    k = k + 1 if exclude_self else k

    out = np.empty(query.shape[0], dtype=np.float64)
    offset = 0
    for chunk in pairwise_distances_chunked(
        query,
        points,
        working_memory=working_memory,
    ):
        # partition = O(n) selection of the k smallest, no full sort needed
        part = np.partition(chunk, k - 1, axis=1)[:, :k]
        out[offset : offset + chunk.shape[0]] = part.max(axis=1)
        offset += chunk.shape[0]

    return out


def count_points_in_ball(
    query: np.ndarray,
    center: np.ndarray,
    radius: np.ndarray | float,
    working_memory: int = WORKING_MEMORY,
) -> np.ndarray:

    # scalar -> one radius per center, read-only view, no copy
    radius = np.broadcast_to(np.asarray(radius, dtype=np.float64), (center.shape[0],))

    out = np.zeros(center.shape[0], dtype=np.float64)
    offset = 0
    for chunk in pairwise_distances_chunked(
        center,
        query,
        working_memory=working_memory,
    ):
        r = radius[offset : offset + chunk.shape[0], None]
        out[offset : offset + chunk.shape[0]] = (chunk <= r).sum(axis=1)
        offset += chunk.shape[0]
    return out


def count_balls_containing(
    query: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    working_memory: int = WORKING_MEMORY,
) -> np.ndarray:

    assert centers.shape[0] == radii.shape[0]

    out = np.zeros(query.shape[0], dtype=np.float64)
    offset = 0
    for chunk in pairwise_distances_chunked(
        query,
        centers,
        working_memory=working_memory,
    ):
        out[offset : offset + chunk.shape[0]] = (chunk <= radii[None, :]).sum(axis=1)
        offset += chunk.shape[0]
    return out


def kth_nearest_neighbours(
    query: np.ndarray,
    points: np.ndarray,
    k: int,
    working_memory: int = WORKING_MEMORY,
):
    """k nearest points per query, sorted by distance. -> (distances, indices), each (|query|, k)"""
    distances = np.empty((query.shape[0], k), dtype=np.float64)
    indices = np.empty((query.shape[0], k), dtype=np.int64)
    offset = 0
    for chunk in pairwise_distances_chunked(
        query,
        points,
        working_memory=working_memory,
    ):
        part_idx = np.argpartition(chunk, k - 1, axis=1)[:, :k]
        part_dist = np.take_along_axis(chunk, part_idx, axis=1)
        order = np.argsort(part_dist, axis=1)
        n = chunk.shape[0]
        indices[offset : offset + n] = np.take_along_axis(part_idx, order, axis=1)
        distances[offset : offset + n] = np.take_along_axis(part_dist, order, axis=1)
        offset += n
    return distances, indices


# --------------------------------------------------------------------------------
# Cov
# --------------------------------------------------------------------------------


def cov_score(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> np.ndarray:

    z_to_X_knn_radii = kth_nn_radius(z, X, k)
    z_to_Y_knn_radii = kth_nn_radius(z, Y, k)

    n_X_in_Y_ball = count_points_in_ball(X, z, z_to_Y_knn_radii)
    n_Y_in_X_ball = count_points_in_ball(Y, z, z_to_X_knn_radii)

    return n_X_in_Y_ball / (n_Y_in_X_ball + 1e-12)


# --------------------------------------------------------------------------------
# iPR (adapative bandwidth Kernel Density Estimator)
# --------------------------------------------------------------------------------


def ipr_score(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> np.ndarray:

    X_to_X_knn_radii = kth_nn_radius(X, X, k)
    Y_to_Y_knn_radii = kth_nn_radius(Y, Y, k)

    p_hat = count_balls_containing(z, X, X_to_X_knn_radii)
    q_hat = count_balls_containing(z, Y, Y_to_Y_knn_radii)

    return p_hat / (q_hat + 1e-12)


# --------------------------------------------------------------------------------
# KDE (fixed bandwidth Kernel Density Estimator)
# --------------------------------------------------------------------------------


def kde_score(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> np.ndarray:

    # nn_X = NearestNeighbors(n_neighbors=k).fit(X)
    # nn_Y = NearestNeighbors(n_neighbors=k).fit(Y)

    # distances_X, _ = nn_X.kneighbors(X)
    # distances_Y, _ = nn_Y.kneighbors(Y)
    # knn_radii_X = distances_X[:, -1]  # dist from x to k-th neighbour in X
    # knn_radii_Y = distances_Y[:, -1]  # dist from y to k-th neighbour in Y

    # # rho_X = np.mean(knn_radii_X)  # single global bandwidth for P
    # # rho_Y = np.mean(knn_radii_Y)  # single global bandwidth for Q
    # bw = np.mean([knn_radii_X, knn_radii_Y])

    # # distances from each z to every x / y
    # dist_z_to_X, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
    # dist_z_to_Y, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

    # p_hat = (dist_z_to_X <= bw).sum(axis=1).astype(float)  # (|z|,)
    # q_hat = (dist_z_to_Y <= bw).sum(axis=1).astype(float)  # (|z|,)

    # return p_hat / (q_hat + 1e-12)

    X_to_X_knn_radii = kth_nn_radius(X, X, k)
    Y_to_Y_knn_radii = kth_nn_radius(Y, Y, k)

    bandwidth = np.mean(np.concatenate([X_to_X_knn_radii, Y_to_Y_knn_radii]))

    p_hat = count_points_in_ball(X, z, bandwidth)
    q_hat = count_points_in_ball(Y, z, bandwidth)

    return p_hat / (q_hat + 1e-12)


# --------------------------------------------------------------------------------
# kNN
# --------------------------------------------------------------------------------


def knn_score(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> np.ndarray:

    XY = np.vstack([X, Y])
    labels = np.array([0] * len(X) + [1] * len(Y))  # 0: X, 1: Y

    # for each z, count how many of its k nearest neighbours are real vs fake
    nn = NearestNeighbors(n_neighbors=k).fit(XY)
    _, indices = nn.kneighbors(z)  # (|z|, k)

    neighbour_labels = labels[indices]
    n_X = (neighbour_labels == 0).sum(axis=1).astype(float)
    n_Y = (neighbour_labels == 1).sum(axis=1).astype(float)

    return n_X / (n_Y + 1e-12)

    # XY = np.vstack([X, Y])
    # labels = np.array([0] * len(X) + [1] * len(Y))  # 0: X, 1: Y

    # _, indices = kth_nearest_neighbours(z, XY, k)  # (|z|, k)

    # neighbour_labels = labels[indices]
    # n_X = (neighbour_labels == 0).sum(axis=1).astype(float)
    # n_Y = (neighbour_labels == 1).sum(axis=1).astype(float)

    # return n_X / (n_Y + 1e-12)


# --------------------------------------------------------------------------------
# PR Curve
# --------------------------------------------------------------------------------


SCORE = {
    "cov": cov_score,
    "ipr": ipr_score,
    "kde": kde_score,
    "knn": knn_score,
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

    scores = np.unique(np.concatenate([P_scores, Q_scores]))
    if len(scores) > len(lambdas):
        thresholds = np.concatenate([[0.0], [np.inf], 1 / lambdas])
    else:
        thresholds = np.concatenate([[0.0], [np.inf], scores])

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

    # thresholds = np.concatenate([[np.inf], 1.0 / lambdas[1:-1], [0.0]])  # (L,)
    scores = np.unique(np.concatenate([P_scores, Q_scores]))
    if len(scores) > len(lambdas):
        thresholds = np.concatenate([[0.0], [np.inf], 1 / lambdas])
    else:
        thresholds = np.concatenate([[0.0], [np.inf], scores])
    # thresholds = 1 / lambdas  # (L,)
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
                return_params=clf == "gt",
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
                return_params=clf == "gt",
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

    classifiers = [] or CLF
    shifts = [1, 3]
    # n_runs = 10 # as per the paper
    n_runs = 1
    k_list = [4, max(1, int(np.sqrt(N)))]
    # k_list = [4]

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

    plt.subplots(1, len(k_list), sharey=True, figsize=(int(5 * len(k_list)), 5))
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
                # plt.fill_between(
                #     recall,
                #     mean_precision - std_precision,
                #     mean_precision + std_precision,
                #     color=CLF_COLORS[clf],
                #     alpha=0.15,
                # )
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
            r"$k=\sqrt{{n}}$" if k == max(1, int(np.sqrt(N))) else rf"$k={k}$",
            fontsize=FONTSIZE,
        )

    plt.suptitle("without split", fontsize=FONTSIZE)
    # plt.legend()
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
