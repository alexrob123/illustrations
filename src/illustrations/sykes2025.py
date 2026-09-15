"""
A New Perspective on Precision and Recall for Generative Models
Benjamin Sykes (UNICAEN, ENSICAEN, GREYC), Loïc Simon (UNICAEN, ENSICAEN, GREYC), Julien Rabin (UNICAEN, ENSICAEN, GREYC), Jalal Fadili (UNICAEN, ENSICAEN, GREYC)
https://arxiv.org/abs/2511.02414v1
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

SEED = 0

CLF_COLORS = {
    "Cov": "#d61b1c",
    "GT": "#6e18bd",
    "iPR": "#1774b4",
    "KDE": "#14a014",
    "kNN": "#ff8820",
}

############
# Sampling #
############


def sample_gaussian(
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


def sample_gmm(
    means: list,
    P_weights: list,
    Q_weights: list,
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
    X = sample_mixture(n, P_weights)  # real
    Y = sample_mixture(n, Q_weights)  # fake

    return X, Y


##########
# Scores #
##########

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


def gaussian_score(
    z: np.ndarray,
    mu_p: np.ndarray,
    mu_q: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Bayes-optimal score q(z)/p(z).
    """
    dim = z.shape[1]

    p_z = multivariate_normal.pdf(z, mean=mu_p, cov=sigma * np.eye(dim))
    q_z = multivariate_normal.pdf(z, mean=mu_q, cov=sigma * np.eye(dim))

    return q_z / p_z


def log_likelihood_ratio_score_gmm(
    x: np.ndarray,
    means: np.ndarray,
    P_weights: np.ndarray,
    Q_weights: np.ndarray,
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

    log_p = logsumexp(exponand, b=P_weights, axis=1)
    log_q = logsumexp(exponand, b=Q_weights, axis=1)

    return log_p - log_q


def cov_score(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    As per experiments to match Fig 16, authors seem to not exlude self.
    """
    # manual flag for testing purposes
    allow_exclusion = False

    z_to_X_knn_radii = kth_nn_radius(
        z,
        X,
        k,
        exclude_self=allow_exclusion and np.array_equal(z, X),
    )
    z_to_Y_knn_radii = kth_nn_radius(
        z,
        Y,
        k,
        exclude_self=allow_exclusion and np.array_equal(z, Y),
    )

    n_X_in_Y_ball = count_points_in_ball(X, z, z_to_Y_knn_radii)
    n_Y_in_X_ball = count_points_in_ball(Y, z, z_to_X_knn_radii)

    return n_X_in_Y_ball, n_Y_in_X_ball


def ipr_score(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    As per experiments to match Fig 16, authors seem to not exlude self.
    """
    # manual flag for testing purposes
    allow_exclusion = False

    X_to_X_knn_radii = kth_nn_radius(X, X, k, exclude_self=allow_exclusion)
    Y_to_Y_knn_radii = kth_nn_radius(Y, Y, k, exclude_self=allow_exclusion)

    p_hat = count_balls_containing(z, X, X_to_X_knn_radii)
    q_hat = count_balls_containing(z, Y, Y_to_Y_knn_radii)

    return p_hat, q_hat


# def kde_score(
#     z: np.ndarray,
#     X: np.ndarray,
#     Y: np.ndarray,
#     k: int,
# ) -> np.ndarray:

#     # nn_X = NearestNeighbors(n_neighbors=k).fit(X)
#     # nn_Y = NearestNeighbors(n_neighbors=k).fit(Y)

#     # distances_X, _ = nn_X.kneighbors(X)
#     # distances_Y, _ = nn_Y.kneighbors(Y)
#     # knn_radii_X = distances_X[:, -1]  # dist from x to k-th neighbour in X
#     # knn_radii_Y = distances_Y[:, -1]  # dist from y to k-th neighbour in Y

#     # # rho_X = np.mean(knn_radii_X)  # single global bandwidth for P
#     # # rho_Y = np.mean(knn_radii_Y)  # single global bandwidth for Q
#     # bw = np.mean([knn_radii_X, knn_radii_Y])

#     # # distances from each z to every x / y
#     # dist_z_to_X, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
#     # dist_z_to_Y, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

#     # p_hat = (dist_z_to_X <= bw).sum(axis=1).astype(float)  # (|z|,)
#     # q_hat = (dist_z_to_Y <= bw).sum(axis=1).astype(float)  # (|z|,)

#     # return p_hat / (q_hat + 1e-12)

#     X_to_X_knn_radii = kth_nn_radius(X, X, k)
#     Y_to_Y_knn_radii = kth_nn_radius(Y, Y, k)

#     bandwidth = np.mean(np.concatenate([X_to_X_knn_radii, Y_to_Y_knn_radii]))

#     p_hat = count_points_in_ball(X, z, bandwidth)
#     q_hat = count_points_in_ball(Y, z, bandwidth)

#     return p_hat / (q_hat + 1e-12)


def kde_score(z, X, Y, k):
    # manual flag for testing purposes
    allow_exclusion = False

    X_to_X_knn_radii = kth_nn_radius(X, X, k, exclude_self=allow_exclusion)
    Y_to_Y_knn_radii = kth_nn_radius(Y, Y, k, exclude_self=allow_exclusion)

    bandwidth_X = np.mean(X_to_X_knn_radii)
    bandwidth_Y = np.mean(Y_to_Y_knn_radii)

    p_hat = count_points_in_ball(X, z, bandwidth_X)
    q_hat = count_points_in_ball(Y, z, bandwidth_Y)

    return p_hat, q_hat


def knn_score(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    As per experiments to match Fig 16, authors seem to not exlude self.
    """
    # manual flag for testing purposes
    allow_exclusion = False

    XY = np.vstack([X, Y])
    labels = np.array([0] * len(X) + [1] * len(Y))  # 0: X, 1: Y

    # exclude self
    self_query = np.array_equal(z, X) or np.array_equal(z, Y)
    n_neighbors = k + 1 if allow_exclusion and self_query else k

    # _, indices = kth_nearest_neighbours(z, XY, k)  # (|z|, k)

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(XY)
    _, indices = nn.kneighbors(z)  # (|z|, k)

    if allow_exclusion and self_query:
        indices = indices[:, 1:]

    neighbour_labels = labels[indices]
    n_X = (neighbour_labels == 0).sum(axis=1).astype(float)
    n_Y = (neighbour_labels == 1).sum(axis=1).astype(float)

    return n_X, n_Y


#############
# PR curves #
#############


def pr_curve_gaussian_GT(
    X: np.ndarray,
    Y: np.ndarray,
    mu_p: np.ndarray,
    mu_q: np.ndarray,
    lambdas: np.ndarray,
):
    # Bayes scores
    scores_X = gaussian_score(X, mu_p, mu_q)  # (N,)
    scores_Y = gaussian_score(Y, mu_p, mu_q)  # (N,)

    thresholds = np.concatenate([[0.0], lambdas, [np.inf]])  # (L,)
    fpr = (scores_X[:, None] < thresholds[None, :]).mean(axis=0)  # (L,)
    fnr = (scores_Y[:, None] >= thresholds[None, :]).mean(axis=0)  # (L,)

    alphas_interior = lambdas * fpr[1:-1] + fnr[1:-1]
    betas_interior = alphas_interior / lambdas

    alpha_inf = fnr[fpr == 0].min()
    beta_0 = fpr[fnr == 0].min()

    alphas = np.concatenate([[0.0], alphas_interior, [alpha_inf]])
    betas = np.concatenate([[beta_0], betas_interior, [0.0]])

    return alphas, betas


def pr_curve_GMM_GT(
    X: np.ndarray,
    Y: np.ndarray,
    means: np.ndarray,
    P_weights: np.ndarray,
    Q_weights: np.ndarray,
    lambdas: np.ndarray,
):
    scores_X = log_likelihood_ratio_score_gmm(X, means, P_weights, Q_weights)
    scores_Y = log_likelihood_ratio_score_gmm(Y, means, P_weights, Q_weights)

    thresholds = np.concatenate([[0.0], -np.log(lambdas), [np.inf]])
    fpr = (scores_X[:, None] < thresholds[None, :]).mean(axis=0)
    fnr = (scores_Y[:, None] >= thresholds[None, :]).mean(axis=0)

    alphas_interior = lambdas * fpr[1:-1] + fnr[1:-1]
    betas_interior = alphas_interior / lambdas

    alpha_inf = fnr[fpr == 0].min()
    beta_0 = fpr[fnr == 0].min()

    alphas = np.concatenate([[0.0], alphas_interior, [alpha_inf]])
    betas = np.concatenate([[beta_0], betas_interior, [0.0]])

    return alphas, betas


SCORE = {
    "Cov": cov_score,
    "iPR": ipr_score,
    "KDE": kde_score,
    "kNN": knn_score,
}


def classification(
    score_vs_X: np.ndarray, score_vs_Y: np.ndarray, gamma: float
) -> np.ndarray:
    if gamma == np.inf:
        # always real: FPR=0, FNR=1
        return np.ones(score_vs_X.shape[0], dtype=bool)
    elif gamma == 0.0:
        # never real: FPR=1, FNR=0
        return np.zeros(score_vs_X.shape[0], dtype=bool)
    elif gamma >= 1.0:
        return (gamma * score_vs_X) >= score_vs_Y
    elif gamma < 1.0:
        return (gamma * score_vs_X) > score_vs_Y
    else:
        raise (f"Warning: gamma={gamma} is not a valid threshold.")


def pr_curve(
    X: np.ndarray,
    Y: np.ndarray,
    lambdas: np.ndarray,
    clf: str,
    k: int,
    split: bool = False,
):
    if split:
        X_train, X_test = np.split(X, 2)
        Y_train, Y_test = np.split(Y, 2)
    else:
        X_train, X_test = X, X
        Y_train, Y_test = Y, Y

    if k == -1:
        k = int(np.sqrt(len(X_train)))
        print(f"Using k = sqrt(n) for {clf} (n={len(X_train)}, k={k})")

    # Matching scores
    scores_X_vs_X, scores_X_vs_Y = SCORE[clf](X_test, X_train, Y_train, k)  # (N,)
    scores_Y_vs_X, scores_Y_vs_Y = SCORE[clf](Y_test, X_train, Y_train, k)  # (N,)

    # Classification
    gammas = np.concatenate([[0.0], lambdas, [np.inf]])
    fpr = np.array(
        [1.0 - classification(scores_X_vs_X, scores_X_vs_Y, g).mean() for g in gammas]
    )
    fnr = np.array(
        [classification(scores_Y_vs_X, scores_Y_vs_Y, g).mean() for g in gammas]
    )

    # Precision/Recall
    risk = lambdas[:, None] * fpr[None, :] + fnr[None, :]  # (L, L)
    min_risk = risk.min(axis=1)  # (L,)

    # recall as lambda -> 0: min fpr among gammas achieving fnr == 0 (never empty: gamma=0 qualifies)
    zero_fnr = fnr == 0
    beta_at_0 = fpr[zero_fnr].min()

    # precision as lambda -> inf: min fnr among gammas achieving fpr == 0 (never empty: gamma=inf qualifies)
    zero_fpr = fpr == 0
    alpha_at_inf = fnr[zero_fpr].min()

    alphas = np.concatenate([[0.0], min_risk, [alpha_at_inf]])
    betas = np.concatenate([[beta_at_0], min_risk / lambdas, [0.0]])

    return alphas, betas


###########
# Figures #
###########


def plot_fig_2(out=None):
    """
    Comparing two shifted Gaussians
    """
    print("\n##### Fig 2 #####\n")

    # Randomness
    seed = 0
    rng = np.random.default_rng(seed)

    # Parameters
    n_GT = 100_000
    # n_runs = 10
    n_runs = 2
    n = 10_000
    dim = 64
    shifts = [1, 3]
    classifiers = ["Cov", "iPR", "KDE", "kNN"]
    k_values = [4, -1]  # to int(np.sqrt(n)) when -1
    split_values = [True, False]

    lambdas = np.tan(np.linspace(0, np.pi / 2, 152))[1:-1]

    # Sample data
    X_GT, Y_GT = defaultdict(dict), defaultdict(dict)
    X, Y = defaultdict(dict), defaultdict(dict)
    for shift in shifts:
        X_GT[shift], Y_GT[shift] = sample_gaussian(
            mu_p=np.zeros(dim),
            mu_q=np.ones(dim) * shift / np.sqrt(dim),
            dim=dim,
            n=n_GT,
            rng=rng,
        )
        X[shift], Y[shift] = sample_gaussian(
            mu_p=np.zeros(dim),
            mu_q=np.ones(dim) * shift / np.sqrt(dim),
            dim=dim,
            n=n,
            rng=rng,
        )

    # Ground Truth
    alphas_GT, betas_GT = {}, {}
    for shift in shifts:
        alphas_GT[shift], betas_GT[shift] = pr_curve_gaussian_GT(
            X_GT[shift],
            Y_GT[shift],
            mu_p=np.zeros(dim),
            mu_q=np.ones(dim) * shift / np.sqrt(dim),
            lambdas=lambdas,
        )

    # PR curves
    alphas = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    betas = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    pbar = tqdm(classifiers, desc="Computing PR curves", unit="classifier")
    for clf in pbar:
        pbar.set_postfix({"clf": clf})
        for k in k_values:
            for shift in shifts:
                for split in split_values:
                    alpha_interior_runs = []
                    alpha_inf_runs = []
                    beta_0_runs = []

                    for i in range(n_runs):
                        alphas_i, betas_i = pr_curve(
                            X[shift], Y[shift], lambdas, clf, k=k, split=split
                        )
                        alpha_interior_runs.append(alphas_i[1:-1])
                        alpha_inf_runs.append(alphas_i[-1])
                        beta_0_runs.append(betas_i[0])

                    alpha_interior_mean = np.mean(alpha_interior_runs, axis=0)
                    alpha_inf_mean = np.mean(alpha_inf_runs)
                    beta_0_mean = np.mean(beta_0_runs)

                    # safe because lambdas is fixed across runs, not resampled per run
                    beta_interior_mean = alpha_interior_mean / lambdas

                    alphas[clf][k][shift][split] = np.concatenate(
                        [[0.0], alpha_interior_mean, [alpha_inf_mean]]
                    )
                    betas[clf][k][shift][split] = np.concatenate(
                        [[beta_0_mean], beta_interior_mean, [0.0]]
                    )

    # Figure
    plt.subplots(2, 2, figsize=(8, 8))

    for i, split in enumerate(split_values):
        for j, k in enumerate(k_values):
            plt.subplot(2, 2, i * 2 + (j + 1))

            for shift_idx, shift in enumerate(shifts):
                plt.plot(
                    betas_GT[shift],
                    alphas_GT[shift],
                    label="GT" if shift_idx == 0 else None,
                    color=CLF_COLORS["GT"],
                    linestyle="--",
                    linewidth=2.0,
                )

                for clf in classifiers:
                    plt.plot(
                        betas[clf][k][shift][split],
                        alphas[clf][k][shift][split],
                        label=clf if shift_idx == 0 else None,
                        color=CLF_COLORS[clf],
                        linestyle="-",
                        linewidth=2.0,
                    )

            plt.xlim(0, 1.1)
            plt.ylim(0, 1.1)
            plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.xlabel(r"Recall ($\beta$)", fontsize=12)
            plt.ylabel(r"Precision ($\alpha$)", fontsize=12)
            if i * 2 + (j + 1) in [2, 3, 4]:
                plt.legend()

            k_text = r"\sqrt{n}" if k == -1 else str(k)
            plt.title(rf"$k={k_text},\ split={split}$", fontsize=12)

    plt.tight_layout()

    if out is None:
        plt.show()
    else:
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()


def plot_fig_16(out=None):
    """
    Comparing two Gaussian mixtures
    """
    print("\n##### Fig 16 #####\n")

    # Randomness
    seed = 0
    rng = np.random.default_rng(seed)

    # Parameters
    n_GT = 100_000
    n = 1_000
    dim = 64
    means = np.array([0, -5, 3, 5])
    P_weights = np.array([0.3, 0.2, 0.5, 0.0])
    Q_weights = np.array([0.0, 0.5, 0.2, 0.3])
    classifiers = ["Cov", "iPR", "KDE", "kNN"]
    k_values = [4, -1]  # to int(np.sqrt(n)) when -1
    split_values = [True, False]

    lambdas = np.tan(np.linspace(0, np.pi / 2, 152))[1:-1]

    # Sample data
    X_GT, Y_GT = sample_gmm(means, P_weights, Q_weights, dim, n_GT, rng)
    X, Y = sample_gmm(means, P_weights, Q_weights, dim, n, rng)

    # Ground Truth
    alphas_GT, betas_GT = pr_curve_GMM_GT(
        X_GT,
        Y_GT,
        means,
        P_weights,
        Q_weights,
        lambdas,
    )

    # Figure
    plt.subplots(2, 2, figsize=(8, 8))

    for i, split in enumerate(split_values):
        for j, k in enumerate(k_values):
            plt.subplot(2, 2, i * 2 + (j + 1))

            plt.plot(
                betas_GT,
                alphas_GT,
                label="GT",
                color=CLF_COLORS["GT"],
                linestyle="--",
                linewidth=2.0,
            )

            for clf in classifiers:
                alphas, betas = pr_curve(
                    X,
                    Y,
                    lambdas,
                    clf,
                    k,
                    split,
                )
                plt.plot(
                    betas,
                    alphas,
                    label=clf,
                    color=CLF_COLORS[clf],
                    linestyle="-",
                    linewidth=2.0,
                )

            plt.xlim(0, 1.1)
            plt.ylim(0, 1.1)
            plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.xlabel(r"Recall ($\beta$)", fontsize=12)
            plt.ylabel(r"Precision ($\alpha$)", fontsize=12)
            plt.legend()

            k_text = r"\sqrt{n}" if k == -1 else str(k)
            plt.title(rf"$k={k_text},\ split={split}$", fontsize=12)

    plt.tight_layout()

    if out is None:
        plt.show()
    else:
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()


if __name__ == "__main__":
    plot_fig_2("outputs/Sykes2025_Fig2.png")
    plot_fig_16("outputs/Sykes2025_Fig16.png")
