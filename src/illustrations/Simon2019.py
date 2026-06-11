import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def synthetic_norm_pdf(x, mean=0.0, sigma=1.0, eps=0.0):
    if np.isscalar(x):
        x = np.array([x])
    else:
        x = np.asarray(x)

    y = norm.pdf(x, loc=mean, scale=sigma)
    y_shifted = y - eps

    mask = y_shifted > 0
    x_positive = x[mask]
    y_positive = y_shifted[mask]

    return x_positive, y_positive


def synthetic_mixture_pdf(x, means, sigmas, eps=0.0, weights=None):
    x = np.asarray(x)
    n_components = len(means)

    if weights is None:
        weights = np.ones(n_components) / n_components
    else:
        weights = np.asarray(weights)
        weights = weights / weights.sum()  # normalize

    y_total = np.zeros_like(x, dtype=float)

    for mean, sigma, weight in zip(means, sigmas, weights):
        y_total += weight * norm.pdf(x, loc=mean, scale=sigma)

    y_shifted = y_total - eps
    mask = y_shifted > 0
    x_positive = x[mask]
    y_positive = y_shifted[mask]

    return x_positive, y_positive


def sample_synthetic_norm(
    x_min, x_max, mean=0.0, sigma=1.0, eps=0.0, n_samples=100_000
):
    y = np.array([])

    while len(y) < n_samples:
        x = np.linspace(x_min, x_max, n_samples)

        x, y = synthetic_norm_pdf(x, mean=mean, sigma=sigma, eps=eps)

        x_min = x.min()
        x_max = x.max()

    return x, y


def sample_synthetic_mixture(
    x_min, x_max, means, sigmas, weights, eps=0.0, n_samples=100_000
):
    y = np.array([])

    while len(y) < n_samples:
        x = np.linspace(x_min, x_max, n_samples)

        x, y = synthetic_mixture_pdf(
            x, means=means, sigmas=sigmas, weights=weights, eps=eps
        )

        x_min = x.min()
        x_max = x.max()

    return x, y


x_P_left, P_left = sample_synthetic_mixture(
    -10,
    10,
    n_samples=100_000,
    means=[1.5],
    sigmas=[1.0],
    weights=[1.0],
    eps=0.02,
)
x_P_mid, P_mid = sample_synthetic_mixture(
    -10,
    10,
    n_samples=100_000,
    means=[-2.5, 1.3],
    sigmas=[1.0, 1.2],
    weights=[0.3, 0.7],
    eps=0.02,
)
x_P_right, P_right = sample_synthetic_mixture(
    -10,
    10,
    n_samples=100_000,
    means=[-3.0, 0.0, 1.75],
    sigmas=[0.75, 0.5, 0.6],
    weights=[0.6, 0.15, 0.25],
    eps=0.02,
)


x_Q_left, Q_left = x_P_mid, P_mid
x_Q_mid, Q_mid = x_P_left, P_left
x_Q_right, Q_right = sample_synthetic_mixture(
    -10,
    10,
    n_samples=100_000,
    means=[-0.1, 1.5, 3.5],
    sigmas=[0.55, 0.5, 0.35],
    weights=[0.39, 0.36, 0.25],
    eps=0.02,
)


def plot_density(ax, x, y, color, label):
    ax.fill_between(x, y, color=color)
    ax.plot(x, y, color="k", linewidth=0.6)
    ax.axhline(y=0, color="black", linewidth=1.0)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(
        -5,
        0.1,
        label,
        fontsize=12,
        va="top",
    )


def plot_fig_1(output_file):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1], hspace=0.3, wspace=0.1)

    # Top two rows: densities
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])

    plot_density(ax00, x_P_left, P_left, color="#51a7f9", label=r"$P$")
    plot_density(ax10, x_Q_left, Q_left, color="#51a7f9", label=r"$Q$")

    plot_density(ax01, x_P_mid, P_mid, color="#70bf41", label=r"$P$")
    plot_density(ax11, x_Q_mid, Q_mid, color="#70bf41", label=r"$Q$")

    plot_density(ax02, x_P_right, P_right, color="#ec5d57", label=r"$P$")
    plot_density(ax12, x_Q_right, Q_right, color="#ec5d57", label=r"$Q$")

    for ax in [ax00, ax01, ax02, ax10, ax11, ax12]:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- bottom row: precision-recall stylized panels ----------------------------
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax22 = fig.add_subplot(gs[2, 2])

    for ax in (ax20, ax21, ax22):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall " + r"$\beta$", fontsize=11)
        ax.set_ylabel("Precision " + r"$\alpha$", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # TODO
    ax20.fill_between([0, 1], 0, 0.7, color="#51a7f9")
    ax21.fill_between([0, 0.75], 0, 1.0, color="#70bf41")

    r = np.linspace(0, 1, 400)
    precision_curve = 1.0 - 0.85 * (np.clip((r - 0.15) / 0.85, 0, 1.0)) ** 0.9
    precision_curve -= 0.12 * np.exp(-((r - 0.45) ** 2) / (2 * 0.12**2))
    precision_curve = np.clip(precision_curve, 0, 1)
    ax22.fill_between(r, 0, precision_curve, color="#ec5d57")
    ax22.plot(r, precision_curve, color="k", linewidth=0.8)

    plt.tight_layout()

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()
