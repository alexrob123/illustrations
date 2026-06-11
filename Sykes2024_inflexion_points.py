from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from illustrations.Sykes2024 import (
    FONTSIZE,
    LAMBDAS,
    knn_score,
    pr_curve,
    sample_gauss_mix,
)

out = "./out-misc/"
Path(out).parent.mkdir(parents=True, exist_ok=True)

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.left"] = True
plt.rcParams["axes.spines.bottom"] = True
plt.rcParams["axes.grid"] = False
plt.rcParams["grid.alpha"] = 0.2
plt.rcParams["font.size"] = 16
plt.rcParams["legend.framealpha"] = 0.0
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["xaxis.labellocation"] = "center"
plt.rcParams["yaxis.labellocation"] = "center"
plt.rcParams["legend.fontsize"] = "x-small"
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern Roman"],
    }
)


# --------------------------------------------------------------------------------

num_components = 3
means = np.array(range(0, num_components)) * 2.75

X, Y = sample_gauss_mix(
    means=means,
    weights_P=[1 / len(means)] * len(means),
    weights_Q=[1 / len(means)] * len(means),
    dim=2,
    n=10_000,
)

plt.plot(X[:, 0], X[:, 1], label="P", alpha=0.5, linestyle="", marker="o", markersize=1)
plt.plot(Y[:, 0], Y[:, 1], label="Q", alpha=0.5, linestyle="", marker="s", markersize=1)
plt.legend()
plt.show()

# --------------------------------------------------------------------------------

plt.hist(X[:, 0], bins=100, density=True, alpha=0.5, label="P")
plt.hist(Y[:, 0], bins=100, density=True, alpha=0.5, label="Q")
plt.legend()
plt.show()

# --------------------------------------------------------------------------------

clf = "knn"

plt.figure()
for k in [2, 3, 4, 5, 10, 100]:
    alphas, betas = pr_curve(X, Y, lambdas=LAMBDAS, clf=clf, k=k)
    plt.plot(betas, alphas, label=f"k={k}")
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
plt.xlabel(r"Recall ($\beta_\lambda$)", fontsize=FONTSIZE)
plt.ylabel(r"Precision ($\alpha_\lambda$)", fontsize=FONTSIZE)
plt.title(f"PR curve for {clf} classifier", fontsize=FONTSIZE)
plt.legend()

if out is None:
    plt.show()
else:
    plt.savefig(
        out + "sykes2024-pr_curves.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
# --------------------------------------------------------------------------------

clf = "knn"
k = 5
alphas, betas = pr_curve(X, Y, lambdas=LAMBDAS, clf=clf, k=k)

# Consider the score 2/3
P_scores = knn_score(X, X, Y, k=k)  # (N,)
Q_scores = knn_score(Y, X, Y, k=k)  # (N,)
x = np.linspace(0, 1, 100)

plt.figure()
plt.plot(betas, alphas, label="PR curve")

for k_1 in range(0, k):
    k_1 = np.array([k_1])
    with np.errstate(divide="ignore"):
        thresh_1 = k_1 / (k - k_1)
    fpr_1 = (P_scores[:, None] < thresh_1).mean(axis=0)  # (L,)
    fnr_1 = (Q_scores[:, None] >= thresh_1).mean(axis=0)  # (L,)

    with np.errstate(divide="ignore"):
        thresh_2 = (k_1 + 1) / (k - (k_1 + 1))
    fpr_2 = (P_scores[:, None] < thresh_2).mean(axis=0)  # (L,)
    fnr_2 = (Q_scores[:, None] >= thresh_2).mean(axis=0)  # (L,)

    lam_star = -(fnr_2 - fnr_1) / (fpr_2 - fpr_1)
    plt.plot(x, lam_star * x)

plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
plt.xlabel(r"Recall ($\beta_\lambda$)", fontsize=FONTSIZE)
plt.ylabel(r"Precision ($\alpha_\lambda$)", fontsize=FONTSIZE)
plt.title(f"PR curve for {clf} classifier", fontsize=FONTSIZE)
plt.legend()

if out is None:
    plt.show()
else:
    plt.savefig(
        out + "sykes2024-inflexion_points.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

# --------------------------------------------------------------------------------
