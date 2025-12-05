import matplotlib.pyplot as plt
import numpy as np

P_list = [
    np.array([0.5, 0.5]),  # example (a)
    np.array([1.0, 0.0]),  # example (b)
    np.array([0.5, 0.5]),  # example (c)
    np.array([1.0, 0.0]),  # example (d)
    np.array([0.5, 0.5]),  # example (e)
    np.array([0.9, 0.1]),  # example (f)
]

Q_list = [
    np.array([1.0, 0.0]),  # example (a)
    np.array([0.5, 0.5]),  # example (b)
    np.array([0.5, 0.5]),  # example (c)
    np.array([0.0, 1.0]),  # example (d)
    np.array([0.9, 0.1]),  # example (e)
    np.array([0.5, 0.5]),  # example (f)
]

labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


def compute_PRD(P, Q, lambdas):
    precisions = []
    recalls = []
    for lam in lambdas:
        prec = np.sum(np.minimum(lam * P, Q))
        rec = np.sum(np.minimum(P, 1 / lam * Q))
        precisions.append(prec)
        recalls.append(rec)
    return np.array(precisions), np.array(recalls)


def plot_fig_2(output_file):
    fig, axes = plt.subplots(2, len(P_list), figsize=(16, 4))
    colors = ["#6f8ebf", "#77b986", "#cf7174", "#9a8ec1", "#d6c78f", "#83c3d7"]

    for i, (P, Q) in enumerate(zip(P_list, Q_list)):
        loc = [0, 1]

        j = 0
        axes[j, i].bar(
            loc, P, color=colors[i], edgecolor="black", linewidth=0.8, label="P"
        )
        axes[j, i].set_ylim(0, 1)
        axes[j, i].set_xticks([])
        axes[j, i].set_yticks([])
        for spine in ["left", "right", "top"]:
            axes[j, i].spines[spine].set_visible(False)
        if i == 0:
            axes[j, i].text(
                -0.2,
                0.5,
                r"$P$",
                transform=axes[j, i].transAxes,
                ha="center",
                va="center",
                fontsize=16,
            )

        j = 1
        axes[j, i].bar(
            loc, Q, color=colors[i], edgecolor="black", linewidth=0.8, label="Q"
        )
        axes[j, i].set_ylim(0, 1)
        axes[j, i].set_xticks([])
        axes[j, i].set_yticks([])
        for spine in ["left", "right", "top"]:
            axes[j, i].spines[spine].set_visible(False)
        if i == 0:
            axes[j, i].text(
                -0.2,
                0.5,
                r"$Q$",
                transform=axes[j, i].transAxes,
                ha="center",
                va="center",
                fontsize=16,
            )
        axes[j, i].text(
            0.5,
            -0.2,
            labels[j],
            transform=axes[j, i].transAxes,
            ha="center",
            va="top",
        )

    plt.tight_layout()

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()


def plot_fig_3(output_file):
    lin = np.linspace(0.000_001, 0.999_999, num=1_000)
    lambdas = lin / (1 - lin)

    fig, axes = plt.subplots(1, len(P_list), figsize=(16, 4))
    colors = ["#6f8ebf", "#77b986", "#cf7174", "#9a8ec1", "#d6c78f", "#83c3d7"]

    for i, (P, Q) in enumerate(zip(P_list, Q_list)):
        precision, recall = compute_PRD(P, Q, lambdas)
        axes[i].fill_between(recall, 0, precision, color=colors[i])
        axes[i].plot(recall, precision, color="black", linewidth=1.2)
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].tick_params(axis="x", length=0, pad=10, labelsize=12)
        axes[i].tick_params(axis="y", length=0, pad=10, labelsize=12)
        axes[i].set_xlabel(r"$\beta$", fontsize=14, labelpad=10)
        axes[i].xaxis.set_label_position("top")
        axes[i].set_aspect("equal", "box")

        if i == 0:
            axes[i].text(
                -0.3,
                0.5,
                r"$\alpha$",
                transform=axes[i].transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )

    plt.tight_layout()

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()
