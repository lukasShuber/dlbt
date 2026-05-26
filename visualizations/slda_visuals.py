import numpy as np
import matplotlib.pyplot as plt

COLORS = ["#C44E52", "#4C72B0", "#55A868"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_feature_space(path="panel_A_feature_space_clean.pdf", seed=4):
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.set_aspect("equal")
    ax.axis("off")

    # schematic high-dimensional axes
    origin = np.array([0.0, 0.0])
    axes = [
        ((0, 1.25), "dim 1"),
        ((1.35, 0.05), "dim 2"),
        ((0.85, -0.65), "dim 3"),
        ((-0.75, -0.75), "dim 4"),
        ((-1.05, 0.40), "dim 5"),
    ]

    for vec, lab in axes:
        v = np.array(vec)
        ax.arrow(0, 0, v[0], v[1], head_width=0.04, head_length=0.07,
                 length_includes_head=True, color="black", lw=1.5)
        ax.text(v[0]*1.08, v[1]*1.08, lab, ha="center", va="center", fontsize=9)

    ax.text(-0.15, -1.10, r"$\cdots$", fontsize=16)
    ax.text(-0.95, -1.18, "dim 1024", fontsize=9)

    # feature cloud
    X = rng.multivariate_normal([0.2, 0.15], [[0.22, 0.06], [0.06, 0.16]], 800)
    ax.scatter(X[:, 0], X[:, 1], s=8, color="0.35", alpha=0.22, lw=0)

    # linear decoder boundaries as cross-sections through feature space
    boundaries = [
        {"angle": 65,  "offset": -0.05, "color": COLORS[0], "label": "Task 1"},
        {"angle": -25, "offset": 0.08,  "color": COLORS[1], "label": "Task 2"},
        {"angle": 12,  "offset": -0.22, "color": COLORS[2], "label": r"Task $T$"},
    ]

    xx = np.linspace(-1.15, 1.35, 200)
    for b in boundaries:
        theta = np.deg2rad(b["angle"])
        m = np.tan(theta)
        yy = m * xx + b["offset"]

        ax.plot(xx, yy, color=b["color"], lw=2.2)
        ax.fill_between(xx, yy-0.10, yy+0.10, color=b["color"], alpha=0.16)

        xlab = 0.78
        ylab = m * xlab + b["offset"]
        ax.text(xlab+0.06, ylab+0.06, b["label"], color=b["color"],
                fontsize=10, fontweight="bold")

    ax.set_title(r"Schematic CLIP feature space $f(x)\in\mathbb{R}^{1024}$",
                 fontsize=12, pad=8)

    ax.set_xlim(-1.45, 1.55)
    ax.set_ylim(-1.35, 1.45)

    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_logistic_curves(path="panel_A_logistic_curves_clean.pdf"):
    specs = [
        ("Task 1", COLORS[0], 1.35, -0.6, r"$\eta_{x,1}=w_1^\top \tilde f_1(x)+b_1$"),
        ("Task 2", COLORS[1], 0.95,  0.1, r"$\eta_{x,2}=w_2^\top \tilde f_2(x)+b_2$"),
        (r"Task $T$", COLORS[2], 0.70, 0.75, r"$\eta_{x,T}=w_T^\top \tilde f_T(x)+b_T$"),
    ]

    eta = np.linspace(-4, 4, 500)
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.05), sharey=True)

    for ax, (title, color, slope, shift, formula) in zip(axes, specs):
        p = sigmoid(slope * (eta - shift))

        ax.plot(eta, p, color=color, lw=2.5)
        ax.axhline(0.5, color="0.65", ls="--", lw=0.8)
        ax.axvline(shift, color="0.65", ls="--", lw=0.8)

        ax.set_title(title, color=color, fontsize=10, fontweight="bold", pad=4)
        ax.text(0.5, 1.08, formula, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([-4, 0, 4])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(r"decoder logit $\eta_{x,t}$", fontsize=9)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$P(\mathrm{right}\mid x,t)$", fontsize=9)

    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_feature_space()
    plot_logistic_curves()