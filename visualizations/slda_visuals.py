# schematic_highdim_cloud.py

import numpy as np
import matplotlib.pyplot as plt


def draw_arrow(ax, start, end, color="black", lw=1.5, mutation_scale=12):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=mutation_scale,
        ),
    )


def plot_highdim_cloud(
    save_path="schematic_highdim_cloud.pdf",
    seed=4,
):
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.set_aspect("equal")
    ax.axis("off")

    origin = np.array([0.0, 0.0])

    # ------------------------------------------------------------------
    # High-dimensional coordinate system
    # ------------------------------------------------------------------
    axes = [
        (np.array([0.0, 3.2]), r"$d_1$", (0.00, 0.24)),
        (np.array([3.4, 0.0]), r"$d_2$", (0.24, 0.02)),
        (np.array([2.4, -0.55]), r"$d_3$", (0.24, -0.02)),
        (np.array([1.75, -0.95]), r"$d_4$", (0.18, -0.12)),
        (np.array([-0.95, -1.35]), r"$d_{1024}$", (-0.34, -0.16)),
    ]

    for end, label, offset in axes:
        draw_arrow(
            ax,
            origin,
            end,
            color="black",
            lw=1.6,
            mutation_scale=13,
        )

        ax.text(
            end[0] + offset[0],
            end[1] + offset[1],
            label,
            fontsize=18,
            ha="center",
            va="center",
        )

    # omitted dimensions
    ax.text(
        0.35,
        -1.42,
        r"$\cdots$",
        fontsize=24,
        ha="center",
        va="center",
    )

    # ------------------------------------------------------------------
    # Point cloud
    # ------------------------------------------------------------------
    mean = np.array([0.75, 1.0])

    cov = np.array([
        [0.46, 0.14],
        [0.14, 0.34],
    ])

    X = rng.multivariate_normal(mean, cov, size=1100)

    ax.scatter(
        X[:, 0],
        X[:, 1],
        s=14,
        color="0.35",
        alpha=0.15,
        linewidths=0,
    )

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------
    ax.set_xlim(-2.35, 3.75)
    ax.set_ylim(-1.65, 3.45)

    fig.savefig(
        save_path,
        bbox_inches="tight",
        dpi=400,
    )

    plt.close(fig)


if __name__ == "__main__":
    plot_highdim_cloud()