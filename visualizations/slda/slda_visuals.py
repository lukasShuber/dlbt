import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


COLORS = {
    "task1": "#1b9e77",
    "task2": "#d95f02",
    "taskT": "#7570b3",
}


def draw_arrow(ax, start, end, color="black", lw=1.4, mutation_scale=12, zorder=10):
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
        zorder=zorder,
    )


def add_translucent_plane(ax, pts, facecolor, edgecolor, alpha=0.18, zorder=1, lw=1.2):
    patch = Polygon(
        pts,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=lw,
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)


def plot_highdim_cloud_boundaries(
    save_path="visualizations/slda/clip_reps.png",
    seed=13,
):
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.set_aspect("equal")
    ax.axis("off")

    origin = np.array([0.0, 0.0])

# ------------------------------------------------------------------
# Translucent slicing planes
# ------------------------------------------------------------------
    def make_plane(center, u, v):
        center = np.array(center)
        u = np.array(u)
        v = np.array(v)
        return np.array([
            center - u - v,
            center + u - v,
            center + u + v,
            center - u + v,
        ])

    plane1 = make_plane(
        center=(0.55, 0.65),
        u=(2.35, 0.32),
        v=(0.22, 1.10),
    )

    plane2 = make_plane(
        center=(0.85, 0.95),
        u=(1.90, -0.82),
        v=(0.96, 0.28),
    )

    plane3 = make_plane(
        center=(0.40, 0.80),
        u=(1.30, 1.48),
        v=(-1.18, 0.32),
    )

    # add_translucent_plane(
    #     ax,
    #     plane1,
    #     facecolor="#7fc97f",
    #     edgecolor="#66c2a5",
    #     alpha=0.3,
    #     zorder=2,
    #     lw=1.15,
    # )

    # add_translucent_plane(
    #     ax,
    #     plane2,
    #     facecolor="#beaed4",
    #     edgecolor="#8da0cb",
    #     alpha=0.3,
    #     zorder=2,
    #     lw=1.10,
    # )

    # add_translucent_plane(
    #     ax,
    #     plane3,
    #     facecolor="#fdc086",
    #     edgecolor="#fc8d62",
    #     alpha=0.3,
    #     zorder=2,
    #     lw=1.10,
    # )

    # ------------------------------------------------------------------
    # Point cloud
    # ------------------------------------------------------------------
    mean = np.array([0.72, 0.95])
    cov = np.array([
        [0.55, 0.16],
        [0.16, 0.40],
    ])
    X = rng.multivariate_normal(mean, cov, size=1200)

    ax.scatter(
        X[:600, 0],
        X[:600, 1],
        s=26,
        color="0.35",
        alpha=0.17,
        linewidths=0,
        zorder=3,
    )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    axes = [
        (np.array([0.0, 3.15]), r"$d_1$", (0.00, 0.28)),
        (np.array([3.35, 0.0]), r"$d_2$", (0.28, 0.02)),
        (np.array([2.45, -0.55]), r"$d_3$", (0.28, -0.03)),
        (np.array([1.75, -0.95]), r"$d_4$", (0.22, -0.14)),
        (np.array([-0.90, -1.30]), r"$d_{1024}$", (-0.38, -0.18)),
    ]

    for end, label, offset in axes:
        draw_arrow(
            ax,
            origin,
            end,
            color="black",
            lw=1.45,
            mutation_scale=13,
            zorder=10,
        )
        ax.text(
            end[0] + offset[0],
            end[1] + offset[1],
            label,
            fontsize=18,
            ha="center",
            va="center",
            zorder=11,
        )

    # round-dot ellipsis
    ax.scatter(
        [0.18, 0.38, 0.58],
        [-1.42, -1.42, -1.42],
        s=15,
        color="black",
        zorder=11,
    )

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------
    ax.set_xlim(-2.55, 4.05)
    ax.set_ylim(-1.85, 3.65)

    fig.savefig(
        save_path,
        bbox_inches="tight",
        dpi=400,
    )
    plt.close(fig)


if __name__ == "__main__":
    plot_highdim_cloud_boundaries()