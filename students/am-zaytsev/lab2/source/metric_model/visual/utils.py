from collections import defaultdict

import matplotlib.pyplot as plt


def draw_loo_res(loo_res):
    # Group by color key
    groups = defaultdict(list)
    for i in loo_res:
        x, group = i[0][0], i[0][1]
        groups[group].append((x, i[-1]))

    # Sort points within each group by x
    groups = {g: sorted(pts) for g, pts in groups.items()}

    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, (group, pts) in enumerate(groups.items()):
        xs, ys = zip(*pts)
        if idx == 0:
            ax.plot(
                xs, ys, marker="o", markersize=0, linewidth=2, alpha=1.0, label=group
            )
        else:
            ax.plot(
                xs,
                ys,
                linestyle="--",
                marker="o",
                markersize=0,
                linewidth=1.2,
                alpha=0.35,
                label=group,
            )

    # --- Y range from the first group only ---
    first_ys = list(groups.values())[0]  # [(x, y), ...] of group 1
    first_ys = [y for _, y in first_ys]
    y_min, y_max = min(first_ys), max(first_ys)

    # Optional: pad a little so markers aren't clipped
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    ax.set_ylim(y_min - pad, y_max + pad)
    # Or with zero padding:
    # ax.set_ylim(y_min, y_max)

    ax.set_xlabel("k")
    ax.set_ylabel("Error")
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.4, zorder=0)
    ax.legend(title="Kernel")
    plt.tight_layout()
    plt.show()
    return fig
