import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_u(u: np.ndarray, title="u"):
    plt.figure(figsize=(5,4))
    im = plt.imshow(u, cmap="RdYlGn")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()

def plot_residual(hist, title="Residual"):
    if len(hist) == 0:
        return

    its = [d["it"] for d in hist]

    plt.figure(figsize=(5, 4))

    # delta residual if present
    if "res_delta" in hist[0]:
        vals = [d["res_delta"] for d in hist]
        plt.semilogy(its, vals)

    # default to d=1 residual
    elif "res_true" in hist[0]:
        vals = [d["res_true"] for d in hist]
        plt.semilogy(its, vals)

    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title(title)
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()
    plt.show()

def write_video(snapshots, iters_snap=None, filename="out.mp4", fps=10, cmap="RdYlGn"):
    if not snapshots:
        raise ValueError("No snapshots to write.")

    vmin = min(s.min() for s in snapshots)
    vmax = max(s.max() for s in snapshots)

    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(snapshots[0], cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    title = ax.set_title("")

    def update(k):
        im.set_data(snapshots[k])
        if iters_snap is not None:
            title.set_text(f"Iteration {iters_snap[k]}")
        else:
            title.set_text(f"Frame {k}")
        return (im, title)

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=False)

    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    ani.save(filename, writer=writer)
    plt.close(fig)

def plot_delta_residual_sweep(results, *, which="res_delta"):
    """
    results: dict mapping delta -> dict, where dict has key "hist"
    which:   "res_delta" or "res_true"
    """
    plt.figure(figsize=(7, 5))

    for d in sorted(results.keys()):
        hist = results[d]["hist"]
        if not hist:
            continue

        its = [e["it"] for e in hist]

        # δ = 1 → always plot true residual
        if np.isclose(d, 1.0):
            if "res_true" not in hist[0]:
                continue
            vals = [e["res_true"] for e in hist]
            plt.semilogy(its, vals, label=r"$\delta=1$ (Jacobi)")
            continue

        # δ < 1 → plot requested residual
        if which not in hist[0]:
            continue
        vals = [e[which] for e in hist]
        plt.semilogy(its, vals, label=fr"$\delta={d}$")

    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title(r"Residuals of $M(u_{\delta,n})=f$")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# def plot_alpha(alpha: np.ndarray, mask: np.ndarray):
#     a = alpha.astype(float).copy()
#     a[~mask] = np.nan
#     plt.figure(figsize=(5, 4))
#     plt.imshow(a)
#     plt.colorbar(label="walker choice")
#     plt.title(r"Random policy $\alpha$")
#     plt.tight_layout()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def plot_alpha(alpha: np.ndarray, mask: np.ndarray):
    # copy and hide boundary
    a = alpha.astype(float).copy()
    a[~mask] = np.nan

    cw0 = "tab:brown"
    cw1 = "tab:olive"

    # define two colors: walker 0, walker 1
    cmap = ListedColormap([cw0, cw1])

    plt.figure(figsize=(5, 4))
    plt.imshow(a, cmap=cmap, vmin=0, vmax=1)

    # legend instead of colorbar
    legend_patches = [
        mpatches.Patch(color=cw0, label="Walker 0"),
        mpatches.Patch(color=cw1, label="Walker 1"),
    ]
    plt.legend(handles=legend_patches, loc="upper right")

    plt.title(r"Random policy $\alpha$")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def plot_u_alpha_diff(u_alpha: np.ndarray,
                      u_opt: np.ndarray,
                      mask: np.ndarray,
                      *,
                      symmetric: bool = True):
    """
    Heatmap of difference between fixed-policy solution u_alpha
    and optimal solution u.

    Parameters
    ----------
    u_alpha : ndarray
        Solution under fixed policy alpha
    u_opt : ndarray
        Optimal solution
    mask : ndarray (bool)
        True on interior, False on boundary
    symmetric : bool
        If True, color scale is symmetric about 0
    """
    diff = (u_alpha - u_opt).astype(float)
    diff[~mask] = np.nan

    plt.figure(figsize=(5, 4))

    if symmetric:
        vmax = np.nanmax(np.abs(diff))
        vmin = -vmax
    else:
        vmin, vmax = np.nanmin(diff), np.nanmax(diff)

    im = plt.imshow(diff, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(r"$u_\alpha - u$")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()