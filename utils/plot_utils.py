# Plotting utilities for SplatAM
import os
import time
import math
import torch
import numpy as np
from typing import Any, Dict, List, Tuple

# Visualization dependencies
import cv2
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize


def _make_canvas(cols, rows, cell_w, cell_h, text_h, pad):
    H = rows * (cell_h + text_h) + (rows - 1) * pad
    W = cols *  cell_w           + (cols - 1) * pad
    return np.zeros((H, W, 3), np.uint8)


def grid_to_cv2(occ: torch.Tensor, free_val: int = 255,
                occ_val: int = 0, scale: int = 1) -> np.ndarray:
    """
    Convert occupancy grid to an OpenCV BGR image (white = free, black = occupied).
    Output resolution equals the grid resolution; you can cv2.resize() later.
    """
    img = (~occ).cpu().numpy().astype(np.uint8) * free_val  # free cells = 255
    img[img == 0] = occ_val                                 # occupied = 0
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if scale != 1:
        h, w = img3.shape[:2]
        img3 = cv2.resize(img3, (w*scale, h*scale),
                          interpolation=cv2.INTER_NEAREST)
    return img3


def make_occupancy_grid(
        xyz        : torch.Tensor,
        init_pose  : torch.Tensor,
        z_slice    : float = 0.50,
        z_tol      : float = 0.10,
        cell       : float = 0.50,
        min_points : int   = 10,
):
    """
    Return (occ_mask, extent) for all points whose z-coord lies in
    [z_slice ± z_tol]. A cell is occupied if ≥ min_points fall into it.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N,3)")
    
    # Convert points to camera frame
    pts_ones = torch.ones(xyz.shape[0], 1).cuda().float()
    pts4 = torch.cat((xyz, pts_ones), dim=1)
    xyz = (init_pose @ pts4.T).T[:, :3]

    # Filter points by z level
    z_low, z_high = z_slice - z_tol, z_slice + z_tol
    use = (xyz[:, 2] >=  z_low) & (xyz[:, 2] <= z_high)
    if not use.any():
        return (torch.zeros((1, 1), dtype=torch.bool, device=xyz.device),
                {"xmin":0, "xmax":0, "ymin":0, "ymax":0, "cell":cell})
    flat = xyz[use, :2] # Shape: (M,2)

    # Grid index for every point
    xmin, ymin = flat.min(0).values
    xmax, ymax = flat.max(0).values
    ix = torch.div(flat[:, 0] - xmin, cell, rounding_mode='floor').long()
    iy = torch.div(flat[:, 1] - ymin, cell, rounding_mode='floor').long()
    W = (torch.div(xmax - xmin, cell, rounding_mode='floor').long() + 1).item()
    H = (torch.div(ymax - ymin, cell, rounding_mode='floor').long() + 1).item()

    # 1-D indices
    lin = (iy * W + ix).cpu()

    # Compute level histogram
    counts = torch.bincount(lin, minlength=H*W).reshape(H, W).to(xyz.device)

    # Mask and book-keeping
    occ = (counts >= min_points).bool().flip(0)  # Y axis up
    extent = {
        "xmin": xmin.item(),  "xmax": xmax.item(),
        "ymin": ymin.item(),  "ymax": ymax.item(),
        "cell": cell,
    }
    return occ, extent


def _first_dict(item: Any) -> Dict[str, float] | None:
    """Return the first gain-dict inside *item* or None."""
    if isinstance(item, dict):
        return item
    if isinstance(item, (list, tuple)) and item:
        return item[0] if isinstance(item[0], dict) else None
    return None


def plot_pose_gains(
    gains_dict: dict,
    n_per_fig: int = 8,
    max_figs: int = 3,
    save_dir: str = "/home/dev/splatam/experiments/",
    prefix: str = "gains",
) -> None:
    """
    Plot EIG, SIL and mixed gains for the *longest* sequences only.

    * Keeps the top (max_figs x n_per_fig) poses ranked by sequence length.
    * Draws ≤ max_figs figures, each with ≤ n_per_fig pose-curves.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Rank poses
    poses_sorted : List[Tuple[str, List[Any]]] = sorted(
        gains_dict.items(),
        key=lambda kv: len(kv[1]),
        reverse=True
    )

    limit = max_figs * n_per_fig
    poses_top = poses_sorted[:limit]
    if not poses_top:
        print("plot_pose_gains: nothing to plot.")
        return

    # Global time horizon (longest sequence among *selected* poses)
    T = max(len(seq) for _, seq in poses_top)
    x = np.arange(1, T + 1)

    # Generate plots
    n_figs = math.ceil(len(poses_top) / n_per_fig)
    n_figs = min(n_figs, max_figs)

    for fig_idx in range(n_figs):
        start = fig_idx * n_per_fig
        stop  = start + n_per_fig
        chunk = poses_top[start:stop]

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 12))
        axes[0].set_title("SIL gains")
        axes[1].set_title("EIG gains")
        axes[2].set_title("Mixed gains")
        axes[2].set_xlabel("Time step")

        for pose_key, seq in chunk:
            sil = np.full(T, np.nan)
            eig = np.full(T, np.nan)
            mix = np.full(T, np.nan)

            for t, item in enumerate(seq):
                rec = _first_dict(item)
                if rec is None:
                    continue
                sil[t] = rec.get("sil",  np.nan)
                eig[t] = rec.get("eig",  np.nan)
                mix[t] = rec.get("gain", np.nan)

            label = (
                f"<{pose_key[:4]}, {pose_key[4:8]}, {pose_key[8:12]}>, "
                f"{pose_key[12:]}°"
            )
            # Draw marker so even single-point series are visible
            axes[0].plot(x, sil, marker="o", linestyle="-", label=label)
            axes[1].plot(x, eig, marker="o", linestyle="-", label=label)
            axes[2].plot(x, mix, marker="o", linestyle="-", label=label)

        axes[0].legend(fontsize=7, loc="upper right")

        fig.tight_layout()
        fname = os.path.join(
            save_dir,
            f"{prefix}_{fig_idx:03d}_{time.time_ns()}.png"
        )
        fig.savefig(fname, dpi=300)
        plt.close(fig)


def plot_value_psnr(
    x_arr: list,
    y_arr: list,
    axis_name: str = "EIG",
    save_dir: str = "/home/dev/splatam/experiments/",
    prefix: str = "psnr_eig",
) -> None:
    """
    Plot <value> vs PSNR.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_arr, y_arr, alpha=0.8)
    ax.set_xlabel("PSNR")
    ax.set_ylabel(f"{axis_name}")
    ax.set_title(f"{axis_name} vs PSNR")

    # Save the figure
    fig.tight_layout()
    fname = os.path.join(save_dir, f"{prefix}_{time.time_ns()}.pdf")
    fig.savefig(fname, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_eig_psnr_slice(
    psnr_arr: list,
    eig_arr: list,
    sil_arr: list,
    thr: float = 100.0,
    axis_name: str = "EIG",
    save_dir: str = "/home/dev/splatam/experiments/",
    prefix: str = "psnr_eig",
) -> None:
    """
    Plot <value> vs PSNR.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Retrieve indexes where silhouette is lower than threshold
    sil_mask = np.array(sil_arr) < thr

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(psnr_arr[sil_mask], eig_arr[sil_mask], alpha=0.8)
    ax.set_xlabel("PSNR")
    ax.set_ylabel(f"{axis_name}")
    ax.set_title(f"{axis_name} vs PSNR")

    ax.dataLim.update_from_data_xy(np.column_stack([psnr_arr, eig_arr]))
    ax.autoscale_view()

    # Save the figure
    fig.tight_layout()
    fname = os.path.join(save_dir, f"{prefix}_{time.time_ns()}_sliced.pdf")
    fig.savefig(fname, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_combined_psnr(
    psnr_values: list,
    sil_values: list,
    eig_values: list,
    save_dir: str = "/home/dev/splatam/experiments/",
    prefix: str = "psnr_combined",
) -> None:
    """
    Plot SIL and EIG against PSNR, with points colour-graded by EIG (“height”).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")

    # Colour-grade by EIG
    norm    = Normalize(vmin=min(eig_values), vmax=max(eig_values))
    scatter = ax.scatter(psnr_values, sil_values, eig_values,
                         c=eig_values, cmap="viridis", norm=norm,
                         alpha=0.85, edgecolors="k", linewidths=0.2)

    ax.set_xlabel("PSNR")
    ax.set_ylabel("SIL")
    ax.set_zlabel("EIG")
    ax.set_title("SIL vs EIG vs PSNR")

    # Add colour-bar so the mapping is obvious
    cbar = fig.colorbar(scatter, ax=ax, pad=0.08)
    cbar.set_label("EIG (colour-mapped)")

    # Save the figure
    fig.tight_layout()
    img_name = f"{prefix}_{time.time_ns()}"
    fig.savefig(os.path.join(save_dir, img_name+".pdf"),
                format="pdf", bbox_inches="tight", pad_inches=0.02)

    # Save pickled figure
    pkl_name = os.path.join(save_dir, img_name+".fig.pkl")
    with open(pkl_name, "wb") as f:
        pickle.dump(fig, f)
    plt.close(fig)


def dump_realtime_dataset(dataset, out_dir):
    """
    Given `dataset` as an iterable of (color, depth, intrinsics, gt_pose) tensors,
    write each item to out_dir/frame_00000.npz, frame_00001.npz, etc.
    """
    os.makedirs(out_dir, exist_ok=True)
    for idx, (color, depth, K, pose) in enumerate(dataset):
        # move to CPU + numpy
        color_np = color.cpu().numpy()
        depth_np = depth.cpu().numpy()
        K_np    = K.cpu().numpy()[:3, :3]
        pose_np = pose.cpu().numpy()

        fname = os.path.join(out_dir, f"frame_{idx:05d}.npz")
        np.savez(
            fname,
            color=color_np,
            depth=depth_np,
            intrinsics=K_np,
            gt_pose=pose_np
        )
        # (optionally) print progress every N frames:
        if idx % 100 == 0:
            print(f"  dumped {idx} --> {fname}")
