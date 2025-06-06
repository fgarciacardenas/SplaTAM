#!/usr/bin/env python3
"""
view_realtime_dataset.py  —  robust RGB-D player for the ROS-dumped .npz files
---------------------------------------------------------------------------
usage: python view_realtime_dataset.py /path/to/frames
       [--wait 16] [--depth-vmin 0.2] [--depth-vmax 3.5] [--enc rgb8]

* Esc or q to quit.
* --enc rgb8  → the .npz colour is RGB (default)
  --enc bgr8  → the .npz colour is BGR (skip the extra swap)
"""

import argparse, glob, os
import cv2, numpy as np
from scipy.spatial.transform import Rotation
import math

def depth_to_color(depth, vmin=None, vmax=None):
    if vmin is None: vmin = np.nanmin(depth)
    if vmax is None: vmax = np.nanmax(depth)
    depth = np.clip(depth, vmin, vmax)
    depth = (depth - vmin) / (vmax - vmin + 1e-12)   # 0-1
    depth_u8 = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

def to_u8_rgb(arr, encoding='rgb8'):
    """
    Accepts:
      • float32 / float64 either 0-1 or 0-255
      • uint8 already 0-255
      • shape (H,W,3) or (3,H,W)
    Returns BGR uint8 for OpenCV.
    """
    if arr.ndim == 3 and arr.shape[0] in (3,4) and arr.shape[-1] not in (3,4):
        arr = np.transpose(arr, (1,2,0))   # (C,H,W) → (H,W,C)

    if arr.dtype != np.uint8:
        mx = arr.max()
        if mx > 1.01:      # 0-255 float
            arr = np.clip(arr,0,255).astype(np.uint8)
        else:              # 0-1 float
            arr = (arr * 255).clip(0,255).astype(np.uint8)

    # OpenCV wants BGR
    if encoding.lower() == 'rgb8':
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

def visualise(folder, wait, vmin, vmax, enc):
    files = sorted(glob.glob(os.path.join(folder, 'frame_*.npz')))
    if not files:
        raise FileNotFoundError(f'No frame_*.npz in {folder}')

    win = 'rgb-d viewer'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for f in files:
        data  = np.load(f)
        color = data['color']
        depth = data['depth']
        pose  = data['gt_pose']

        img_color = to_u8_rgb(color, encoding=enc)
        img_depth = depth_to_color(depth.squeeze(), vmin, vmax)
        img_depth = cv2.resize(img_depth, (img_color.shape[1], img_color.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        mosaic = np.hstack([img_color, img_depth])

        t = pose[:3,3]

        # Convert pose to quaternion
        q = Rotation.from_matrix(pose[:3,:3]).as_quat()

        cv2.setWindowTitle(win, f"t = [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}, {q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}] m")
        cv2.imshow(win, mosaic)

        k = cv2.waitKey(wait) & 0xFF
        if k in (27, ord('q')): break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--wait', type=int, default=1,
                    help='delay between frames in ms (default: 1)')
    ap.add_argument('--depth-vmin', type=float, help='clamp depth lower (m)')
    ap.add_argument('--depth-vmax', type=float, help='clamp depth upper (m)')
    ap.add_argument('--enc', choices=['rgb8','bgr8'], default='rgb8',
                    help='colour encoding inside the npz (default: rgb8)')
    args = ap.parse_args()
    visualise(args.folder, args.wait, args.depth_vmin, args.depth_vmax, args.enc)
