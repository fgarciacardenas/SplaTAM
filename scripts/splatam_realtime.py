import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader
from typing import Any, Dict, List, Tuple

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset, HabitatDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify, calc_psnr

from hessian_diff_gaussian_rasterization_w_depth import GaussianRasterizer as Renderer
from datasets.gradslam_datasets.geometryutils import relative_transformation
from utils.slam_helpers import quat_mult

# ROS dependencies
import rospy
# ros_numpy patch for Python 3.10+
np.float = float
import ros_numpy
from std_msgs.msg import Bool, Float32MultiArray
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from scipy.spatial.transform import Rotation
import collections

# Control flags
VERBOSE = False
STREAM_VIZ = False
DUMP_DATA = False
GRID_VIZ = False       # Show XY occupancy grid
CURRENT_VIZ = True     # Show current frame render image
CANDIDATE_VIZ = False  # Show Silhouette and RGB renders

OCC_SCALE = 30            # px per grid cell (30: every 0.5 m cell becomes 30×30 px)
PT_COLOR  = (0, 255, 255) # 2D point color (BGR – cyan)
PT_RADIUS = 1             # 2D point size (px)


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["habitat"]:
        return HabitatDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(ros_handler, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None):
    # Block until ROS trigger fires
    while not rospy.is_shutdown():
        # Send ready signal
        ros_handler.send_ready()
        
        # Wait for handler to be triggered
        if not ros_handler.triggered:
            rospy.sleep(0.01)
            continue

        # Read data in buffers
        ros_data = ros_handler.get_current_data()
        if ros_data is None:
            if VERBOSE:
                rospy.logwarn("No ROS data yet, skipping")
            continue

        ros_handler.triggered = False
        break

    if rospy.is_shutdown():
        raise rospy.ROSInterruptException()

    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = ros_data

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    # Connect objects to ros_handler
    ros_handler.cam = cam
    ros_handler.w2c_ref = w2c
    ros_handler.params = params

    # Collect new data
    ros_handler.map_ready = True

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    # Poses are relative to the first frame
    # dataset = get_dataset(
    #     config_dict=gradslam_data_cfg,
    #     basedir=dataset_config["basedir"],
    #     sequence=os.path.basename(dataset_config["sequence"]),
    #     start=dataset_config["start"],
    #     end=dataset_config["end"],
    #     stride=dataset_config["stride"],
    #     desired_height=dataset_config["desired_image_height"],
    #     desired_width=dataset_config["desired_image_width"],
    #     device=device,
    #     relative_pose=True,
    #     ignore_bad=dataset_config["ignore_bad"],
    #     use_train_split=dataset_config["use_train_split"],
    # )
    # num_frames = dataset_config["num_frames"]
    # if num_frames == -1:
    #     num_frames = len(dataset)
    num_frames = 1000

    # Initialize ROS Node
    ros_handler = RosHandler(gradslam_data_cfg)

    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(ros_handler, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset,
                                                                        gaussian_distribution=config['gaussian_distribution'])                                                                                                                  
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(ros_handler, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'],
                                                                                        gaussian_distribution=config['gaussian_distribution'])
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Initialize dataset
    dataset = []

    # Iterate over Scan
    print("Initializing SLAM...")
    time_idx = 0
    while not rospy.is_shutdown():
        # Exit if node finished flag is triggered
        if ros_handler.finished:
            num_frames = time_idx
            break

        if time_idx == 0:
            ros_data = ros_handler.get_first_data()
        else:
            # Block until ROS trigger fires
            rospy.logwarn_once("Waiting for trigger...")

            if not ros_handler.triggered:
                # Check if new silhouette data has been requested
                ros_handler.send_gains()
                rospy.sleep(0.01)
                continue

            # Get new data sample
            ros_data = ros_handler.get_current_data()
            if ros_data is None:
                if VERBOSE:
                    rospy.logwarn("No ROS data yet, skipping")
                continue
            ros_handler.triggered = False

        # Increment current index
        time_idx += 1
        print("Adding Frame to Gaussian Splat %d" % time_idx)

        # Load RGBD frames incrementally instead of all frames
        color, depth, intrinsics, gt_pose = ros_data
        dataset.append((color, depth, intrinsics, gt_pose))

        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        
        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        # Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                   plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter)
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                            wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], config['gaussian_distribution'])
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                # Loss for current frame
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        
        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1

        # Clean data
        torch.cuda.empty_cache()

        # Compute Hessian for training poses
        # TODO
        ros_handler.compute_H_visited_inv()
        
        # Collect new data
        ros_handler.map_ready = True
    

    if DUMP_DATA:
        dump_realtime_dataset(dataset, '/home/dev/frame_rt')

    # Exit condition
    if rospy.is_shutdown():
        print("ROS Node Shutdown")
        cv2.destroyAllWindows()
        exit(1)

    # Finalize Parameters
    params['cam_unnorm_rots'] = params['cam_unnorm_rots'][..., :num_frames]
    params['cam_trans'] = params['cam_trans'][..., :num_frames]

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
    
    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)

    # Close CV2 Windows
    cv2.destroyAllWindows()

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()

    # Plot poses and corresponding gains
    plot_pose_gains(ros_handler.global_gains, save_dir=output_dir + '/poses')

    # Plot <value> vs PSNR
    plot_value_psnr(ros_handler.gain_psnr_arr, value="sil", axis_name="SIL", prefix="psnr_sil", save_dir=output_dir + '/psnr_plots')
    plot_value_psnr(ros_handler.gain_psnr_arr, value="eig", axis_name="EIG", prefix="psnr_eig", save_dir=output_dir + '/psnr_plots')
    plot_value_psnr(ros_handler.gain_psnr_arr, value="loc", axis_name="LOC", prefix="psnr_loc", save_dir=output_dir + '/psnr_plots')
    plot_value_psnr(ros_handler.gain_psnr_arr, value="fim", axis_name="FIM", prefix="psnr_fim", save_dir=output_dir + '/psnr_plots')
    plot_value_psnr(ros_handler.gain_psnr_arr, value="gain", axis_name="SUM", prefix="psnr_sum", save_dir=output_dir + '/psnr_plots')


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

        fig, axes = plt.subplots(5, 1, sharex=True, figsize=(12, 12))
        axes[0].set_title("SIL gains")
        axes[1].set_title("EIG gains")
        axes[2].set_title("LOC gains")
        axes[3].set_title("FIM gains")
        axes[4].set_title("Mixed gains")
        axes[4].set_xlabel("time step")

        for pose_key, seq in chunk:
            sil = np.full(T, np.nan)
            eig = np.full(T, np.nan)
            loc = np.full(T, np.nan)
            fim = np.full(T, np.nan)
            mix = np.full(T, np.nan)

            for t, item in enumerate(seq):
                rec = _first_dict(item)
                if rec is None:
                    continue
                sil[t] = rec.get("sil",  np.nan)
                eig[t] = rec.get("eig",  np.nan)
                loc[t] = rec.get("loc",  np.nan)
                fim[t] = rec.get("fim",  np.nan)
                mix[t] = rec.get("gain", np.nan)

            label = (
                f"<{pose_key[:4]}, {pose_key[4:8]}, {pose_key[8:12]}>, "
                f"{pose_key[12:]}°"
            )
            # Draw marker so even single-point series are visible
            axes[0].plot(x, sil, marker="o", linestyle="-", label=label)
            axes[1].plot(x, eig, marker="o", linestyle="-", label=label)
            axes[2].plot(x, loc, marker="o", linestyle="-", label=label)
            axes[3].plot(x, fim, marker="o", linestyle="-", label=label)
            axes[4].plot(x, mix, marker="o", linestyle="-", label=label)

        axes[0].legend(fontsize=7, loc="upper right")

        fig.tight_layout()
        fname = os.path.join(
            save_dir,
            f"{prefix}_{fig_idx:03d}_{time.time_ns()}.png"
        )
        fig.savefig(fname, dpi=300)
        plt.close(fig)


def plot_value_psnr(
    gains_arr: list,
    value: str = "eig",
    axis_name: str = "EIG",
    save_dir: str = "/home/dev/splatam/experiments/",
    prefix: str = "psnr_eig",
) -> None:
    """
    Plot <value> vs PSNR.
    """
    os.makedirs(save_dir, exist_ok=True)

    if not gains_arr:
        print("plot_<value>_psnr: nothing to plot.")
        return

    # Extract <value> and PSNR values
    values = [g[value] for g in gains_arr]
    psnr_values = [g["psnr"] for g in gains_arr]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(psnr_values, values, alpha=0.8)
    ax.set_xlabel("PSNR")
    ax.set_ylabel(f"{axis_name}")
    ax.set_title(f"{axis_name} vs PSNR")

    # Save the figure
    fname = os.path.join(save_dir, f"{prefix}_{time.time_ns()}.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def transform_gaussians(params: dict, cam_pose: torch.Tensor,
                        requires_grad: bool = False) -> dict:
    """
    Function to transform Isotropic or Anisotropic Gaussians from world frame to camera frame.
    
    Args:
        params: dict of parameters
        cam_pose: camera pose
        requires_grad: flag whether to keep gradients attached
    
    Returns:
        transformed_gaussians: Gaussians in the camera frame
    """

    # Check if Gaussians need to be rotated (Isotropic or Anisotropic)
    if params['log_scales'].shape[1] == 1:
        transform_rots = False # Isotropic Gaussians
    else:
        transform_rots = True # Anisotropic Gaussians
    
    # Get Centers and Unnorm Rots of Gaussians in World Frame
    pts = params['means3D']
    unnorm_rots = params['unnorm_rotations']

    # Detach or keep gradients
    if not requires_grad:
        pts = pts.detach()
        unnorm_rots = unnorm_rots.detach()
    
    # Transform Centers of Gaussians to Camera Frame
    transformed_gaussians = {}
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (cam_pose @ pts4.T).T[:, :3]
    transformed_gaussians['means3D'] = transformed_pts

    # Transform Rots of Gaussians to Camera Frame
    if transform_rots:
        norm_rots = F.normalize(unnorm_rots)
        transformed_rots = quat_mult(matrix_to_quaternion(cam_pose[:3,:3]), norm_rots)
        transformed_gaussians['unnorm_rotations'] = transformed_rots
    else:
        transformed_gaussians['unnorm_rotations'] = unnorm_rots

    return transformed_gaussians
    

def get_renders(params, cam_pose, cam_data):
    """
    Function to transform Isotropic or Anisotropic Gaussians from world frame to camera frame.
    
    Args:
        params: dict of parameters
        cam_pose: camera pose
        cam_data: camera data
        _render: flag whether to return RGB render
    
    Returns:
        silhouette: silhouette image
        rgb_render: RGB rendered image (optional, default: None)
    """
    with torch.no_grad():
        # Transform Gaussians from world frame to camera frame
        transformed_gaussians = transform_gaussians(params, cam_pose, requires_grad = False)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        rgb_render, _, _, = Renderer(raster_settings=cam_data['cam'])(**rendervar)
        
        # Get depth render
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, cam_data['w2c'],
                                                                     transformed_gaussians)
        
        # Extract silhouette image
        depth_sil, _, _, = Renderer(raster_settings=cam_data['cam'])(**depth_sil_rendervar)
        silhouette = depth_sil[1, :, :]

        return silhouette, rgb_render


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


def _make_canvas(cols, rows, cell_w, cell_h, text_h, pad):
    H = rows * (cell_h + text_h) + (rows - 1) * pad
    W = cols *  cell_w           + (cols - 1) * pad
    return np.zeros((H, W, 3), np.uint8)


class RosHandler:
    def __init__(self, config_dict, max_queue_size=5, max_dt=0.08):
        # Parameters
        self.max_dt = max_dt
        self.max_queue_size = max_queue_size

        # Camera objects
        self.cam     = None
        self.w2c_ref = None
        self.params  = None

        # Rotate camera_frame to camera_optical_frame
        self.r_cam_to_opt = Rotation.from_quat([-0.5, 0.5, -0.5, 0.5])

        # Define gain factors
        self.k_fisher = 1
        self.k_sil = 300
        self.k_sum = 5
        self.H_train_inv = None

        # Initialize visualization windows
        if CANDIDATE_VIZ:
            cv2.namedWindow("Renders", cv2.WINDOW_NORMAL)
        if GRID_VIZ:
            cv2.namedWindow("Occupancy", cv2.WINDOW_NORMAL)
            self.last_grid_img = None

        # Render canvas
        self._cols   = 4
        self._rows   = 2
        self._cell_w = 480
        self._cell_h = 640
        self._text_h = 32
        self._pad = 12
        self._viz_canvas = _make_canvas(self._cols, self._rows,
                                        self._cell_w, self._cell_h,
                                        self._text_h, self._pad)

        # Pose gain request
        self.gs_poses = collections.deque(maxlen=max_queue_size)
        self.global_gains = {}
        self.gain_psnr_arr = []

        # Store visited poses
        self.visited_poses = []

        # Queues for synchronization
        self.rgb_queue   = collections.deque(maxlen=max_queue_size)
        self.depth_queue = collections.deque(maxlen=max_queue_size)
        self.pose_queue  = collections.deque(maxlen=max_queue_size)
        
        self.initial_pose = None
        self.first_dataframe = None
        self.triggered    = False
        self.finished     = False
        self.map_ready    = True

        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        K = torch.from_numpy(self.as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy]))
        #K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        self.intrinsics = torch.eye(3).to(K).type(torch.float)
        self.intrinsics[:3, :3] = K

        # Set up ROS subscribers
        rospy.Subscriber('/ifpp_camera/rgb/image_rect_color', Image,
                         self._rgb_cb, queue_size=1)
        rospy.Subscriber('/ifpp_camera/depth/depth_registered', Image,
                         self._depth_cb, queue_size=1)
        rospy.Subscriber('/odometry', Odometry,
                         self._pose_cb, queue_size=1)
        rospy.Subscriber('/ifpp/trigger_signal', Bool,
                         self._trigger_cb, queue_size=1)
        rospy.Subscriber('/ifpp/finished_signal', Bool,
                         self._finished_cb, queue_size=1)
        rospy.Subscriber('/ifpp/stop_gs', Bool,
                         self._terminate_cb, queue_size=1)
        rospy.Subscriber('/ifpp/gs_poses', PoseArray,
                         self._gs_poses_cb, queue_size=1)
        
        # Set up ROS publishers
        self.gain_pub = rospy.Publisher('/ifpp/gs_gains', Float32MultiArray, queue_size=1)
        self.ready_pub = rospy.Publisher('/ifpp/ready_signal', Bool, queue_size=1)
        
        # GUI windows
        if STREAM_VIZ:
            cv2.namedWindow('RGB',   cv2.WINDOW_NORMAL)
            cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)

    def as_intrinsics_matrix(self, intrinsics):
        K = np.eye(3)
        K[0, 0], K[1, 1] = intrinsics[0], intrinsics[1]
        K[0, 2], K[1, 2] = intrinsics[2], intrinsics[3]
        return K

    def _ts_ns(self, header):
        return int(header.stamp.secs)*1_000_000_000 + int(header.stamp.nsecs)

    def _rgb_cb(self, msg: Image):
        if not self.map_ready: return
        ts = self._ts_ns(msg.header)
        arr = ros_numpy.numpify(msg)
        self.rgb_queue.append({'ts': ts, 'arr': arr, 'enc': msg.encoding})

    def _depth_cb(self, msg: Image):
        if not self.map_ready: return
        ts = self._ts_ns(msg.header)
        arr = ros_numpy.numpify(msg)
        if arr.dtype in (np.float32, np.float64):
            arr = (arr * 1000).astype(np.uint16)
        self.depth_queue.append({'ts': ts, 'arr': arr, 'enc': msg.encoding})

    def _pose_cb(self, msg: Odometry):
        if not self.map_ready: return
        ts  = self._ts_ns(msg.header)
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        # Compute orientation rotated into optical frame
        q_cam = np.array([ori.x, ori.y, ori.z, ori.w])
        r_cam = Rotation.from_quat(q_cam)
        r_opt = r_cam * self.r_cam_to_opt
        q_opt = r_opt.as_quat()

        # Generate pose array
        arr = np.array([pos.x, pos.y, pos.z, *q_opt], dtype=np.float32)

        self.pose_queue.append({'ts': ts, 'arr': arr})

    def _trigger_cb(self, msg: Bool):
        if msg.data and self.map_ready:
            self.triggered = True
            #self.gs_poses.clear()

    def _finished_cb(self, msg: Bool):
        if msg.data:
            self.finished = True

    def _terminate_cb(self, msg: Bool):
        if msg.data:
            self.finished = True

    def _gs_poses_cb(self, msg: PoseArray):
        if VERBOSE:
            rospy.loginfo(f"Received {len(msg.poses)} GS poses")
        pose_arr = []
        for pose_msg in msg.poses:
            # Compute orientation rotated into optical frame
            q_cam = np.array([pose_msg.orientation.x, pose_msg.orientation.y,
                              pose_msg.orientation.z, pose_msg.orientation.w])
            r_cam = Rotation.from_quat(q_cam)
            r_opt = r_cam * self.r_cam_to_opt
            q_opt = r_opt.as_quat()

            # Generate pose array
            pose = np.array([pose_msg.position.x, 
                             pose_msg.position.y, 
                             pose_msg.position.z, 
                             *q_opt], dtype=np.float32)
            pose_arr.append(pose)
        pose_arr = np.array(pose_arr)
        self.gs_poses.append(pose_arr)

    def send_ready(self):
        msg = Bool()
        msg.data = True
        self.ready_pub.publish(msg)

    def send_gains(self):
        if self.triggered:
            rospy.logwarn("Cannot process GS poses while mapping is running.")
            return

        if len(self.gs_poses) == 0:
            #rospy.logwarn("No GS poses available to send gains.")
            return
        
        if self.cam is None or self.params is None:
            rospy.logwarn_once("GS map not ready yet - ignoring gain request.")
            return
        
        if self.initial_pose is None:
            rospy.logwarn_once("Initial pose not set - ignoring gain request.")
            return

        # Check whether more than one array was found
        n_samples = len(self.gs_poses)
        if n_samples > 1:
            rospy.logwarn(f"Attempting to process {n_samples} pose arrays! Cleaning queue...")
            for _ in range(n_samples - 1):
                self.gs_poses.popleft()

        # Refresh occupancy grid once per call
        if GRID_VIZ:
            print("Updating occupancy grid...")
            self._update_occupancy_window(z_slice=1, z_tol=0.15, cell=0.2, min_points=10)
            print("Occupancy grid updated.")

        # Now process each collected pose
        pose_arr = self.gs_poses.popleft()
        gains    = []
        sil_arr, rgb_arr, gains_arr = [], [], []
        cam_data = {'cam': self.cam, 'w2c': self.w2c_ref}

        with torch.no_grad():
            for sil_idx, vec in enumerate(pose_arr):
                # Compute relative poses
                pose_mat = self.pose_matrix_from_quaternion(vec)
                pose_mat = torch.from_numpy(pose_mat).float().cuda()
                pose_vec = relative_transformation(self.initial_pose, pose_mat,
                                                   orthogonal_rotations=False)

                # Generate renders
                sil, rgb_render = get_renders(self.params, torch.linalg.inv(pose_vec), cam_data)
                
                # Compute Silhouette gains
                g_sil = float((sil < 0.5).sum().item())

                # Normalize Silhouette gains by number of pixels
                g_sil /= (cam_data['cam'].image_width * cam_data['cam'].image_height)

                # Compute Fisher Information gains
                # if self.k_fisher != 0:
                #     with torch.enable_grad():
                #         g_fisher, eig, loc = self.compute_eig(torch.linalg.inv(pose_vec), cam_data)
                # else:
                g_fisher, eig, loc = 0, 0, 0
                eig = self.compute_eig_score(pose_vec)
                #for now fish only EIG
                g_fisher = eig
                # Scale gains
                g_sil *= self.k_sil
                g_fisher *= self.k_fisher

                # Compute mixed gains
                g = self.k_sum * (g_fisher + g_sil)
                gains.append(g)

                # Gains dictionary
                gains_dict = {
                    'sil': g_sil,
                    'eig': eig,
                    'loc': loc,
                    'fim': g_fisher,
                    'gain': g
                }
                gains_arr.append(gains_dict)

                # Compute yaw angle
                yaw = math.degrees(math.atan2(2*(vec[6]*vec[5] + vec[3]*vec[4]),
                                              1 - 2*(vec[4]**2 + vec[5]**2)))
                
                # Compute dictionary key
                key = f"{vec[0]:.2f}{vec[1]:.2f}{vec[2]:.2f}{yaw:.0f}"
                
                # Store poses and corresponding gains
                if key not in self.global_gains:
                    self.global_gains[key] = []
                self.global_gains[key].append(gains_dict)

                # Visualize renders
                if CANDIDATE_VIZ:
                    sil_arr.append(sil)
                    rgb_arr.append(rgb_render)

                if sil_idx == (len(pose_arr) - 1):
                    if CANDIDATE_VIZ and rgb_arr and sil_arr and (sil_idx > 0):
                        self._show_rgb_sil(rgb_arr, sil_arr, pose_arr, gains_arr, mode=2)
                    sil_arr, rgb_arr  = [], []
        
        # Publish gains
        msg = Float32MultiArray()
        msg.data = gains
        if True: # VERBOSE
            rospy.loginfo(f"Publishing gains {gains} for position [{pose_arr[0][0]:.2f}, {pose_arr[0][1]:.2f}, {pose_arr[0][2]:.2f}]...")
        self.gain_pub.publish(msg)

    def associate_frames(self, t_rgb, t_depth, t_pose):
        matches = []
        for i, tr in enumerate(t_rgb):
            j = np.argmin(np.abs(t_depth - tr))
            k = np.argmin(np.abs(t_pose - tr))
            if (abs(t_depth[j]-tr) < self.max_dt and
                abs(t_pose[k]-tr) < self.max_dt):
                matches.append((i, j, k))
        return matches

    def get_first_data(self):
        return self.first_dataframe

    def get_current_data(self):
        if not (self.rgb_queue and self.depth_queue and self.pose_queue):
            return None
        
        # Extract timestamps
        ts_rgb  = np.array([item['ts'] for item in self.rgb_queue])
        ts_dep  = np.array([item['ts'] for item in self.depth_queue])
        ts_pose = np.array([item['ts'] for item in self.pose_queue])

        assoc = self.associate_frames(ts_rgb, ts_dep, ts_pose)
        if not assoc:
            return None
        
        # Lock further callbacks until consumed
        self.map_ready = False
        if VERBOSE:
            print("RGB Timestamp: %d, Depth Timestamp: %d, Pose Timestamp: %d" % (rgb['ts'], dep['ts'], pose['ts']))

        # Choose most recent match
        i,j,k = assoc[-1]
        rgb = self.rgb_queue[i]
        dep = self.depth_queue[j]
        pose = self.pose_queue[k]

        # Convert and return
        color = torch.from_numpy(rgb['arr']).float().cuda()
        depth = torch.from_numpy(np.expand_dims(dep['arr'], axis=2) / 
                                 self.png_depth_scale).float().cuda()
        pose_mat = self.pose_matrix_from_quaternion(pose['arr'])
        pose_mat = torch.from_numpy(pose_mat).float().cuda()

        # Remove values large than max depth
        # print("Depth min pre: ", depth.min().item())
        # print("Depth max pre: ", depth.max().item())
        depth[depth >= 9.5] = -1
        # print("Depth min post: ", depth.min().item())
        # print("Depth max post: ", depth.max().item())

        if self.initial_pose is None:
            self.initial_pose = pose_mat
        trans_pose = relative_transformation(self.initial_pose, pose_mat, orthogonal_rotations=False)

        intrinsics = self.intrinsics.float().cuda()

        # Clear used entries
        self.clear_used(i,j,k)

        # Optional: Visualize the RGBD data
        if STREAM_VIZ:
            self.plot_images(color, depth)

        # Optional: Visualize the Silhouette and RGB render
        if CURRENT_VIZ and (self.params is not None):
            # Process RGB-D Data
            post_color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
            post_depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
            
            # Generate Silhouette and RGB renders (before mapping)
            cam_data = {'cam': self.cam, 'w2c': self.w2c_ref}
            sil, rgb_render = get_renders(self.params, torch.linalg.inv(trans_pose), cam_data)
            
            # Mask invalid depth in GT
            valid_depth_mask = (post_depth > 0)

            # Compute PSNR
            weighted_im = rgb_render * valid_depth_mask
            weighted_gt_im = post_color * valid_depth_mask
            psnr = calc_psnr(weighted_im, weighted_gt_im).mean().cpu().numpy()

            # Compute Silhouette gains
            g_sil = float((sil < 0.5).sum().item())

            # Normalize Silhouette gains by number of pixels
            g_sil /= (cam_data['cam'].image_width * cam_data['cam'].image_height)

            # Compute Fisher Information gains
            if self.k_fisher != 0:
                with torch.enable_grad():
                    g_fisher, eig, loc = self.compute_eig(torch.linalg.inv(trans_pose), cam_data)
            else:
                g_fisher, eig, loc = 0, 0, 0
            
            # Scale gains
            g_sil *= self.k_sil
            g_fisher *= self.k_fisher

            # Compute mixed gains
            g = 5 * (g_fisher + g_sil)
            
            # Store poses with corresponding gains and psnr
            gains_dict = {
                'pose': pose,
                'sil': g_sil,
                'eig': eig,
                'loc': loc,
                'fim': g_fisher,
                'gain': g,
                'psnr': psnr
            }
            self.gain_psnr_arr.append(gains_dict)

            # Plot previous renders and new view
            self.plot_renders(rgb_render, sil, color, gains_dict)

        if self.first_dataframe is None:
            self.first_dataframe = (color, depth, intrinsics, trans_pose)

        # Append visited pose
        self.visited_poses.append(trans_pose.cuda())

        return color, depth, intrinsics, trans_pose

    def clear_used(self, i, j, k):
        # Drop all entries up to index i/j/k
        for _ in range(i+1): self.rgb_queue.popleft()
        for _ in range(j+1): self.depth_queue.popleft()
        for _ in range(k+1): self.pose_queue.popleft()

    def pose_matrix_from_quaternion(self, arr):
        rot = Rotation.from_quat(arr[3:])
        mat = np.eye(4)
        mat[:3,:3] = rot.as_matrix()
        mat[:3,3]  = arr[:3]
        return mat

    def plot_images(self, rgb = None, depth = None):
        # Handle RGB input
        if rgb:
            arr = rgb['arr']
            enc = rgb['enc']
            if VERBOSE:
                rospy.loginfo(f"[RGB dbg] enc={enc} shape={arr.shape}"
                            f" dtype={arr.dtype} min={arr.min()} max={arr.max()}")
            # normalize floats
            if arr.dtype in (np.float32, np.float64):
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            # convert color order if needed
            if arr.ndim == 3 and arr.shape[2] == 3:
                if 'rgb' in enc.lower():
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imshow('RGB', arr)
        else:
            if VERBOSE:
                rospy.logwarn("No RGB data yet.")

        # Handle depth input
        if depth:
            arr = depth['arr']
            enc = depth['enc']
            if VERBOSE:
                rospy.loginfo(f"[Depth dbg] enc={enc} shape={arr.shape}"
                            f" dtype={arr.dtype} min={arr.min()} max={arr.max()}")
            arr = arr.astype(np.float32)
            if arr.max() > 0:
                arr = (arr / arr.max() * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, np.uint8)
            cv2.imshow('Depth', arr)
        else:
            if VERBOSE:
                rospy.logwarn("No Depth data yet.")
        
        cv2.waitKey(1)

    def plot_renders(self, rgb_render, sil_render, rgb_img, gains):
        """
        Visualise (render RGB, silhouette mask, ground-truth RGB) side-by-side.
        """
        # Helper: Tensor/np-array → BGR uint8
        def _to_bgr_u8(t):
            """Handle (C,H,W) or (H,W) tensors / numpy; output BGR uint8."""
            if torch.is_tensor(t):
                arr = t.detach().cpu()
                if arr.ndim == 3:              # (C,H,W) or (H,W,C)
                    if arr.shape[0] in (3, 4): # channel-first
                        arr = arr.permute(1, 2, 0)
                    arr = arr.numpy()
                else:                          # (H,W)
                    arr = arr.numpy()
            else:
                arr = t                         # already numpy

            if arr.ndim == 2:                   # grayscale → 3-ch
                arr = (arr < 0.5).astype(np.uint8) * 255
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                # float 0-1 or 0-255?  make sure we end in 0-255 uint8
                if arr.dtype != np.uint8:
                    maxv = arr.max()
                    if maxv <= 1.0:
                        arr = (arr * 255)
                    arr = arr.clip(0, 255).astype(np.uint8)
                # RGB → BGR for OpenCV
                if arr.shape[2] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr

        img_render = _to_bgr_u8(rgb_render)
        img_sil    = _to_bgr_u8(sil_render)
        img_rgb    = _to_bgr_u8(rgb_img)

        # Normalise heights, concatenate horizontally
        max_h = max(img_render.shape[0], img_sil.shape[0], img_rgb.shape[0])

        def _pad_to_h(img, H):
            if img.shape[0] == H:
                return img
            pad = np.zeros((H - img.shape[0], img.shape[1], 3), np.uint8)
            return np.vstack([img, pad])

        img_render = _pad_to_h(img_render, max_h)
        img_sil    = _pad_to_h(img_sil,    max_h)
        img_rgb    = _pad_to_h(img_rgb,    max_h)

        canvas = cv2.hconcat([img_render, img_sil, img_rgb])

        # Header text
        title = (f"PSNR {gains['psnr']:.2f} | "
                 f"EIG {gains['eig']:.2f} | "
                 f"SIL {gains['sil']:.2f} | "
                 f"SUM {gains['gain']:.2f}")
        cv2.putText(canvas, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)

        # Show & return
        cv2.imshow("Current View", canvas)
        cv2.waitKey(1)

    def _build_tile(self, img3u8, caption):
        """Return <tile_h+text_h, cell_w, 3> ready to blit."""
        # Resize image for canvas
        img = cv2.resize(img3u8, (self._cell_w, self._cell_h),
                         interpolation=cv2.INTER_NEAREST)

        # Add caption to image strip
        strip = np.zeros((self._text_h, self._cell_w, 3), np.uint8)
        cv2.putText(strip, caption, (2, self._text_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1,
                    lineType=cv2.LINE_AA)
        return np.vstack([img, strip])

    def _blit_tile(self, canvas, next_idx, tile):
        r = next_idx // self._cols
        c = next_idx %  self._cols

        # where the tile starts (upper-left corner)
        y0 = r * (self._cell_h + self._text_h + self._pad)
        x0 = c * (self._cell_w             + self._pad)

        canvas[y0:y0 + self._cell_h + self._text_h,
            x0:x0 + self._cell_w] = tile
        return next_idx + 1

    def _show_rgb_sil(self, rgb_tensor, sil_tensor, pose_arr, gains, mode=1):
        """
        Draw RGB and SIL images inter-leaved on the shared 4x2 canvas.
        """
        canvas = self._viz_canvas
        canvas[:] = 0

        next_idx = 0
        for i, pose_vec in enumerate(pose_arr):
            # RGB tile
            rgb = (rgb_tensor[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cap = self._make_caption(pose_vec, gains[i], mode)
            tile = self._build_tile(rgb, cap)
            next_idx = self._blit_tile(canvas, next_idx, tile)

            # SIL tile
            sil = (sil_tensor[i] < .5).cpu().numpy().astype(np.uint8) * 255
            sil = cv2.cvtColor(sil, cv2.COLOR_GRAY2BGR)

            tile = self._build_tile(sil, cap)
            next_idx = self._blit_tile(canvas, next_idx, tile)

        cv2.imshow("Renders", canvas)
        cv2.waitKey(1)

    def _make_caption(self, pose_vec, gains, mode):
        """Return a short text for the bottom strip."""
        x, y, z = pose_vec[:3]
        yaw = math.degrees(math.atan2(
            2 * (pose_vec[6] * pose_vec[5] + pose_vec[3] * pose_vec[4]),
            1 - 2 * (pose_vec[4] ** 2 + pose_vec[5] ** 2)
        ))
        cap = f"<{x:.2f},{y:.2f},{z:.2f},{yaw:+.0f} deg> G:{gains['gain']:.1f}"
        if mode == 1:
            cap += f", FIM:{gains['fim']:.1f}, SIL:{gains['sil']:.1f}"
        else:
            cap += f", EIG:{gains['eig']:.1f}, SIL:{gains['sil']:.1f}"
        return cap

    def _update_occupancy_window(self, z_slice=0.0, z_tol=0.15,
                                 cell=0.20, min_points=5):
        """Re-draw occupancy + slice-points"""
        if not GRID_VIZ or self.params is None:
            return

        with torch.no_grad():
            # Occupancy grid (bool tensor) + spatial extent
            occ, extent = make_occupancy_grid(
                self.params['means3D'],
                init_pose=self.initial_pose,
                z_slice=z_slice, z_tol=z_tol,
                cell=cell, min_points=min_points
            )

            # Convert grid to CV2 format
            img = grid_to_cv2(occ, scale=OCC_SCALE)  # white = free, black = occ
            h, w = occ.shape                         # grid size (rows, cols)

            # Extract ALL points that belong to the same Z-slice
            xyz  = self.params['means3D']  # Shape: (N,3)
            pts_ones = torch.ones(xyz.shape[0], 1).cuda().float()
            pts4 = torch.cat((xyz, pts_ones), dim=1)
            xyz  = (self.initial_pose @ pts4.T).T[:, :3]
            z    = xyz[:, 2]
            mask = (z >= (z_slice - z_tol)) & (z <= (z_slice + z_tol))
            
            if mask.any():
                xy = xyz[mask, :2].detach().cpu()  # Shape: (M,2)

                # Metric to grid
                x_rel = (xy[:, 0] - extent['xmin']) / extent['cell']
                y_rel = (xy[:, 1] - extent['ymin']) / extent['cell']

                # Grid to pixel
                px = (x_rel * OCC_SCALE).numpy()
                py = ((h - 1 - y_rel) * OCC_SCALE).numpy()

                # Draw every point
                for xi, yi in zip(px.astype(np.int32), py.astype(np.int32)):
                    cv2.circle(
                        img, 
                        center=(xi, yi), 
                        radius=PT_RADIUS,
                        color=PT_COLOR,
                        thickness=-1,
                        lineType=cv2.LINE_AA
                    )

        # Refresh the window
        if self.last_grid_img is None or not np.array_equal(img, self.last_grid_img):
            self.last_grid_img = img.copy()
            cv2.imshow("Occupancy", img)
            cv2.resizeWindow("Occupancy", img.shape[1], img.shape[0])
        else:
            cv2.imshow("Occupancy", img)
        cv2.waitKey(1)

    def update_global_fim(self, fim_diag: torch.Tensor, momentum: float = 0.9):
        """
        Keep an exponential-moving-average of the per-frame Fisher information
        diagonal. Resize the stored vector automatically whenever the number
        of Gaussians changes (after densification / pruning).
        """
        # Initial function call
        if not hasattr(self, "_fim_global"):
            self._fim_global = fim_diag.clone().detach() + 1e-9
            return

        # Retrieve previous and new sizes
        old_N = self._fim_global.numel()
        new_N = fim_diag.numel()

        # Gaussian were added or pruned
        if new_N != old_N:
            if new_N > old_N: # If new Gaussians were added
                # Keep running stats
                padded = fim_diag.clone().detach()
                padded[:old_N] = self._fim_global
                self._fim_global = padded
            else: # If Gaussians were pruned
                self._fim_global = self._fim_global[:new_N]

        # Exponential-moving-average update
        self._fim_global = (momentum * self._fim_global + (1.0 - momentum) * fim_diag + 1e-9)

    def compute_eig(self, pose_mat: torch.Tensor, cam_data: dict):
        # Initialize render variables for computing scene (mapping) gradients
        gaussians_scene = transform_gaussians(self.params, pose_mat, requires_grad=True)
        rendervar_scene = transformed_params2rendervar(self.params, gaussians_scene)
        rgb_scene, _, _, = Renderer(raster_settings=cam_data['cam'])(**rendervar_scene)

        # Compute Fisher Information Matrix (diagonalized)
        # The rendered RGB image rgb_scene is differentiated w.r.t. 
        # the Gaussian parameters w_list (means3D, logit_opacities)
        w_list = [self.params['means3D'], self.params['logit_opacities']]
        grads  = torch.autograd.grad(
            outputs       = rgb_scene,
            inputs        = w_list,
            grad_outputs  = torch.ones_like(rgb_scene),
            create_graph  = False, retain_graph = False
        )

        # Shape FIM as 1D vector
        fim_diag = torch.cat([(g.detach() ** 2).reshape(-1) for g in grads])

        # Update global Fisher Information Matrix
        # Dividing the current per-parameter Fisher information by the running average 
        # implements the expected information gain (EIG) for new scene knowledge
        self.update_global_fim(fim_diag)

        # Compute the EIG of the scene
        eig_scene = (fim_diag / self._fim_global).sum()

        # Normalize EIG by number of pixels
        eig_scene /= (cam_data['cam'].image_width * cam_data['cam'].image_height)

        # Initialize render variables for computing pose gradients
        pose_dyn = pose_mat.clone().detach().requires_grad_(True)
        gaussians_pose = transform_gaussians(self.params, pose_dyn, requires_grad=False)
        rendervar_pose = transformed_params2rendervar(self.params, gaussians_pose)
        rgb_pose, _, _ = Renderer(raster_settings=cam_data['cam'])(**rendervar_pose)

        # Jacobians of camera pose with respect to the mean and covariances of each Gaussian
        J_pose, = torch.autograd.grad(
            outputs      = rgb_pose,
            inputs       = pose_dyn,
            grad_outputs = torch.ones_like(rgb_pose),
            retain_graph = False
        )

        # If the view is uninformative, return zero gain
        if J_pose.abs().max() == 0:
            z = torch.tensor(0., device=pose_mat.device).item()
            return z, eig_scene.item(), z

        # Compute Cramer-Rao lower-bound surrogate for pose variance
        Jv = J_pose.reshape(-1)
        JJ = torch.outer(Jv, Jv) # 6x6 Fisher matrix
        JJ64 = JJ.double()
        eps  = 1e-3 * torch.eye(JJ.size(0), device=JJ.device, dtype=JJ64.dtype)
        _, logabsdet = torch.linalg.slogdet(JJ64 + eps)
        loc_cost = logabsdet.float()

        # Combine gain terms
        eig_gain = 2.0
        score = eig_scene - eig_gain * loc_cost
        return -score.item(), eig_scene.item(), loc_cost.item()

    def compute_H_visited_inv(self):
        H_train = None
        for pose in self.visited_poses:
            cur_H = self.compute_Hessian( torch.linalg.inv(pose), return_points=True)
            if H_train is None:
                H_train = torch.zeros(*cur_H.shape, device=cur_H.device, dtype=cur_H.dtype)
            H_train += cur_H
        
        self.H_train_inv = torch.reciprocal(H_train + 0.1)


    def compute_eig_score(self, pose):
        """ Compute pose scores for the poses """

        cur_H = self.compute_Hessian( torch.linalg.inv(pose), return_points=True)
        print("cur_H shape: ", cur_H.shape)

        view_score = torch.sum(cur_H * self.H_train_inv).item() 
        print("EIG: ", view_score)

        return view_score

    @torch.enable_grad()
    def compute_Hessian(self, rel_w2c, return_points = False, 
                        return_pose = False):
        """
            Compute uncertainty at candidate pose
                params: Gaussian slam params
                candidate_trans: (3, )
                candidate_rot: (4, )
                return_points:
                    if True, then the Hessian matrix is returned in shape (N, C), 
                    else, it is flatten in 1-D.

        """
        if isinstance(rel_w2c, np.ndarray):
            rel_w2c = torch.from_numpy(rel_w2c).cuda()
        rel_w2c = rel_w2c.float()

        # transform to candidate frame
        with torch.no_grad():
            pts = self.params['means3D']
            pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
            pts4 = torch.cat((pts, pts_ones), dim=1)

            transformed_pts = (rel_w2c @ pts4.T).T[:, :3]
            rgb_colors = self.params['rgb_colors']
            rotations = F.normalize(self.params['unnorm_rotations'])
            opacities = torch.sigmoid(self.params['logit_opacities'])
            scales = torch.exp(self.params['log_scales'])
            if scales.shape[-1] == 1: # isotropic
                scales = torch.tile(scales, (1, 3))

        num_points = transformed_pts.shape[0]

        rendervar = {
            'means3D': transformed_pts.requires_grad_(True),
            'colors_precomp': rgb_colors.requires_grad_(True),
            'rotations': rotations.requires_grad_(True),
            'opacities': opacities.requires_grad_(True),
            'scales': scales.requires_grad_(True),
            'means2D': torch.zeros_like(transformed_pts, requires_grad=True, device="cuda") + 0
        }

        # for means3D, rotation won't change sum of square since R^T R = I
        rendervar['means2D'].retain_grad()
        im, _, _, = Renderer(raster_settings=self.cam, backward_power=2)(**rendervar)
        im.backward(gradient=torch.ones_like(im) * 1e-3)

        if return_points:
            cur_H = torch.cat([transformed_pts.grad.detach().reshape(num_points, -1),  
                                opacities.grad.detach().reshape(num_points, -1)], dim=1)

        else:
            cur_H = torch.cat([transformed_pts.grad.detach().reshape(-1), 
                                opacities.grad.detach().reshape(-1)])
            
        # set grad to zero
        for k, v in rendervar.items():
            v.grad.fill_(0.)

        if not return_pose:
            return cur_H
        else:
            return cur_H, torch.eye(6).cuda()




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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    # Set up ROS subscribers
    rospy.init_node('ifpp_3dgs')

    # Start RGBD SLAM
    try:
        rgbd_slam(experiment.config)
    except rospy.ROSInterruptException:
        print("Caught shutdown request — exiting cleanly")