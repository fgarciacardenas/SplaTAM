# ROS Handler for Gaussian SLAM
import math
import numpy as np
import collections
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

# ROS dependencies
import rospy
np.float = float # ros_numpy patch for Python 3.10+
import ros_numpy
from std_msgs.msg import Bool, Float32MultiArray
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray

# Splatam dependencies
from utils.slam_external import calc_psnr
from datasets.gradslam_datasets.geometryutils import relative_transformation
from hessian_diff_gaussian_rasterization_w_depth import GaussianRasterizer as Renderer
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    matrix_to_quaternion, quat_mult
)

# Visualization utils
import cv2
from utils.plot_utils import _make_canvas, grid_to_cv2, make_occupancy_grid


class RosHandler:
    def __init__(self, config_dict, ros_handler_config, max_queue_size=5, max_dt=0.08):
        # Control flags
        self._verbose = False
        self._dump_data = False
        self._stream_viz = False
        self._grid_viz = False       # Show XY occupancy grid
        self._current_viz = True     # Show current frame render image
        self._candidate_viz = False  # Show Silhouette and RGB renders
        
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
        self.k_eig = ros_handler_config['k_eig']
        self.k_sil = ros_handler_config['k_sil']
        self.k_sum = ros_handler_config['k_sum']
        self.nl_sil = ros_handler_config['nl_sil']
        self.nl_eig = ros_handler_config['nl_eig']

        # Fisher Information Matrix parameters
        self.H_train_inv = None
        self.monte_carlo = ros_handler_config['use_monte']
        self.N_monte_carlo = ros_handler_config['n_monte']

        # Print configuration
        print("ROS Handler Configuration:")
        print(f"  k_eig: {self.k_eig}")
        print(f"  k_sil: {self.k_sil}")
        print(f"  k_sum: {self.k_sum}")
        print(f"  nl_sil: {self.nl_sil}")
        print(f"  nl_eig: {self.nl_eig}")
        print(f"  monte_carlo: {self.monte_carlo}")
        print(f"  N_monte_carlo: {self.N_monte_carlo}")
        
        # Store visited poses
        self.visited_poses = []

        # Initialize visualization windows
        if self._candidate_viz:
            cv2.namedWindow("Renders", cv2.WINDOW_NORMAL)
        if self._grid_viz:
            cv2.namedWindow("Occupancy", cv2.WINDOW_NORMAL)

            # Define occupancy grid parameters
            self.last_grid_img = None
            self.occ_scale = 30               # px per grid cell
            self.occ_pt_color = (0, 255, 255) # 2D point color
            self.occ_pt_rad = 1               # 2D point size (px)

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
        if self._stream_viz:
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
        if self._verbose:
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
        if self._grid_viz:
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
                if self.k_eig != 0:
                    g_eig = self.compute_eig_score(pose_vec)
                else:
                    g_eig = 0

                # Scale gains
                g_sil *= self.k_sil
                g_eig *= self.k_eig

                # Apply non-linear scaling to silhouette gains
                if self.nl_sil:
                    g_sil = (3400 / (1 + math.exp(-0.002*g_sil))) - 1700
                if self.nl_eig:
                    g_eig = (3400 / (1 + math.exp(-0.002*g_eig))) - 1700

                # Compute mixed gains
                g_sum = self.k_sum * (g_eig + g_sil)
                gains.append(g_sum)

                # Gains dictionary
                gains_dict = {
                    'sil': g_sil,
                    'eig': g_eig,
                    'gain': g_sum
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
                if self._candidate_viz:
                    sil_arr.append(sil)
                    rgb_arr.append(rgb_render)

                if sil_idx == (len(pose_arr) - 1):
                    if self._candidate_viz and rgb_arr and sil_arr and (sil_idx > 0):
                        self._show_rgb_sil(rgb_arr, sil_arr, pose_arr, gains_arr)
                    sil_arr, rgb_arr  = [], []
        
        # Publish gains
        msg = Float32MultiArray()
        msg.data = gains
        if True: # self._verbose
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
        if self._verbose:
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
        if self._stream_viz:
            self.plot_images(color, depth)

        # Optional: Visualize the Silhouette and RGB render
        if self._current_viz and (self.params is not None):
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
            if self.k_eig != 0:
                g_eig = self.compute_eig_score(trans_pose)
            else:
                g_eig = 0
            
            # Scale gains
            g_sil *= self.k_sil
            g_eig *= self.k_eig

            # Apply non-linear scaling to silhouette gains
            if self.nl_sil:
                g_sil = (3400 / (1 + math.exp(-0.002*g_sil))) - 1700
            if self.nl_eig:
                g_eig = (3400 / (1 + math.exp(-0.002*g_eig))) - 1700

            # Compute mixed gains
            g_sum = 5 * (g_eig + g_sil)
            
            # Store poses with corresponding gains and psnr
            gains_dict = {
                'pose': pose,
                'sil': g_sil,
                'eig': g_eig,
                'gain': g_sum,
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
            if self._verbose:
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
            if self._verbose:
                rospy.logwarn("No RGB data yet.")

        # Handle depth input
        if depth:
            arr = depth['arr']
            enc = depth['enc']
            if self._verbose:
                rospy.loginfo(f"[Depth dbg] enc={enc} shape={arr.shape}"
                            f" dtype={arr.dtype} min={arr.min()} max={arr.max()}")
            arr = arr.astype(np.float32)
            if arr.max() > 0:
                arr = (arr / arr.max() * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, np.uint8)
            cv2.imshow('Depth', arr)
        else:
            if self._verbose:
                rospy.logwarn("No Depth data yet.")
        
        cv2.waitKey(1)


    def plot_renders(self, rgb_render, sil_render, rgb_img, gains):
        """
        Visualise (render RGB, silhouette mask, ground-truth RGB) side-by-side.
        """
        # Helper: Tensor/np-array to BGR uint8
        def _to_bgr_u8(t):
            """Handle (C,H,W) or (H,W) tensors / numpy; output BGR uint8."""
            if torch.is_tensor(t):
                arr = t.detach().cpu()
                if arr.ndim == 3:
                    if arr.shape[0] in (3, 4):
                        arr = arr.permute(1, 2, 0)
                    arr = arr.numpy()
                else:
                    arr = arr.numpy()
            else:
                arr = t

            # Convert grayscale to BGR
            if arr.ndim == 2:
                arr = (arr < 0.5).astype(np.uint8) * 255
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                if arr.dtype != np.uint8:
                    maxv = arr.max()
                    if maxv <= 2.0:
                        arr = (arr * 255)
                    arr = arr.clip(0, 255).astype(np.uint8)
                
                # RGB to BGR for OpenCV
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


    def _show_rgb_sil(self, rgb_tensor, sil_tensor, pose_arr, gains):
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

            cap = self._make_caption(pose_vec, gains[i])
            tile = self._build_tile(rgb, cap)
            next_idx = self._blit_tile(canvas, next_idx, tile)

            # SIL tile
            sil = (sil_tensor[i] < .5).cpu().numpy().astype(np.uint8) * 255
            sil = cv2.cvtColor(sil, cv2.COLOR_GRAY2BGR)

            tile = self._build_tile(sil, cap)
            next_idx = self._blit_tile(canvas, next_idx, tile)

        cv2.imshow("Renders", canvas)
        cv2.waitKey(1)


    def _make_caption(self, pose_vec, gains):
        """Return a short text for the bottom strip."""
        x, y, z = pose_vec[:3]
        yaw = math.degrees(math.atan2(
            2 * (pose_vec[6] * pose_vec[5] + pose_vec[3] * pose_vec[4]),
            1 - 2 * (pose_vec[4] ** 2 + pose_vec[5] ** 2)
        ))
        cap = f"<{x:.2f},{y:.2f},{z:.2f},{yaw:+.0f} deg> G:{gains['gain']:.1f}"
        cap += f", EIG:{gains['eig']:.1f}, SIL:{gains['sil']:.1f}"
        return cap


    def _update_occupancy_window(self, z_slice=0.0, z_tol=0.15,
                                 cell=0.20, min_points=5):
        """Re-draw occupancy + slice-points"""
        if not self._grid_viz or self.params is None:
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
            img = grid_to_cv2(occ, scale=self.occ_scale)  # white = free, black = occ
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
                px = (x_rel * self.occ_scale).numpy()
                py = ((h - 1 - y_rel) * self.occ_scale).numpy()

                # Draw every point
                for xi, yi in zip(px.astype(np.int32), py.astype(np.int32)):
                    cv2.circle(
                        img, 
                        center=(xi, yi), 
                        radius=self.occ_pt_rad,
                        color=self.occ_pt_color,
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


    def compute_H_visited_inv(self):
        """Compute the inverse of the inverted Hessian matrix for all visited poses."""
        H_train = None
        num_visited = len(self.visited_poses)
        selected_poses = None

        #do monte carlo sampling
        if(self.monte_carlo):
            if num_visited <= self.N_monte_carlo:
                selected_poses = self.visited_poses
            else:
                indices = np.random.choice(num_visited, size=self.N_monte_carlo, replace=False)
                selected_poses = [self.visited_poses[i] for i in indices]
        else:
            #use all visited poses
            selected_poses = self.visited_poses

        for pose in selected_poses:
                cur_H = self.compute_Hessian( torch.linalg.inv(pose), return_points=True)
                if H_train is None:
                    H_train = torch.zeros(*cur_H.shape, device=cur_H.device, dtype=cur_H.dtype)
                H_train += cur_H
        self.H_train_inv = torch.reciprocal(H_train + 0.1)


    def compute_eig_score(self, pose):
        """Compute pose scores for the poses (w2c)."""
        cur_H = self.compute_Hessian(torch.linalg.inv(pose), return_points=True)
        view_score = torch.sum(cur_H * self.H_train_inv).item()
        return view_score


    @torch.enable_grad()
    def compute_Hessian(self, rel_w2c, return_points = False, return_pose = False):
        """Compute uncertainty at candidate pose
            
        Args:
            params: Gaussian slam params
            candidate_trans: (3,)
            candidate_rot: (4,)
            return_pose: if True, return the Hessian matrix with shape (N, C)
        
        Returns:
            if return_pose is True, the Hessian matrix is returned with shape (N, C), 
            else, it is flatten to 1-D.
        """
        if isinstance(rel_w2c, np.ndarray):
            rel_w2c = torch.from_numpy(rel_w2c).cuda()
        rel_w2c = rel_w2c.float()

        # Transform to candidate frame
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

        # For means3D, rotation won't change sum of square since R^T R = I
        rendervar['means2D'].retain_grad()
        im, _, _, = Renderer(raster_settings=self.cam, backward_power=2)(**rendervar)
        im.backward(gradient=torch.ones_like(im) * 1e-3)

        if return_points:
            cur_H = torch.cat([transformed_pts.grad.detach().reshape(num_points, -1),  
                                opacities.grad.detach().reshape(num_points, -1)], dim=1)

        else:
            cur_H = torch.cat([transformed_pts.grad.detach().reshape(-1), 
                                opacities.grad.detach().reshape(-1)])
            
        # Set grad to zero
        for k, v in rendervar.items():
            v.grad.fill_(0.)

        if not return_pose:
            return cur_H
        else:
            return cur_H, torch.eye(6).cuda()
        

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
