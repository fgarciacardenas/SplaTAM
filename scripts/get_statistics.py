import os
import sys
import time
import glob
import numpy as np
import os
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
from utils.plot_utils import plot_value_psnr, plot_combined_psnr, plot_eig_psnr_slice

data_paths = {
    'voxblox': [
        # Final
        '/home/dev/SplaTAM/experiments/habitat/final_vox1/',
        '/home/dev/SplaTAM/experiments/habitat/final_vox2/',
        '/home/dev/SplaTAM/experiments/habitat/final_vox3/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale40i1/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale40map60i3/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale20i0/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale20i1/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale20i2/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale8i0/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale8i1/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale8i2/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale3i0/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale3i1/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale3i2/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale1i0/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale1i1/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr10scale1i2/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr1scale1i0/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr1scale1i1/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr1scale1i2/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr05scale1i0/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr05scale1i1/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr05scale1i2/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr05scale1i2map60i5/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr025scale1i0/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr025scale1i1/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr025scale1i2/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr01scale1i0/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr01scale1i1/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr01scale1i2/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr005scale1i0/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr005scale1i1/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr005scale1i2/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr00scale1map50/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr01scale1map60i2/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr01scale1map60i3/',
    #     '/home/dev/SplaTAM/experiments/habitat/mediathr005scale1map60i4/'
    ],

    'silhouette': [
        # Final
        '/home/dev/SplaTAM/experiments/habitat/final_sil1/',
        '/home/dev/SplaTAM/experiments/habitat/final_sil2/',
        '/home/dev/SplaTAM/experiments/habitat/final_sil3/',
    #     '/home/dev/splatam/experiments/habitat/trajSIL00/',
    #     '/home/dev/splatam/experiments/habitat/trajSIL01/',
    #     '/home/dev/splatam/experiments/habitat/trajSIL02/',
    #     '/home/dev/splatam/experiments/habitat/trajSIL03/',
    #     '/home/dev/splatam/experiments/habitat/trajSIL04/',
    #     '/home/dev/splatam/experiments/habitat/trajSIL05/',

    #     #test silhouette
    #     '/home/dev/splatam/experiments/habitat/sil00/',
    #     '/home/dev/splatam/experiments/habitat/sil01/',
    #     '/home/dev/splatam/experiments/habitat/silh02/',
    #      #non linear gains
    #     '/home/dev/splatam/experiments/habitat/sil10/',
    #     '/home/dev/splatam/experiments/habitat/sil11/',
    #     '/home/dev/splatam/experiments/habitat/sil12/',
    #     #1200 gain
    #     '/home/dev/splatam/experiments/habitat/sil20/',
    #     '/home/dev/splatam/experiments/habitat/sil21/',
    #     '/home/dev/splatam/experiments/habitat/sil22/',
    #     # #test on mappping iterations with silhouette
    #      '/home/dev/splatam/experiments/habitat/sil30/',
    #     '/home/dev/splatam/experiments/habitat/sil31/',
    #     '/home/dev/splatam/experiments/habitat/sil32/',       
    #     #additional test on median and sil
    #     '/home/dev/splatam/experiments/habitat/mediathr025scale1i0sil/',
    #     '/home/dev/splatam/experiments/habitat/mediathr025scale1i1sil/',
    #     '/home/dev/splatam/experiments/habitat/mediathr025scale1i2sil/',

    #     '/home/dev/splatam/experiments/habitat/mediathr1scale1i0sil/',
    #     '/home/dev/splatam/experiments/habitat/mediathr1scale1i1sil/',
    #     '/home/dev/splatam/experiments/habitat/mediathr1scale1i2sil/',
    ],

    'fisher': [
        # Final
        # '/home/dev/SplaTAM/experiments/habitat/final_eig1/',
        # '/home/dev/SplaTAM/experiments/habitat/final_eig2/',
        # '/home/dev/SplaTAM/experiments/habitat/final_eig3/',
        '/home/dev/SplaTAM/experiments/habitat/final_eig1_nl/',
        '/home/dev/SplaTAM/experiments/habitat/final_eig2_nl/',
        '/home/dev/SplaTAM/experiments/habitat/final_eig3_nl/',

        # '/home/dev/splatam/experiments/habitat/trajFISH00/',
        # '/home/dev/splatam/experiments/habitat/trajFISH01/',
        # '/home/dev/splatam/experiments/habitat/trajFISH02/',
        # '/home/dev/splatam/experiments/habitat/trajFISH03/',
        # '/home/dev/splatam/experiments/habitat/trajFISH04/',
        # #non linear gains
        # '/home/dev/splatam/experiments/habitat/fish10/',
        # '/home/dev/splatam/experiments/habitat/fish11/',
        # '/home/dev/splatam/experiments/habitat/fish12/',
        #monte carlo
        # '/home/dev/SplaTAM/experiments/habitat/monte00/',
        # '/home/dev/SplaTAM/experiments/habitat/monte01/',
        # '/home/dev/SplaTAM/experiments/habitat/monte02/',
        # '/home/dev/SplaTAM/experiments/habitat/monte10/',
        # '/home/dev/SplaTAM/experiments/habitat/monte12/',
        # '/home/dev/SplaTAM/experiments/habitat/monte20/',
        # '/home/dev/SplaTAM/experiments/habitat/monte21/',
        # '/home/dev/SplaTAM/experiments/habitat/monte22/',
        # '/home/dev/SplaTAM/experiments/habitat/monte30/',
        # '/home/dev/SplaTAM/experiments/habitat/monte31/',
        # '/home/dev/SplaTAM/experiments/habitat/monte32/',
        # Sum
    ],

    'sum': [
        # Final
        '/home/dev/SplaTAM/experiments/habitat/final_sum/',
        '/home/dev/SplaTAM/experiments/habitat/final_sum0/',
        '/home/dev/SplaTAM/experiments/habitat/final_sum1/',
    ]
}


def get_stats(dpath_list: str) -> None:
    # Create objects to hold data
    eig_arr   = []
    l1_arr    = []
    lpips_arr = []
    psnr_arr  = []
    rmse_arr  = []
    ssim_arr  = []
    
    # Read data files
    for dpath in dpath_list:
        eig_arr.append(np.loadtxt(dpath + 'eval/eig.txt', skiprows=1))
        l1_arr.append(np.loadtxt(dpath + 'eval/l1.txt', skiprows=1))
        lpips_arr.append(np.loadtxt(dpath + 'eval/lpips.txt', skiprows=1))
        psnr_arr.append(np.loadtxt(dpath + 'eval/psnr.txt', skiprows=1))
        rmse_arr.append(np.loadtxt(dpath + 'eval/rmse.txt', skiprows=1))
        ssim_arr.append(np.loadtxt(dpath + 'eval/ssim.txt', skiprows=1))

    # Concatenate arrays
    eig   = np.concatenate(eig_arr)
    l1    = np.concatenate(l1_arr)
    lpips = np.concatenate(lpips_arr)
    psnr  = np.concatenate(psnr_arr)
    rmse  = np.concatenate(rmse_arr)
    ssim  = np.concatenate(ssim_arr)
    
    # Compute statistics
    mean_eig   = np.mean(eig)
    mean_l1    = np.mean(l1)
    mean_lpips = np.mean(lpips)
    mean_psnr  = np.mean(psnr)
    mean_rmse  = np.mean(rmse)
    mean_ssim  = np.mean(ssim)

    std_eig   = np.std(eig)
    std_l1    = np.std(l1)
    std_lpips = np.std(lpips)
    std_psnr  = np.std(psnr)
    std_rmse  = np.std(rmse)
    std_ssim  = np.std(ssim)

    # Generate output strings
    output_str = f"Mean EIG: {mean_eig:.4f}, Std EIG: {std_eig:.4f}\n"
    output_str += f"Mean L1: {mean_l1:.4f}, Std L1: {std_l1:.4f}\n"
    output_str += f"Mean LPIPS: {mean_lpips:.4f}, Std LPIPS: {std_lpips:.4f}\n"
    output_str += f"Mean PSNR: {mean_psnr:.4f}, Std PSNR: {std_psnr:.4f}\n"
    output_str += f"Mean RMSE: {mean_rmse:.4f}, Std RMSE: {std_rmse:.4f}\n"
    output_str += f"Mean SSIM: {mean_ssim:.4f}, Std SSIM: {std_ssim:.4f}\n"

    # Print statistics
    print(output_str)

    # Return the statistics
    return output_str


def get_running_stats(dpath_list: str, save_path: str, key: str) -> None:
    # Create objects to hold data
    sil_arr   = []
    eig_arr   = []
    psnr_arr  = []

    # Read data files
    for output_path in dpath_list:
        fname = glob.glob(output_path + '/running_metrics_*')[0]
        data = np.loadtxt(fname, delimiter=",", skiprows=1)

        # Slice the 2-D array into 1-D arrays
        sil_list  = data[:, 0]
        eig_list  = data[:, 1]
        psnr_list = data[:, 2]
        #gain_list = data[:, 3]

        sil_arr.append(sil_list)
        eig_arr.append(eig_list)
        psnr_arr.append(psnr_list)

    # Concatenate arrays
    sil   = np.concatenate(sil_arr)
    eig   = np.concatenate(eig_arr)
    psnr  = np.concatenate(psnr_arr)

    # Compute statistics
    mean_sil  = np.mean(sil)
    mean_eig  = np.mean(eig)
    mean_psnr = np.mean(psnr)

    std_sil  = np.std(sil)
    std_eig  = np.std(eig)
    std_psnr = np.std(psnr)

    # Generate output strings
    output_str = f"Mean SIL (running): {mean_sil:.4f}, Std SIL: {std_sil:.4f}\n"
    output_str += f"Mean EIG (running): {mean_eig:.4f}, Std EIG: {std_eig:.4f}\n"
    output_str += f"Mean PSNR (running): {mean_psnr:.4f}, Std PSNR: {std_psnr:.4f}\n"

    # Print statistics
    print(output_str)

    # Plot <value> vs PSNR
    plot_value_psnr(psnr, sil, axis_name="SIL", prefix="psnr_sil", save_dir=save_path + f'psnr_plots_{key}')
    plot_value_psnr(psnr, eig, axis_name="EIG", prefix="psnr_eig", save_dir=save_path + f'psnr_plots_{key}')
    plot_combined_psnr(psnr, sil, eig, prefix="psnr_combined", save_dir=save_path + f'psnr_plots_{key}')
    plot_eig_psnr_slice(psnr, eig, sil, thr=300, axis_name="EIG", prefix="psnr_eig", save_dir=save_path + f'psnr_plots_{key}')

    # Return the statistics
    return output_str


def main():
    # Create objects to hold data
    output_path = "/home/dev/SplaTAM/experiments/habitat/run_stats/"

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Record the start time
    _time = time.time_ns()

    # Extract statistics for each data path
    for key, paths in data_paths.items():
        print(f"\n\n------ Statistics for {key} data ------")
        stats_str = get_stats(paths)
        running_str = get_running_stats(paths, output_path, key)

        # Store statistics in a text file
        output_file = os.path.join(output_path, f"statistics_{_time}.txt")
        with open(output_file, 'a') as f:
            f.write(f"\n\n------ Statistics for {key} data ------\n")
            f.write(stats_str)
            f.write(running_str)


if __name__ == "__main__":
    main()
