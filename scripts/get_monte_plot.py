import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt


data_paths = {
    'no_monte': [
        #'/home/dev/SplaTAM/experiments/habitat/monte00/',
        '/home/dev/SplaTAM/experiments/habitat/monte01/',
        '/home/dev/SplaTAM/experiments/habitat/monte02/'
    ],

    'monte_25': [
        '/home/dev/SplaTAM/experiments/habitat/monte10/',
        '/home/dev/SplaTAM/experiments/habitat/monte12/'
    ],

    'monte_40': [
        '/home/dev/SplaTAM/experiments/habitat/monte20/',
        '/home/dev/SplaTAM/experiments/habitat/monte21/',
        '/home/dev/SplaTAM/experiments/habitat/monte22/'
    ],

    'monte_55': [
        '/home/dev/SplaTAM/experiments/habitat/monte30/',
        '/home/dev/SplaTAM/experiments/habitat/monte31/',
        '/home/dev/SplaTAM/experiments/habitat/monte32/'
    ]
}


def plot_monte_slice(
    data: dict,
    thr: float = 100.0,
    axis_name: str = "EIG",
    save_dir: str = "/home/dev/splatam/experiments/",
    prefix: str = "psnr_eig",
) -> None:
    """
    Plot <value> vs PSNR.
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Iterate through each dataset in the data dictionary
    for key, values in data.items():
        sil_arr = values['sil']
        eig_arr = values['eig']
        psnr_arr = values['psnr']

        # Retrieve indexes where silhouette is lower than threshold
        sil_mask = np.array(sil_arr) < thr

        # Scatter plot for the current dataset
        ax.scatter(psnr_arr[sil_mask], eig_arr[sil_mask], alpha=0.5, label=key)

    # Set the axis limits
    ax.dataLim.update_from_data_xy(np.column_stack([data['no_monte']['psnr'], data['no_monte']['eig']]))
    ax.autoscale_view()

    # Set labels and title
    ax.set_xlabel("PSNR")
    ax.set_ylabel(f"{axis_name}")
    ax.set_title(f"{axis_name} vs PSNR")

    # Add a legend
    ax.legend(title="Dataset", loc='upper right')

    # Save the figure
    fig.tight_layout()
    fname = os.path.join(save_dir, f"{prefix}_sliced.pdf")
    fig.savefig(fname, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


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


def get_running_stats(dpath_list: str) -> None:
    # Create objects to hold data
    sil_arr   = []
    eig_arr   = []
    psnr_arr  = []

    # Read data files
    for dpath in dpath_list:
        fname = glob.glob(dpath + '/running_metrics_*')[0]
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

    # Return the concatenated arrays
    return sil, eig, psnr


def main():
    # Create objects to hold data
    data = {}
    output_path = "/home/dev/SplaTAM/experiments/habitat/monte_stats/"

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Record the start time
    _time = time.time_ns()

    # Extract statistics for each data path
    for key, paths in data_paths.items():
        print(f"\n\n------ Statistics for {key} data ------")
        output_str = get_stats(paths)
        sil, eig, psnr = get_running_stats(paths)
        data[key] = {
            'sil': sil,
            'eig': eig,
            'psnr': psnr
        }

        # Store statistics in a text file
        output_file = os.path.join(output_path, f"statistics_{_time}.txt")
        with open(output_file, 'a') as f:
            f.write(f"\n\n------ Statistics for {key} data ------\n")
            f.write(output_str)

    # Plot EIG vs PSNR with silhouette thresholding
    plot_monte_slice(data, thr=1000.0, save_dir=output_path, prefix=f"psnr_eig_{_time}")


if __name__ == "__main__":
    main()
