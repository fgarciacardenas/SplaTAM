import numpy as np

data_paths = {
    'voxblox': [
        '/home/dev/splatam/experiments/habitat/room-final04_0/',
        '/home/dev/splatam/experiments/habitat/room-final05_0/'
    ],

    'silhouette': [
        '/home/dev/splatam/experiments/habitat/room-final04_0/',
        '/home/dev/splatam/experiments/habitat/room-final05_0/'
    ],

    'fisher': [
        '/home/dev/splatam/experiments/habitat/room-final04_0/',
        '/home/dev/splatam/experiments/habitat/room-final05_0/'
    ]
}


def get_stats(dpath: str) -> None:
    # Read data files
    eig   = np.loadtxt(dpath +   'eval/eig.txt', skiprows=1)
    l1    = np.loadtxt(dpath +    'eval/l1.txt', skiprows=1)
    lpips = np.loadtxt(dpath + 'eval/lpips.txt', skiprows=1)
    psnr  = np.loadtxt(dpath +  'eval/psnr.txt', skiprows=1)
    rmse  = np.loadtxt(dpath +  'eval/rmse.txt', skiprows=1)
    ssim  = np.loadtxt(dpath +  'eval/ssim.txt', skiprows=1)

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

    # Print statistics
    print(f"Mean EIG: {mean_eig:.4f}, Std EIG: {std_eig:.4f}")
    print(f"Mean L1: {mean_l1:.4f}, Std L1: {std_l1:.4f}")
    print(f"Mean LPIPS: {mean_lpips:.4f}, Std LPIPS: {std_lpips:.4f}")
    print(f"Mean PSNR: {mean_psnr:.4f}, Std PSNR: {std_psnr:.4f}")
    print(f"Mean RMSE: {mean_rmse:.4f}, Std RMSE: {std_rmse:.4f}")
    print(f"Mean SSIM: {mean_ssim:.4f}, Std SSIM: {std_ssim:.4f}")


def main():
    for key, paths in data_paths.items():
        print(f"\n\n------ Statistics for {key} data ------")
        for dpath in paths:
            print(f"\nProcessing directory: {dpath}")
            get_stats(dpath)


if __name__ == "__main__":
    main()
