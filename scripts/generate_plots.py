import os
import sys
import glob
import argparse
import numpy as np
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
from utils.plot_utils import plot_value_psnr, plot_combined_psnr


def main(dpath: str):
    # Read the file, skipping the header row we wrote earlier
    fname = glob.glob(dpath + '/running_metrics_*')[0]
    data = np.loadtxt(fname, delimiter=",", skiprows=1)

    # Slice the 2-D array into 1-D arrays
    sil_list  = data[:, 0]
    eig_list  = data[:, 1]
    psnr_list = data[:, 2]
    gain_list = data[:, 3]

    # Plot <value> vs PSNR
    plot_value_psnr(psnr_list, sil_list, axis_name="SIL", prefix="psnr_sil", save_dir=dpath + '/psnr_plots')
    plot_value_psnr(psnr_list, eig_list, axis_name="EIG", prefix="psnr_eig", save_dir=dpath + '/psnr_plots')
    plot_value_psnr(psnr_list, gain_list, axis_name="GAIN", prefix="psnr_gain", save_dir=dpath + '/psnr_plots')
    plot_combined_psnr(psnr_list, sil_list, eig_list, prefix="psnr_combined", save_dir=dpath + '/psnr_plots')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reload and display a saved figure.")
    parser.add_argument("dpath", type=str, help="Path to the stored data file.")
    args = parser.parse_args()
    main(args.dpath)
    