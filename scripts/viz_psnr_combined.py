import pickle
import matplotlib.pyplot as plt
import argparse

def main(fig_path: str = "/home/dev/.../psnr_combined_1727946954.fig.pkl", save_fig: bool = True) -> None:
    # Load the figure from a pickle file
    with open(fig_path, "rb") as f:
        fig = pickle.load(f)

    # Display the figure
    plt.show()

    # Save the figure if required
    if save_fig:
        plt.savefig(f"{fig_path[:-8]}_reloaded.png", dpi=300)
        print("Figure saved after reloading.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reload and display a saved figure.")
    parser.add_argument("fig_path", type=str, help="Path to the saved figure pickle file.")
    parser.add_argument("--save_fig", action="store_true", help="Save the figure after reloading.")
    args = parser.parse_args()
    
    main(fig_path=args.fig_path, save_fig=args.save_fig)
    