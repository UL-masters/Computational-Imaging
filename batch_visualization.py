import os
import numpy as np
import matplotlib.pyplot as plt


# Load phantom and extract number
def load_phantom(path: str):
    phantom = np.load(path)
    filename = os.path.basename(path)
    phantom_number = filename.split("_")[1].split(".")[0]
    print(f"Loaded phantom_{phantom_number}  shape: {phantom.shape}")
    return phantom, phantom_number


# Show and save slices
def show_slices(volume: np.ndarray, title: str = "Volume", save_path: str | None = None) -> None:
    mid_z = volume.shape[0] // 2
    mid_y = volume.shape[1] // 2
    mid_x = volume.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    views = [
        (volume[mid_z],    f"Axial  (z={mid_z})"),
        (volume[:, mid_y, :], f"Coronal (y={mid_y})"),
        (volume[:, :, mid_x], f"Sagittal (x={mid_x})"),
    ]

    vmin, vmax = volume.min(), volume.max()

    for ax, (img, label) in zip(axes, views):
        im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)  


def main():
    input_folder = "phantoms"
    output_folder = "phantoms_overview"
    
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder created: {os.path.abspath(output_folder)}")

    # Loop through phantoms 001 to 110
    for i in range(0, 111):                    # 1 to 110 inclusive
        phantom_filename = f"phantom_{i:03d}.npy"   
        phantom_path = os.path.join(input_folder, phantom_filename)

        if not os.path.exists(phantom_path):
            print(f" Skipping (not found): {phantom_path}")
            continue

        try:
            phantom, phantom_number = load_phantom(phantom_path)
            
            save_path = os.path.join(output_folder, f"phantom_{phantom_number}_overview.png")
            
            show_slices(phantom, f"Phantom {phantom_number}", save_path=save_path)
            
        except Exception as e:
            print(f"Error processing phantom_{i:03d}: {e}")

    print("\nFinished processing all phantoms!")


if __name__ == "__main__":
    main()