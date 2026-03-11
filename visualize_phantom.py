import numpy as np
import matplotlib.pyplot as plt
import os


def load_phantom(path):
    phantom = np.load(path)
    print("Phantom shape:", phantom.shape)
    
    filename = os.path.basename(path)
    phantom_number = filename.split("_")[1].split(".")[0]

    # get phantom number from filename
    print(f"Loaded phantom {phantom_number}")
    print("Shape:", phantom.shape)
    return phantom, phantom_number


def show_slice(volume, title="Volume"):
    
    # show the middle slice of volume
    slice_idx = volume.shape[0] // 2
    plt.imshow(volume[slice_idx], cmap="gray")
    plt.colorbar()
    plt.title(f"{title} (slice {slice_idx})")
    plt.show()


def main():

    # change the phantom number to visualize different phantoms (000-110)
    phantom, phantom_number = load_phantom("phantoms/phantom_003.npy")

    show_slice(phantom, f"Original Phantom {phantom_number}")


if __name__ == "__main__":
    main()