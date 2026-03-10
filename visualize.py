import numpy as np
import matplotlib.pyplot as plt
import astra
import os


def load_phantom(path):
    phantom = np.load(path)
    print("Phantom shape:", phantom.shape)
    
    filename = os.path.basename(path)
    phantom_number = filename.split("_")[1].split(".")[0]

    print(f"Loaded phantom {phantom_number}")
    print("Shape:", phantom.shape)
    return phantom, phantom_number


def show_slice(volume, title="Volume"):
    slice_idx = volume.shape[0] // 2
    plt.imshow(volume[slice_idx], cmap="gray")
    plt.colorbar()
    plt.title(f"{title} (slice {slice_idx})")
    plt.show()

def reconstruct_volume(phantom):

    size = phantom.shape[1]
    angles = np.linspace(0, np.pi, 180)

    vol_geom = astra.create_vol_geom(size, size)
    proj_geom = astra.create_proj_geom('parallel', 1.0, size, angles)

    projector_id = astra.create_projector('linear', proj_geom, vol_geom)

    reconstruction = np.zeros_like(phantom)

    for z in range(phantom.shape[0]):

        slice_img = phantom[z]

        # forward projection
        sino_id, sinogram = astra.create_sino(slice_img, projector_id)

        # reconstruction container
        rec_id = astra.data2d.create('-vol', vol_geom)

        cfg = astra.astra_dict('FBP')
        cfg['ProjectorId'] = projector_id    
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = rec_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        reconstruction[z] = astra.data2d.get(rec_id)

        # cleanup
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)

    astra.projector.delete(projector_id)

    return reconstruction


def main():

    # change the phantom number to visualize different phantoms (000-110)
    phantom, phantom_number = load_phantom("phantoms/phantom_001.npy")

    show_slice(phantom, f"Original Phantom {phantom_number}")

    reconstruction = reconstruct_volume(phantom)

    show_slice(reconstruction, f"Reconstruction {phantom_number}")


if __name__ == "__main__":
    main()