import astra
import matplotlib.pyplot as plt
import numpy as np

# parallel-beam geometry parameters
DETECTOR_SPACING = 1.0   # mm per detector pixel

SIRT_ITERATIONS = 100

# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

# makes a parallel-beam geometry for a given image size and angle set
def make_parallel_geom(size: int, angles: np.ndarray) -> tuple:
    
    vol_geom = astra.create_vol_geom(size, size)
    proj_geom = astra.create_proj_geom(
        "parallel",
        DETECTOR_SPACING,
        size,
        angles,
    )
    return vol_geom, proj_geom


# compute the forward projection of a single 2-D slice using ASTRA
def forward_project(slice_img: np.ndarray, angles: np.ndarray) -> np.ndarray:
    
    size = slice_img.shape[0]
    vol_geom, proj_geom = make_parallel_geom(size, angles)

    proj_id = astra.create_projector("line", proj_geom, vol_geom)

    # ASTRA expects sinogram and volume data to be stored in its own data objects
    # so we create those and copy our slice into the volume one
    vol_id  = astra.data2d.create("-vol",  vol_geom, data=slice_img)
    sino_id = astra.data2d.create("-sino", proj_geom)

    # configure and run the forward projection algorithm
    cfg = astra.astra_dict("FP")
    cfg["ProjectorId"]      = proj_id
    cfg["VolumeDataId"]     = vol_id
    cfg["ProjectionDataId"] = sino_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    sinogram = astra.data2d.get(sino_id)

    # clean up ASTRA objects to free memory
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(vol_id)
    astra.projector.delete(proj_id)

    return sinogram


# add Gaussian noise to a sinogram, with specified noise fraction.
def add_noise(sinogram: np.ndarray, noise_fraction: float) -> np.ndarray:
  
    if noise_fraction == 0.0:
        return sinogram.copy()
    sigma = noise_fraction * sinogram.max()
    noise = np.random.normal(0, sigma, sinogram.shape)
    return sinogram + noise


# ---------------------------------------------------------------------------
# Reconstruction algorithms (FBP and SIRT)
# ---------------------------------------------------------------------------

# reconstruct a single slice with FBP using ASTRA
def reconstruct_fbp(sinogram: np.ndarray, angles: np.ndarray) -> np.ndarray:
    
    size = sinogram.shape[1]
    vol_geom, proj_geom = make_parallel_geom(size, angles)

    proj_id = astra.create_projector("line", proj_geom, vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, data=sinogram)
    rec_id  = astra.data2d.create("-vol",  vol_geom)

    # configure and run the FBP algorithm
    cfg = astra.astra_dict("FBP")
    cfg["ProjectorId"]          = proj_id
    cfg["ProjectionDataId"]     = sino_id
    cfg["ReconstructionDataId"] = rec_id
    cfg["option"] = {"FilterType": "Ram-Lak"}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    reconstruction = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)

    return reconstruction

# reconstruct a single slice with SIRT using ASTRA
def reconstruct_sirt(sinogram: np.ndarray, angles: np.ndarray, iterations: int = SIRT_ITERATIONS) -> np.ndarray:
   
    size = sinogram.shape[1]
    vol_geom, proj_geom = make_parallel_geom(size, angles)

    proj_id = astra.create_projector("line", proj_geom, vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, data=sinogram)
    rec_id  = astra.data2d.create("-vol",  vol_geom)

    # configure and run the SIRT algorithm
    cfg = astra.astra_dict("SIRT")
    cfg["ProjectorId"]          = proj_id
    cfg["ProjectionDataId"]     = sino_id
    cfg["ReconstructionDataId"] = rec_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iterations)

    reconstruction = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)

    return reconstruction

# compute parallel-beam sinograms for every Z-slice of a 3D phantom, with optional noise
def sinogram_volume(phantom: np.ndarray, angles: np.ndarray, noise_fraction: float = 0.0) -> np.ndarray:
   
    num_slices = phantom.shape[0]
    size = phantom.shape[1]
    sinograms = np.zeros((num_slices, len(angles), size))

    for z in range(num_slices):
        sino = forward_project(phantom[z], angles)
        sinograms[z] = add_noise(sino, noise_fraction)

    return sinograms

# reconstruct every slice of a 3D volume with FBP
def reconstruct_volume_fbp(sinograms: np.ndarray, angles: np.ndarray) -> np.ndarray:
    num_slices = sinograms.shape[0]
    size = sinograms.shape[2]
    volume = np.zeros((num_slices, size, size))
    for z in range(num_slices):
        volume[z] = reconstruct_fbp(sinograms[z], angles)
    return volume

# reconstruct every slice of a 3D volume with SIRT
def reconstruct_volume_sirt(sinograms: np.ndarray, angles: np.ndarray, iterations: int = SIRT_ITERATIONS) -> np.ndarray:
    
    num_slices = sinograms.shape[0]
    size = sinograms.shape[2]
    volume = np.zeros((num_slices, size, size))
    for z in range(num_slices):
        volume[z] = reconstruct_sirt(sinograms[z], angles, iterations)
    return volume


# rmse between reference and reconstruction volumes
def rmse(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - reconstruction) ** 2)))