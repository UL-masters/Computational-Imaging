import astra
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim_fn

DETECTOR_SPACING = 1.0
SIRT_ITERATIONS  = 100


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def make_parallel_geom(size: int, angles: np.ndarray) -> tuple:
    vol_geom  = astra.create_vol_geom(size, size)
    proj_geom = astra.create_proj_geom("parallel", DETECTOR_SPACING, size, angles)
    return vol_geom, proj_geom


# ---------------------------------------------------------------------------
# Forward projection & noise
# ---------------------------------------------------------------------------

def forward_project(slice_img: np.ndarray, angles: np.ndarray) -> np.ndarray:
    size = slice_img.shape[0]
    vol_geom, proj_geom = make_parallel_geom(size, angles)
    proj_id = astra.create_projector("line", proj_geom, vol_geom)
    vol_id  = astra.data2d.create("-vol",  vol_geom, data=slice_img)
    sino_id = astra.data2d.create("-sino", proj_geom)

    cfg = astra.astra_dict("FP")
    cfg["ProjectorId"]      = proj_id
    cfg["VolumeDataId"]     = vol_id
    cfg["ProjectionDataId"] = sino_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    sinogram = astra.data2d.get(sino_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(vol_id)
    astra.projector.delete(proj_id)
    return sinogram


def add_noise(sinogram: np.ndarray, noise_fraction: float) -> np.ndarray:
    if noise_fraction == 0.0:
        return sinogram.copy()
    sigma = noise_fraction * sinogram.max()
    return sinogram + np.random.normal(0, sigma, sinogram.shape)


def sinogram_volume(phantom: np.ndarray, angles: np.ndarray,
                    noise_fraction: float = 0.0) -> np.ndarray:
    num_slices = phantom.shape[0]
    size       = phantom.shape[1]
    sinograms  = np.zeros((num_slices, len(angles), size))
    for z in range(num_slices):
        sino = forward_project(phantom[z], angles)
        sinograms[z] = add_noise(sino, noise_fraction)
    return sinograms


# ---------------------------------------------------------------------------
# Reconstruction algorithms
# ---------------------------------------------------------------------------

def reconstruct_fbp(sinogram: np.ndarray, angles: np.ndarray) -> np.ndarray:
    size = sinogram.shape[1]
    vol_geom, proj_geom = make_parallel_geom(size, angles)
    proj_id = astra.create_projector("line", proj_geom, vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, data=sinogram)
    rec_id  = astra.data2d.create("-vol",  vol_geom)

    cfg = astra.astra_dict("FBP")
    cfg["ProjectorId"]          = proj_id
    cfg["ProjectionDataId"]     = sino_id
    cfg["ReconstructionDataId"] = rec_id
    cfg["option"] = {"FilterType": "Ram-Lak"}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    reconstruction = np.clip(astra.data2d.get(rec_id), 0, None)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)
    return reconstruction


def reconstruct_sirt(sinogram: np.ndarray, angles: np.ndarray,
                     iterations: int = SIRT_ITERATIONS) -> np.ndarray:
    size = sinogram.shape[1]
    vol_geom, proj_geom = make_parallel_geom(size, angles)
    proj_id = astra.create_projector("line", proj_geom, vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, data=sinogram)
    rec_id  = astra.data2d.create("-vol",  vol_geom)

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


def reconstruct_sirt_tv(sinogram: np.ndarray, angles: np.ndarray,
                        iterations: int = SIRT_ITERATIONS,
                        lam: float = 0.01,
                        chunk: int = 10) -> np.ndarray:
    """
    SIRT with Total Variation denoising applied every `chunk` iterations.
    Running SIRT in chunks of `chunk` iterations rather than 1 at a time
    dramatically reduces ASTRA overhead while still applying TV regularly.
    """
    size = sinogram.shape[1]
    vol_geom, proj_geom = make_parallel_geom(size, angles)
    proj_id = astra.create_projector("line", proj_geom, vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, data=sinogram)
    rec_id  = astra.data2d.create("-vol",  vol_geom)

    current   = np.zeros((size, size))
    remaining = iterations

    while remaining > 0:
        n = min(chunk, remaining)

        astra.data2d.store(rec_id, current)
        cfg = astra.astra_dict("SIRT")
        cfg["ProjectorId"]          = proj_id
        cfg["ProjectionDataId"]     = sino_id
        cfg["ReconstructionDataId"] = rec_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, n)
        astra.algorithm.delete(alg_id)

        current    = astra.data2d.get(rec_id)
        current    = denoise_tv_chambolle(current, weight=lam)
        remaining -= n

    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)
    return current


# ---------------------------------------------------------------------------
# Volume-level wrappers
# ---------------------------------------------------------------------------

def reconstruct_volume_fbp(sinograms: np.ndarray, angles: np.ndarray) -> np.ndarray:
    num_slices = sinograms.shape[0]
    size       = sinograms.shape[2]
    volume     = np.zeros((num_slices, size, size))
    for z in range(num_slices):
        volume[z] = reconstruct_fbp(sinograms[z], angles)
    return volume


def reconstruct_volume_sirt(sinograms: np.ndarray, angles: np.ndarray,
                             iterations: int = SIRT_ITERATIONS) -> np.ndarray:
    num_slices = sinograms.shape[0]
    size       = sinograms.shape[2]
    volume     = np.zeros((num_slices, size, size))
    for z in range(num_slices):
        volume[z] = reconstruct_sirt(sinograms[z], angles, iterations)
    return volume


def reconstruct_volume_sirt_tv(sinograms: np.ndarray, angles: np.ndarray,
                                iterations: int = SIRT_ITERATIONS,
                                lam: float = 0.01) -> np.ndarray:
    num_slices = sinograms.shape[0]
    size       = sinograms.shape[2]
    volume     = np.zeros((num_slices, size, size))
    for z in range(num_slices):
        volume[z] = reconstruct_sirt_tv(sinograms[z], angles, iterations, lam)
    return volume


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    """Root Mean Squared Error over the full 3D volume."""
    return float(np.sqrt(np.mean((reference - reconstruction) ** 2)))


def ssim_score(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    """
    Mean SSIM over all axial slices.
    data_range is set from the reference volume so it is consistent
    across conditions.
    """
    data_range = float(reference.max() - reference.min())
    scores = [
        ssim_fn(reference[z], reconstruction[z], data_range=data_range)
        for z in range(reference.shape[0])
    ]
    return float(np.mean(scores))


def dice_score(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    """
    Dice coefficient between Otsu-thresholded binary masks of the reference
    and the reconstruction.

    Otsu's method is applied separately to the reference (ground truth stone
    mask) and the reconstruction so that the threshold adapts to the
    reconstruction's contrast level — mimicking a realistic detection scenario
    where the ground truth threshold is not known in advance.

    Returns 0.0 if either mask is empty (no stone present / not detected).
    """
    # threshold each volume independently
    gt_thresh   = threshold_otsu(reference)
    pred_thresh = threshold_otsu(reconstruction)

    gt_mask   = reference       > gt_thresh
    pred_mask = reconstruction  > pred_thresh

    intersection = float((gt_mask & pred_mask).sum())
    denom        = float(gt_mask.sum() + pred_mask.sum())

    if denom == 0:
        return 0.0
    return 2.0 * intersection / denom