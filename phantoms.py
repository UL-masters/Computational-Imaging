# import numpy as np
# from scipy.ndimage import gaussian_filter
# import os

# def generate_phantom(volume_size=(128, 128, 128), num_stones=0, base_att=0.1, stone_att=0.5):
    
#     # coordinate grid centered at origin
#     half_size = np.array(volume_size) / 2
#     x, y, z = np.ogrid[-half_size[0]:half_size[0], -half_size[1]:half_size[1], -half_size[2]:half_size[2]]
    
#     # deformed ellipsoid with random scaling
#     scale_factors = np.random.uniform(0.8, 1.2, 3)
#     a, b, c = half_size * scale_factors
#     base_mask = (x**2 / a**2 + y**2 / b**2 + z**2 / c**2 <= 1)
    
#     # Gaussian noise and smoothing
#     base = base_mask.astype(float) * base_att
#     noise = np.random.normal(0, 0.05, volume_size)
#     base += noise * base_mask
#     base = gaussian_filter(base, sigma=2)
    
#     base = (base > 0.05).astype(float) * base_att
    
#     # add stones
#     stones = np.zeros(volume_size)
#     for _ in range(num_stones):
#         # random position inside base
#         while True:
#             px = np.random.uniform(-a * 0.9, a * 0.9)
#             py = np.random.uniform(-b * 0.9, b * 0.9)
#             pz = np.random.uniform(-c * 0.9, c * 0.9)
#             if (px**2 / a**2 + py**2 / b**2 + pz**2 / c**2 < 0.8):
#                 break
        
#         # stone - small ellipsoid with random size (5-15 voxels radius)
#         sa, sb, sc = np.random.uniform(5, 15, 3)
#         stone_mask = ((x - px)**2 / sa**2 + (y - py)**2 / sb**2 + (z - pz)**2 / sc**2 <= 1)
#         stones += stone_mask.astype(float) * stone_att
    
#     phantom = base + stones
#     return phantom

# def generate_all_phantoms(output_dir='phantoms', volume_size=(128, 128, 128)):
   
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # 3 with 3 stones, 35 with 2, 62 with 1, 11 with 0
#     num_stones_list = [3] * 3 + [2] * 35 + [1] * 62 + [0] * 11
#     np.random.shuffle(num_stones_list)  # for randomness
    
#     for i, num_stones in enumerate(num_stones_list):
#         phantom = generate_phantom(volume_size, num_stones)
#         np.save(os.path.join(output_dir, f'phantom_{i:03d}.npy'), phantom)
#         print(f'Generated phantom {i:03d} with {num_stones} stones')

# if __name__ == '__main__':
#     generate_all_phantoms()

import numpy as np
from scipy.ndimage import gaussian_filter, rotate
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# random rotation matrix generator (for rotating the base object)
def _random_rotation_matrix():
    A = np.random.randn(3, 3)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _make_playdoh_base(volume_size, base_att=0.1):
    half = np.array(volume_size) / 2

    xs = np.linspace(-1, 1, volume_size[0])
    ys = np.linspace(-1, 1, volume_size[1])
    zs = np.linspace(-1, 1, volume_size[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    # --- 1. Base ellipsoid (smooth core shape) ---
    a, b, c = np.random.uniform(0.6, 0.9, 3)
    ellipsoid = (X/a)**2 + (Y/b)**2 + (Z/c)**2

    # --- 2. Add smooth deformation (this is the magic) ---
    noise = gaussian_filter(np.random.randn(*volume_size), sigma=8)
    noise = noise / (np.max(np.abs(noise)) + 1e-8)

    # control how "blobby" it is
    deformation_strength = np.random.uniform(0.15, 0.35)

    field = ellipsoid + deformation_strength * noise

    # --- 3. Threshold to get shape ---
    mask = field <= 1.0

    # --- 4. Smooth edges ---
    mask = gaussian_filter(mask.astype(float), sigma=2)
    mask = mask > 0.5

    # --- 5. Add slight surface texture ---
    base = mask.astype(float) * base_att
    fine_noise = np.random.normal(0, 0.01, volume_size)
    base += fine_noise * mask

    return base, mask

# ---------------------------------------------------------------------------
# Rough ellipsoid stone
# ---------------------------------------------------------------------------

def _make_rough_ellipsoid_stone(volume_size, center, radii, stone_att=0.5,
                                roughness_amplitude=0.25, roughness_freq=3):
    """
    Create a stone mask as a rough ellipsoid.

    Roughness is implemented by modulating the ellipsoid boundary with
    a smooth low-frequency 3-D sinusoidal perturbation (Perlin-lite):
    for each voxel (x, y, z) the "radius fraction"

        r = sqrt((dx/a)^2 + (dy/b)^2 + (dz/c)^2)

    is compared to a threshold

        thresh = 1 + A * sin(f*dx/a + φx) * sin(f*dy/b + φy) * sin(f*dz/c + φz)

    so the boundary ripples inward and outward.

    Parameters
    ----------
    center       : (px, py, pz) in voxel coordinates
    radii        : (sa, sb, sc) semi-axes in voxels
    roughness_amplitude : fraction of radius to perturb (0 = smooth ellipsoid)
    roughness_freq      : spatial frequency of surface ripple
    """
    half = np.array(volume_size) / 2
    xs = np.arange(volume_size[0]) - half[0]
    ys = np.arange(volume_size[1]) - half[1]
    zs = np.arange(volume_size[2]) - half[2]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    px, py, pz = center
    sa, sb, sc = radii

    dx = (X - px) / sa
    dy = (Y - py) / sb
    dz = (Z - pz) / sc

    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # Random phase offsets for the ripple
    phi = np.random.uniform(0, 2 * np.pi, 3)
    ripple = (
        np.sin(roughness_freq * dx + phi[0]) *
        np.sin(roughness_freq * dy + phi[1]) *
        np.sin(roughness_freq * dz + phi[2])
    )
    # Normalise ripple to [-1, 1] and scale
    ripple_max = np.max(np.abs(ripple)) + 1e-8
    ripple = ripple / ripple_max

    threshold = 1.0 + roughness_amplitude * ripple

    stone_mask = r <= threshold
    return stone_mask.astype(float) * stone_att


# ---------------------------------------------------------------------------
# Main phantom generator
# ---------------------------------------------------------------------------

def generate_phantom(volume_size=(128, 128, 128),
                     num_stones=0,
                     base_att=0.1,
                     stone_att=0.5):
    """
    Generate a single phantom volume.

    Base object : corner-cut rotated cube (Play-Doh style, per Section 4.8).
    Stones      : rough ellipsoids (ellipsoid + low-freq surface ripple).

    Parameters
    ----------
    volume_size : (int, int, int)
    num_stones  : number of stone inclusions
    base_att    : attenuation value for base object
    stone_att   : attenuation value for stone(s)

    Returns
    -------
    phantom : np.ndarray of shape volume_size, dtype float64
    """
    base, base_mask = _make_playdoh_base(volume_size, base_att=base_att)

    half = np.array(volume_size) / 2
    cube_half = half * 0.5   # same as used inside _make_corner_cut_base

    stones = np.zeros(volume_size)
    for _ in range(num_stones):
        # Place stone at a random position inside the (unrotated) cube core.
        # We accept positions whose ellipsoidal distance to the centre < 0.75
        # to avoid stones poking out of the base.
        for _attempt in range(200):
            px = np.random.uniform(-cube_half[0] * 0.75, cube_half[0] * 0.75)
            py = np.random.uniform(-cube_half[1] * 0.75, cube_half[1] * 0.75)
            pz = np.random.uniform(-cube_half[2] * 0.75, cube_half[2] * 0.75)
            # Check that the centre voxel is inside the base
            ix = int(px + half[0])
            iy = int(py + half[1])
            iz = int(pz + half[2])
            ix = np.clip(ix, 0, volume_size[0] - 1)
            iy = np.clip(iy, 0, volume_size[1] - 1)
            iz = np.clip(iz, 0, volume_size[2] - 1)
            if base_mask[ix, iy, iz]:
                break

        # Stone radii: 5–15 voxels (matching paper's 3–11 mm at ~1 mm/voxel)
        sa, sb, sc = np.random.uniform(5, 15, 3)

        # Surface roughness: amplitude 20–35 % of radius, freq 2–5
        amp = np.random.uniform(0.20, 0.35)
        freq = np.random.uniform(2, 5)

        stone = _make_rough_ellipsoid_stone(
            volume_size, (px, py, pz), (sa, sb, sc),
            stone_att=stone_att,
            roughness_amplitude=amp,
            roughness_freq=freq,
        )
        stones += stone

    phantom = base + stones
    return phantom


# ---------------------------------------------------------------------------
# Batch generator — matches paper's dataset exactly
# ---------------------------------------------------------------------------

def generate_all_phantoms(output_dir='phantoms', volume_size=(128, 128, 128)):
    """
    Generate 111 phantoms with the exact stone distribution from the paper:
      3  objects with 3 stones
      35 objects with 2 stones
      62 objects with 1 stone
      11 objects with 0 stones
    Total: 111 phantoms  (indices 000-110)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Exact distribution from the paper (Section 4.1)
    num_stones_list = [3] * 3 + [2] * 35 + [1] * 62 + [0] * 11
    np.random.shuffle(num_stones_list)

    for i, num_stones in enumerate(num_stones_list):
        phantom = generate_phantom(volume_size, num_stones)
        out_path = os.path.join(output_dir, f'phantom_{i:03d}.npy')
        np.save(out_path, phantom)
        print(f'Saved phantom_{i:03d}.npy  |  stones: {num_stones}')

    print(f'\nDone — {len(num_stones_list)} phantoms saved to "{output_dir}/"')


if __name__ == '__main__':
    generate_all_phantoms()