import numpy as np
from scipy.ndimage import gaussian_filter
import os


def _random_rotation_matrix() -> np.ndarray:
    """Return a uniformly random 3x3 rotation matrix via QR decomposition."""
    A = np.random.randn(3, 3)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _make_internal_texture(volume_size: tuple, mask: np.ndarray,
                           X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                           base_att: float = 0.1) -> np.ndarray:
    
    R = np.sqrt(X**2 + Y**2 + Z**2)   # normalised radius from centre

    # 1. Outer shell: sigmoid-shaped boost near the surface
    shell_thickness = np.random.uniform(0.15, 0.25)
    shell_att_boost = np.random.uniform(0.02, 0.05)
    shell = shell_att_boost / (1.0 + np.exp(-30.0 * (R - (1.0 - shell_thickness))))

    # 2. Central core: lower attenuation (apple core / orange pith)
    core_radius   = np.random.uniform(0.10, 0.20)
    core_att_drop = np.random.uniform(0.03, 0.06)
    core = -core_att_drop / (1.0 + np.exp(40.0 * (R - core_radius)))

    # 3. Large-scale smooth spatial variation in the flesh
    low_freq = gaussian_filter(np.random.randn(*volume_size), sigma=12)
    low_freq /= np.max(np.abs(low_freq)) + 1e-8
    flesh_variation = np.random.uniform(0.008, 0.018) * low_freq

    # 4. Radial fibrous / segment structure
    n_segments  = np.random.randint(6, 12)
    phi_offset  = np.random.uniform(0, 2 * np.pi)
    theta       = np.arctan2(Y, X) + phi_offset
    fibre_strength = np.random.uniform(0.004, 0.010)
    flesh_weight   = np.exp(-((R - 0.5) ** 2) / (2 * 0.2 ** 2))
    fibres = fibre_strength * np.cos(n_segments * theta) * flesh_weight

    # 5. Fine-scale noise
    fine_noise = np.random.normal(0, 0.005, volume_size)

    texture = base_att + shell + core + flesh_variation + fibres + fine_noise
    texture = np.clip(texture, 0, None)
    texture *= mask.astype(float)

    return texture


def _make_playdoh_base(volume_size: tuple, base_att: float = 0.1):

    xs = np.linspace(-1, 1, volume_size[0])
    ys = np.linspace(-1, 1, volume_size[1])
    zs = np.linspace(-1, 1, volume_size[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    # Random rotation so each phantom has a different orientation
    R = _random_rotation_matrix()
    Xr = R[0, 0]*X + R[0, 1]*Y + R[0, 2]*Z
    Yr = R[1, 0]*X + R[1, 1]*Y + R[1, 2]*Z
    Zr = R[2, 0]*X + R[2, 1]*Y + R[2, 2]*Z

    # Deformed ellipsoid boundary
    a, b, c = np.random.uniform(0.6, 0.9, 3)
    ellipsoid = (Xr/a)**2 + (Yr/b)**2 + (Zr/c)**2

    # Smooth low-frequency deformation for organic boundary
    deform = gaussian_filter(np.random.randn(*volume_size), sigma=8)
    deform /= np.max(np.abs(deform)) + 1e-8
    field = ellipsoid + np.random.uniform(0.15, 0.35) * deform

    # Binary mask with smoothed edges
    mask = gaussian_filter((field <= 1.0).astype(float), sigma=2) > 0.5

    # Internal texture using rotated coords for anatomical consistency
    base = _make_internal_texture(volume_size, mask, Xr, Yr, Zr, base_att)

    return base, mask


# Rough ellipsoid stone

def _make_rough_ellipsoid_stone(volume_size: tuple,
                                center: tuple,
                                radii: tuple,
                                stone_att: float = 0.5,
                                roughness_amplitude: float = 0.25,
                                roughness_freq: float = 3.0) -> np.ndarray:
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

    phi = np.random.uniform(0, 2 * np.pi, 3)
    ripple = (
        np.sin(roughness_freq * dx + phi[0]) *
        np.sin(roughness_freq * dy + phi[1]) *
        np.sin(roughness_freq * dz + phi[2])
    )
    ripple /= np.max(np.abs(ripple)) + 1e-8

    stone_mask = r <= (1.0 + roughness_amplitude * ripple)
    return stone_mask.astype(float) * stone_att



def generate_phantom(volume_size: tuple = (128, 128, 128),
                     num_stones: int = 0,
                     base_att: float = 0.1,
                     stone_att: float = 0.5) -> np.ndarray:
    base, base_mask = _make_playdoh_base(volume_size, base_att=base_att)

    half     = np.array(volume_size) / 2
    cube_half = half * 0.5

    stones = np.zeros(volume_size)
    for _ in range(num_stones):
        for _attempt in range(200):
            px = np.random.uniform(-cube_half[0] * 0.75, cube_half[0] * 0.75)
            py = np.random.uniform(-cube_half[1] * 0.75, cube_half[1] * 0.75)
            pz = np.random.uniform(-cube_half[2] * 0.75, cube_half[2] * 0.75)
            ix = int(np.clip(px + half[0], 0, volume_size[0] - 1))
            iy = int(np.clip(py + half[1], 0, volume_size[1] - 1))
            iz = int(np.clip(pz + half[2], 0, volume_size[2] - 1))
            if base_mask[ix, iy, iz]:
                break

        sa, sb, sc = np.random.uniform(5, 15, 3)
        amp  = np.random.uniform(0.20, 0.35)
        freq = np.random.uniform(2, 5)

        stone = _make_rough_ellipsoid_stone(
            volume_size, (px, py, pz), (sa, sb, sc),
            stone_att=stone_att,
            roughness_amplitude=amp,
            roughness_freq=freq,
        )
        stones += stone

    return base + stones



def generate_all_phantoms(output_dir: str = 'phantoms',
                          volume_size: tuple = (128, 128, 128)) -> None:
    os.makedirs(output_dir, exist_ok=True)

    num_stones_list = [3] * 3 + [2] * 35 + [1] * 62 + [0] * 11
    np.random.shuffle(num_stones_list)

    for i, num_stones in enumerate(num_stones_list):
        phantom = generate_phantom(volume_size, num_stones)
        out_path = os.path.join(output_dir, f'phantom_{i:03d}.npy')
        np.save(out_path, phantom)
        print(f'Saved phantom_{i:03d}.npy  |  stones: {num_stones}')

    print(f'\nDone. {len(num_stones_list)} phantoms saved to "{output_dir}/"')


if __name__ == '__main__':
    generate_all_phantoms()