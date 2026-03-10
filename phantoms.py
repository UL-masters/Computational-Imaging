import numpy as np
from scipy.ndimage import gaussian_filter
import os

def generate_phantom(volume_size=(128, 128, 128), num_stones=0, base_att=0.1, stone_att=0.5):
    
    # coordinate grid centered at origin
    half_size = np.array(volume_size) / 2
    x, y, z = np.ogrid[-half_size[0]:half_size[0], -half_size[1]:half_size[1], -half_size[2]:half_size[2]]
    
    # deformed ellipsoid with random scaling
    scale_factors = np.random.uniform(0.8, 1.2, 3)
    a, b, c = half_size * scale_factors
    base_mask = (x**2 / a**2 + y**2 / b**2 + z**2 / c**2 <= 1)
    
    # Gaussian noise and smoothing
    base = base_mask.astype(float) * base_att
    noise = np.random.normal(0, 0.05, volume_size)
    base += noise * base_mask
    base = gaussian_filter(base, sigma=2)
    
    base = (base > 0.05).astype(float) * base_att
    
    # add stones
    stones = np.zeros(volume_size)
    for _ in range(num_stones):
        # random position inside base
        while True:
            px = np.random.uniform(-a * 0.9, a * 0.9)
            py = np.random.uniform(-b * 0.9, b * 0.9)
            pz = np.random.uniform(-c * 0.9, c * 0.9)
            if (px**2 / a**2 + py**2 / b**2 + pz**2 / c**2 < 0.8):
                break
        
        # stone - small ellipsoid with random size (5-15 voxels radius)
        sa, sb, sc = np.random.uniform(5, 15, 3)
        stone_mask = ((x - px)**2 / sa**2 + (y - py)**2 / sb**2 + (z - pz)**2 / sc**2 <= 1)
        stones += stone_mask.astype(float) * stone_att
    
    phantom = base + stones
    return phantom

def generate_all_phantoms(output_dir='phantoms', volume_size=(128, 128, 128)):
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 3 with 3 stones, 35 with 2, 62 with 1, 11 with 0
    num_stones_list = [3] * 3 + [2] * 35 + [1] * 62 + [0] * 11
    np.random.shuffle(num_stones_list)  # for randomness
    
    for i, num_stones in enumerate(num_stones_list):
        phantom = generate_phantom(volume_size, num_stones)
        np.save(os.path.join(output_dir, f'phantom_{i:03d}.npy'), phantom)
        print(f'Generated phantom {i:03d} with {num_stones} stones')

if __name__ == '__main__':
    generate_all_phantoms()