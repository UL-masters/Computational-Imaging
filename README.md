# Tomographic Reconstruction for Foreign Object Detection
**Group 17 (B) - Computational Imaging and Tomography, Leiden University**  
Alexa van Thiel (s3617238)
Andreea Bîrsan (s4760840)

Implementation and experimental study of FBP, SIRT and SIRT-TV reconstruction
for X-ray based foreign object detection, based on:

> Zeegers et al., *A tomographic workflow to enable deep learning for X-ray based
> foreign object detection*, Expert Systems With Applications, 2022.

---

## Installation

ASTRA Toolbox must be installed via conda 

```bash
conda create -n tomography python=3.11
conda activate tomography
conda install -c astra-toolbox astra-toolbox
pip install numpy scipy matplotlib scikit-image pandas
```

## How to run

### 1. Generate phantoms
```bash
python phantoms.py
```
Generates 111 phantoms into `phantoms/`

### 2. Visualize a phantom
```bash
python visualize_phantom.py --phantom phantoms/phantom_009.npy
```
Change `phantom_009.npy` to any phantom. Saves to `results/`.

### 3. Run experiments

Each experiment saves a CSV to `results/`:

```bash
# Experiment 1: angle sweep (FBP vs SIRT, noiseless)
python experiment_angle_sweep.py

# Experiment 2: joint angle x noise grid (FBP, SIRT, SIRT-TV)
python experiment_grid.py

# Experiment 3: SIRT iteration count sweep
python experiment_SIRT_iterations.py
```

### 4. Generate figures

```bash
# Figures for Experiment 2 (heatmaps, scatter, correlation table)
python analyse_results.py

# Figures for Experiments 1 and 3 (line plots with std bands)
python analyse_angles_iterations.py
```

Figures are saved to `figures/`.

