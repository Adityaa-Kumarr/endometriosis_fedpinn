# Full Detail: How We Generate the 3D Uterus Model

This document describes the complete pipeline for generating the parametric 3D uterus mesh used in the endometriosis digital twin simulator.

---

## 1. Overview

- **Method**: Procedural parametric surface (no external mesh files)
- **Stack**: Pure NumPy (no trimesh, no noise library)
- **Output**: Vertex arrays `(x_uterus, y_uterus, z_uterus)` for Plotly `Mesh3d` or scatter
- **Anatomy**: Pear-shaped uterus with fundus (top), body (middle), cervix (bottom)

---

## 2. Coordinate System

We use a spherical-like parameterization:

| Parameter | Range | Anatomical region |
|-----------|-------|-------------------|
| **u** | 0 → π | 0 = fundus (top), π/2 ≈ body (widest), π = cervix (bottom) |
| **v** | 0 → 2π | Azimuth angle around the long axis (like longitude) |

- **u = 0**: Top of uterus (fundus dome)
- **u ≈ 1.57 (π/2)**: Widest part (uterine body)
- **u > 2.2**: Cervix region (tapered, more textured)
- **u = π**: Bottom of cervix

---

## 3. Helper Functions (Noise Pipeline)

### 3.1 Deterministic 3D Hash

```python
def _hash3(x, y, z, seed=0):
    """Deterministic hash for 3D coordinates (pure NumPy, no C extensions)."""
    n = np.sin(x * 12.9898 + y * 78.233 + z * 45.164 + seed) * 43758.5453
    return n - np.floor(n)
```

- Maps integer grid points to pseudo-random values in [0, 1]
- Used as gradient/noise values in Perlin-like interpolation

### 3.2 Smoothstep Interpolation

```python
def _smoothstep(t):
    """Smooth interpolation for organic noise."""
    return t * t * (3.0 - 2.0 * t)
```

- C2-continuous blend: 0 at t=0, 1 at t=1
- Replaces linear interpolation for smoother noise

### 3.3 3D Perlin-like Noise (Single Octave)

```python
def _pnoise3_numpy(x, y, z, scale, seed=42):
    """Pure NumPy 3D noise (replaces noise.pnoise3 to avoid C++ build dependency)."""
    xs, ys, zs = x * scale + seed, y * scale + seed, z * scale + seed
    xi, yi, zi = np.floor(xs).astype(int), np.floor(ys).astype(int), np.floor(zs).astype(int)
    xf, yf, zf = xs - xi, ys - yi, zs - zi
    xf, yf, zf = _smoothstep(xf), _smoothstep(yf), _smoothstep(zf)
    n000 = _hash3(xi, yi, zi, seed)
    n100 = _hash3(xi + 1, yi, zi, seed)
    n010 = _hash3(xi, yi + 1, zi, seed)
    n110 = _hash3(xi + 1, yi + 1, zi, seed)
    n001 = _hash3(xi, yi, zi + 1, seed)
    n101 = _hash3(xi + 1, yi, zi + 1, seed)
    n011 = _hash3(xi, yi + 1, zi + 1, seed)
    n111 = _hash3(xi + 1, yi + 1, zi + 1, seed)
    nx00 = n000 * (1 - xf) + n100 * xf
    nx10 = n010 * (1 - xf) + n110 * xf
    nx01 = n001 * (1 - xf) + n101 * xf
    nx11 = n011 * (1 - xf) + n111 * xf
    nxy0 = nx00 * (1 - yf) + nx10 * yf
    nxy1 = nx01 * (1 - yf) + nx11 * yf
    return nxy0 * (1 - zf) + nxy1 * zf
```

- Trilinear interpolation of 8 corner hash values
- `scale` controls frequency; `seed` for reproducibility

### 3.4 Multi-Octave Organic Noise

```python
def _apply_organic_noise(x, y, z, scale, octaves=4, seed=42):
    """Applies 3D noise displacement to vertices (pure NumPy, no C extensions)."""
    out = np.zeros_like(x)
    for o in range(octaves):
        f = 2 ** o
        out += _pnoise3_numpy(x, y, z, scale * f, seed + o * 17) / f
    return out
```

- 4 octaves: frequencies 1×, 2×, 4×, 8×; amplitudes 1, 1/2, 1/4, 1/8
- Produces natural-looking tissue variation

---

## 4. Uterus Generation: Step-by-Step

### Step 1: Parameters

```python
a, b = 3.2, 2.2           # Pear profile coefficients
z_scale, xy_scale = 1.0   # Optional scaling (from reference .glb)
half_height = 6.5 * z_scale
inflammation_level = 0.0  # 0..1, from AI prediction
resolution = 1.0          # 1.0=full, 0.5=low detail
```

### Step 2: Parametric Grid

```python
res_u = max(48, min(120, int(120 * resolution)))
u = np.linspace(0, np.pi, res_u)
v = np.linspace(0, 2 * np.pi, res_u)
u_grid, v_grid = np.meshgrid(u, v)
```

### Step 3: Base Radius (Pear Shape)

```python
r_base = a + b * np.cos(u_grid)
```

- **u=0 (fundus)**: r = a + b ≈ 5.4
- **u=π/2 (body)**: r = a ≈ 3.2 (widest)
- **u=π (cervix)**: r = a - b ≈ 1.0 (narrowest)

### Step 4: Flatten Fundus Dome

```python
fundus_shape = 1.0 + 0.12 * (1.0 / (1.0 + np.exp(-6.0 * (u_grid - 0.1))))
r_base = r_base * fundus_shape
```

- Sigmoid centered near u=0.1: flatter dome at fundus instead of spherical bulge
- Adds up to 12% radius increase in the transition from fundus to body

### Step 5: Inflammation Swell

```python
swell = 1.0 + (inflammation_level * 0.15)
```

- At max inflammation (1.0): 15% uniform radial swell

### Step 6: Height Scaling (Shorter Cervix)

```python
cervix_blend = 1.0 / (1.0 + np.exp(-10.0 * (u_grid - 2.4)))
z_scale_u = 1.0 - 0.4 * cervix_blend
z_u_base = half_height * np.cos(u_grid) * swell * z_scale_u
```

- Cervix region (u > 2.4): z scaled by 0.6 (40% shorter)
- Smooth sigmoid transition at u ≈ 2.4

### Step 7: Cartesian Coordinates (XY)

```python
x_u_base = r_base * np.sin(u_grid) * np.cos(v_grid) * swell * xy_scale
y_u_base = r_base * np.sin(u_grid) * np.sin(v_grid) * swell * xy_scale
```

### Step 8: Selective Tissue Noise

```python
cervix_region = (u_grid > 2.2).astype(np.float64)
noise_strength = np.where(cervix_region, 0.18, 0.05)
tissue_noise = _apply_organic_noise(x_u_base, y_u_base, z_u_base, scale=0.3, seed=10) * noise_strength
tissue_noise += cervix_region * _apply_organic_noise(x_u_base + 1, y_u_base, z_u_base, scale=0.5, seed=11) * 0.08
```

- Body: low noise (0.05) for smooth surface
- Cervix: higher noise (0.18 + 0.08) for ribbed/textured appearance

### Step 9: Final Vertices (Outward Displacement)

```python
local_norm = np.sqrt(x_u_base**2 + y_u_base**2) + 1e-6
x_uterus = x_u_base + (x_u_base / local_norm) * tissue_noise
y_uterus = y_u_base + (y_u_base / local_norm) * tissue_noise
z_uterus = z_u_base + (z_u_base / (np.abs(z_u_base) + 1e-6)) * tissue_noise
```

- XY: radial displacement in the horizontal plane
- Z: axial displacement along the long axis
- Avoids full 3D normal for more cylindrical, anatomically plausible displacement

---

## 5. Complete Copy-Paste Code Block

```python
import numpy as np

# ============ NOISE HELPERS ============

def _hash3(x, y, z, seed=0):
    n = np.sin(x * 12.9898 + y * 78.233 + z * 45.164 + seed) * 43758.5453
    return n - np.floor(n)

def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)

def _pnoise3_numpy(x, y, z, scale, seed=42):
    xs, ys, zs = x * scale + seed, y * scale + seed, z * scale + seed
    xi, yi, zi = np.floor(xs).astype(int), np.floor(ys).astype(int), np.floor(zs).astype(int)
    xf, yf, zf = _smoothstep(xs - xi), _smoothstep(ys - yi), _smoothstep(zs - zi)
    n000 = _hash3(xi, yi, zi, seed)
    n100 = _hash3(xi + 1, yi, zi, seed)
    n010 = _hash3(xi, yi + 1, zi, seed)
    n110 = _hash3(xi + 1, yi + 1, zi, seed)
    n001 = _hash3(xi, yi, zi + 1, seed)
    n101 = _hash3(xi + 1, yi, zi + 1, seed)
    n011 = _hash3(xi, yi + 1, zi + 1, seed)
    n111 = _hash3(xi + 1, yi + 1, zi + 1, seed)
    nx00 = n000 * (1 - xf) + n100 * xf
    nx10 = n010 * (1 - xf) + n110 * xf
    nx01 = n001 * (1 - xf) + n101 * xf
    nx11 = n011 * (1 - xf) + n111 * xf
    nxy0 = nx00 * (1 - yf) + nx10 * yf
    nxy1 = nx01 * (1 - yf) + nx11 * yf
    return nxy0 * (1 - zf) + nxy1 * zf

def _apply_organic_noise(x, y, z, scale, octaves=4, seed=42):
    out = np.zeros_like(x)
    for o in range(octaves):
        f = 2 ** o
        out += _pnoise3_numpy(x, y, z, scale * f, seed + o * 17) / f
    return out

# ============ UTERUS 3D GENERATION ============

def generate_uterus_mesh(a=3.2, b=2.2, z_scale=1.0, xy_scale=1.0,
                         inflammation_level=0.0, resolution=1.0):
    """
    Returns (x_uterus, y_uterus, z_uterus) as 2D arrays (res_u x res_u).
    """
    half_height = 6.5 * z_scale
    swell = 1.0 + (inflammation_level * 0.15)

    res_u = max(48, min(120, int(120 * resolution)))
    u = np.linspace(0, np.pi, res_u)
    v = np.linspace(0, 2 * np.pi, res_u)
    u_grid, v_grid = np.meshgrid(u, v)

    # 1. Base pear radius
    r_base = a + b * np.cos(u_grid)

    # 2. Flatten fundus dome
    fundus_shape = 1.0 + 0.12 * (1.0 / (1.0 + np.exp(-6.0 * (u_grid - 0.1))))
    r_base = r_base * fundus_shape

    # 3. Height scaling (shorter cervix)
    cervix_blend = 1.0 / (1.0 + np.exp(-10.0 * (u_grid - 2.4)))
    z_scale_u = 1.0 - 0.4 * cervix_blend
    z_u_base = half_height * np.cos(u_grid) * swell * z_scale_u

    # 4. Cartesian coords
    x_u_base = r_base * np.sin(u_grid) * np.cos(v_grid) * swell * xy_scale
    y_u_base = r_base * np.sin(u_grid) * np.sin(v_grid) * swell * xy_scale

    # 5. Selective noise
    cervix_region = (u_grid > 2.2).astype(np.float64)
    noise_strength = np.where(cervix_region, 0.18, 0.05)
    tissue_noise = _apply_organic_noise(x_u_base, y_u_base, z_u_base, scale=0.3, seed=10) * noise_strength
    tissue_noise += cervix_region * _apply_organic_noise(x_u_base + 1, y_u_base, z_u_base, scale=0.5, seed=11) * 0.08

    # 6. Final vertices
    local_norm = np.sqrt(x_u_base**2 + y_u_base**2) + 1e-6
    x_uterus = x_u_base + (x_u_base / local_norm) * tissue_noise
    y_uterus = y_u_base + (y_u_base / local_norm) * tissue_noise
    z_uterus = z_u_base + (z_u_base / (np.abs(z_u_base) + 1e-6)) * tissue_noise

    return x_uterus, y_uterus, z_uterus

# Example usage:
# x, y, z = generate_uterus_mesh(inflammation_level=0.3, resolution=0.5)
# For Plotly Mesh3d: i, j, k are triangle indices from meshgrid (see simulator for triangulation)
```

---

## 6. Parameter Reference

| Parameter | Default | Effect |
|-----------|---------|--------|
| `a` | 3.2 | Base radius at body (widest) |
| `b` | 2.2 | Pear taper (fundus +b, cervix -b) |
| `z_scale` | 1.0 | Vertical scale |
| `xy_scale` | 1.0 | Horizontal scale |
| `inflammation_level` | 0.0 | 0–1, adds 15% swell at 1.0 |
| `resolution` | 1.0 | Mesh density (0.4–1.0) |

---

## 7. File Location

**Primary implementation**: `digital_twin/uterus_mesh_fixed.py`  
- `generate_uterus_mesh()` — anatomically enhanced parametric mesh with triangulation  
- `build_plotly_traces()` — converts output to Plotly Mesh3d traces  

**Integration**: `digital_twin/simulator.py`  
- `UterusDigitalTwin.generate_3d_scatter_data()` calls `uterus_mesh_fixed.generate_uterus_mesh()` for the uterus
