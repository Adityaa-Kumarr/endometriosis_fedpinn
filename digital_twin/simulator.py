import numpy as np

from .uterus_mesh_fixed import generate_uterus_mesh as _generate_uterus_mesh_fixed


def _hash3(x, y, z, seed=0):
    """Deterministic hash for 3D coordinates (pure NumPy, no C extensions)."""
    n = np.sin(x * 12.9898 + y * 78.233 + z * 45.164 + seed) * 43758.5453
    return n - np.floor(n)


def _smoothstep(t):
    """Smooth interpolation for organic noise."""
    return t * t * (3.0 - 2.0 * t)


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


class UterusDigitalTwin:
    """
    Simulates the physiological state of a uterus for a Digital Twin dashboard.
    Outputs metrics and visual parameters representing Endometriosis progression.
    
    Key Improvements (v2):
    - Deterministic: uses patient_seed for reproducible lesion placement
    - Temporal: can generate disease progression trajectories
    - Endometrioma: properly renders cyst geometry on ovary
    """
    def __init__(self, patient_base_state=None):
        self.state = patient_base_state or {
            'inflammation_level': 0.0,
            'lesion_count': 0,
            'adhesions_present': False,
            'endometrioma_size_cm': 0.0
        }
        
    def update_from_model_prediction(self, probability, stage, future_risk=None, seed=None):
        """
        Updates the internal digital twin state based on AI predictions.
        stage classes: 0=None, 1=Minimal, 2=Mild, 3=Moderate, 4=Severe
        seed: optional int for reproducible lesion/adhesion counts
        """
        if seed is not None:
            np.random.seed(seed)
        self.state['inflammation_level'] = min(1.0, probability * 1.5)
        
        # Base Future Progression
        if future_risk is not None:
            # future_risk is now explicitly passed as a scalar representing the far-future (e.g. 5-yr) risk
            self.state['future_lesion_multiplier'] = min(5.0, future_risk * 2.5) 
        else:
            self.state['future_lesion_multiplier'] = 0.0
        
        # Lesions and size scale with stage
        if stage == 0:
            self.state['lesion_count'] = 0
            self.state['adhesions_present'] = False
            self.state['endometrioma_size_cm'] = 0.0
        elif stage == 1:
            self.state['lesion_count'] = np.random.randint(2, 8)
            self.state['adhesions_present'] = False
            self.state['endometrioma_size_cm'] = 0.0
        elif stage == 2:
            self.state['lesion_count'] = np.random.randint(12, 25)
            self.state['adhesions_present'] = False
            self.state['endometrioma_size_cm'] = np.random.uniform(0.1, 1.5)
        elif stage == 3:
            self.state['lesion_count'] = np.random.randint(30, 60)
            self.state['adhesions_present'] = True
            self.state['endometrioma_size_cm'] = np.random.uniform(1.5, 4.0)
        elif stage == 4:
            self.state['lesion_count'] = np.random.randint(60, 120)
            self.state['adhesions_present'] = True
            self.state['endometrioma_size_cm'] = np.random.uniform(4.0, 10.0)
            
        return self.state

    def _apply_organic_noise(self, x, y, z, scale, octaves=4, seed=42):
        """Applies 3D noise displacement to vertices (pure NumPy, no C extensions)."""
        out = np.zeros_like(x)
        for o in range(octaves):
            f = 2 ** o
            out += _pnoise3_numpy(x, y, z, scale * f, seed + o * 17) / f
        return out

    def generate_3d_scatter_data(self, reference_shape=None, resolution=1.0, patient_seed=None):
        """
        Generates 3D mesh points representing the AI Digital Twin.
        reference_shape: optional dict from reference_shape.get_reference_shape_params()
        resolution: 1.0=full (120), 0.6=medium (72), 0.4=low (48) for faster rendering
        Args:
            patient_seed: int or None. If provided, all random operations use this
                         seed for deterministic, reproducible lesion placement.
        """
        rng = np.random.RandomState(patient_seed if patient_seed is not None else 42)
        
        if reference_shape and isinstance(reference_shape, dict):
            a, b = reference_shape.get("a", 3.2), reference_shape.get("b", 2.2)
            z_scale = reference_shape.get("z_scale", 1.0)
            xy_scale = reference_shape.get("xy_scale", 1.0)
        else:
            a, b = 3.2, 2.2
            z_scale = 1.0
            xy_scale = 1.0
        half_height = 6.5 * z_scale
        uterus_mesh_data = _generate_uterus_mesh_fixed(
            a=a, b=b, z_scale=z_scale, xy_scale=xy_scale,
            inflammation_level=self.state['inflammation_level'],
            resolution=resolution,
            include_cavity=True,
            include_cervical_canal=False,
            include_fallopian_junctions=False,
            seed=42,
        )
        o = uterus_mesh_data['outer']
        x_uterus, y_uterus, z_uterus = o['x'], o['y'], o['z']
        uterus_i, uterus_j, uterus_k = o['i'], o['j'], o['k']
        # Endometrial cavity
        _cav = uterus_mesh_data.get('cavity')
        cavity_mesh = (
            _cav['x'], _cav['y'], _cav['z'], _cav['i'], _cav['j'], _cav['k']
        ) if _cav else None
        
        res_o = max(36, min(60, int(60 * resolution)))
        u_ov = np.linspace(0, np.pi, res_o)
        v_ov = np.linspace(0, 2*np.pi, res_o)
        uo_grid, vo_grid = np.meshgrid(u_ov, v_ov)
        
        # Ovaries: anatomically almond-shaped (~1/3 width of uterus)
        # Assuming uterus width is ~5cm, ovary width is ~1.5cm radius
        ov_scale = 0.85
        x_o_base = ov_scale * np.sin(uo_grid) * np.cos(vo_grid)
        y_o_base = 0.7 * ov_scale * np.sin(uo_grid) * np.sin(vo_grid)
        z_o_base = 1.3 * ov_scale * np.cos(uo_grid) # Almond elongated along Z
        
        # Ovary follicle/texture noise
        ovary_noise_l = self._apply_organic_noise(x_o_base, y_o_base, z_o_base, scale=0.8, seed=20) * 0.25
        ovary_noise_r = self._apply_organic_noise(x_o_base, y_o_base, z_o_base, scale=0.8, seed=30) * 0.25
        
        o_norm = np.sqrt(x_o_base**2 + y_o_base**2 + z_o_base**2) + 1e-6
        xl_ov = x_o_base + (x_o_base/o_norm) * ovary_noise_l
        yl_ov = y_o_base + (y_o_base/o_norm) * ovary_noise_l
        zl_ov = z_o_base + (z_o_base/o_norm) * ovary_noise_l
        
        xr_ov = x_o_base + (x_o_base/o_norm) * ovary_noise_r
        yr_ov = y_o_base + (y_o_base/o_norm) * ovary_noise_r
        zr_ov = z_o_base + (z_o_base/o_norm) * ovary_noise_r
        
        # Add Endometrioma mass to ovaries
        endo_rad = self.state.get('endometrioma_size_cm', 0.0) / 2.0
        if endo_rad > 0:
            rng_seed = np.random.RandomState(42)
            cyst_x = endo_rad * np.sin(uo_grid) * np.cos(vo_grid)
            cyst_y = endo_rad * np.sin(uo_grid) * np.sin(vo_grid)
            cyst_z = endo_rad * np.cos(uo_grid)
            cyst_noise = self._apply_organic_noise(cyst_x, cyst_y, cyst_z, scale=1.5, seed=50) * 0.2
            c_norm = np.sqrt(cyst_x**2 + cyst_y**2 + cyst_z**2) + 1e-6
            cyst_x += (cyst_x/c_norm) * cyst_noise
            cyst_y += (cyst_y/c_norm) * cyst_noise
            cyst_z += (cyst_z/c_norm) * cyst_noise
            
            # Offset cyst to edge of left ovary
            cyst_x -= (ov_scale * 0.8)
            cyst_z += (ov_scale * 0.5)
            
            # Merge cyst into left ovary
            for i in range(res_o):
                for j in range(res_o):
                    dx = xl_ov[i,j] - (-ov_scale * 0.8)
                    dy = yl_ov[i,j] - 0.0
                    dz = zl_ov[i,j] - (ov_scale * 0.5)
                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                    if dist < endo_rad * 1.3:
                        blend = max(0, 1.0 - dist / (endo_rad * 1.3))
                        xl_ov[i,j] += dx * blend * 0.3
                        yl_ov[i,j] += dy * blend * 0.3
                        zl_ov[i,j] += dz * blend * 0.3
        
        # Height of uterus is 8.0 * z_scale. z=0.88 is cornua.
        height_u = 8.0 * z_scale
        R_max = 2.5 * xy_scale
        R_min = 0.6 * xy_scale
        z_cor = 0.88
        r_cor = R_max * (z_cor**1.5) + R_min
        
        x_cornu_r = r_cor * 0.8  # flatten factor
        x_cornu_l = -x_cornu_r
        z_cornu = height_u * z_cor
        y_cornu = 0.0

        # Ovary positions: Lateral + slightly posterior
        ov_center_x = x_cornu_r + 2.5
        ov_center_y = -1.2 # Posterior
        ov_center_z = z_cornu - 1.5

        left_ovary = (xl_ov - ov_center_x, yl_ov + ov_center_y, zl_ov + ov_center_z)
        right_ovary = (xr_ov + ov_center_x, yr_ov + ov_center_y, zr_ov + ov_center_z)
        
        nt, ntheta = max(40, int(80 * resolution)), max(18, int(30 * resolution))
        t = np.linspace(0, 1, nt)
        theta_tube = np.linspace(0, 2*np.pi, ntheta)
        t_grid, theta_grid = np.meshgrid(t, theta_tube)
        
        # Fallopian tube mathematically: x = r*cos(t), y=r*sin(t), z=height+0.5*sin(t) from prompt
        # We parameterize t from 0 to 1 moving laterally from uterus to ovary
        # Left tube
        arch_l = np.sin(np.pi * t_grid)
        curve_x_l = x_cornu_l + (-ov_center_x - x_cornu_l) * t_grid
        curve_y_l = y_cornu + (ov_center_y - y_cornu) * t_grid
        curve_z_l = z_cornu + 0.8 * arch_l * (1.0 - t_grid*0.5) # 20-30 degree upward tilt arch
        
        fimbria_l = np.where(t_grid > 0.85, 0.4 * np.sin(8 * theta_grid) * np.sin(np.pi * (t_grid - 0.85) / 0.15), 0.0)
        flare = np.where(t_grid > 0.85, 1.0 + (t_grid - 0.85) * 4.0, 1.0)
        radius_tube_l = (0.22 - 0.08 * t_grid) * flare + fimbria_l
        radius_tube_l = np.maximum(radius_tube_l, 0.05)
        
        # Add slight wobble
        x_tube_l = curve_x_l + radius_tube_l * np.cos(theta_grid)
        y_tube_l = curve_y_l + radius_tube_l * np.sin(theta_grid)
        z_tube_l = curve_z_l + radius_tube_l * np.sin(theta_grid) * 0.5
        tube_noise_l = self._apply_organic_noise(x_tube_l, y_tube_l, z_tube_l, scale=1.5, seed=60) * 0.05
        left_tube = (x_tube_l + tube_noise_l, y_tube_l + tube_noise_l, z_tube_l + tube_noise_l)
        
        # Right tube
        arch_r = np.sin(np.pi * t_grid)
        curve_x_r = x_cornu_r + (ov_center_x - x_cornu_r) * t_grid
        curve_y_r = y_cornu + (ov_center_y - y_cornu) * t_grid
        curve_z_r = z_cornu + 0.8 * arch_r * (1.0 - t_grid*0.5)
        
        fimbria_r = np.where(t_grid > 0.85, 0.4 * np.sin(8 * theta_grid + 0.5) * np.sin(np.pi * (t_grid - 0.85) / 0.15), 0.0)
        radius_tube_r = (0.22 - 0.08 * t_grid) * flare + fimbria_r
        radius_tube_r = np.maximum(radius_tube_r, 0.05)
        
        x_tube_r = curve_x_r + radius_tube_r * np.cos(theta_grid)
        y_tube_r = curve_y_r + radius_tube_r * np.sin(theta_grid)
        z_tube_r = curve_z_r + radius_tube_r * np.sin(theta_grid) * 0.5
        tube_noise_r = self._apply_organic_noise(x_tube_r, y_tube_r, z_tube_r, scale=1.5, seed=70) * 0.05
        right_tube = (x_tube_r + tube_noise_r, y_tube_r + tube_noise_r, z_tube_r + tube_noise_r)
        
        # Lesion placement mathematically fixed for the new boundaries
        lesion_x, lesion_y, lesion_z = [], [], []
        lesion_colors, lesion_sizes = [], []
        
        if endo_rad > 0:
            for _ in range(int(endo_rad * 30)):
                th = np.random.uniform(0, np.pi)
                ph = np.random.uniform(0, 2*np.pi)
                rad = endo_rad * np.random.uniform(0.8, 1.0)
                lesion_x.append(rad * np.sin(th) * np.cos(ph) - ov_center_x)
                lesion_y.append(rad * np.sin(th) * np.sin(ph) + ov_center_y)
                lesion_z.append(rad * np.cos(th) + ov_center_z)
                lesion_colors.append(0.3)
                lesion_sizes.append(np.random.uniform(10, 20))
                
        n_lesions = self.state.get('lesion_count', 0)
        for _ in range(n_lesions):
            target = rng.choice(['ut_pouch', 'ut_front', 'left_ovary', 'right_ovary', 'tubes'], p=[0.4, 0.2, 0.15, 0.15, 0.1])
            if target == 'ut_pouch':
                z_n = np.random.uniform(0.2, 0.8)
                ph = np.random.uniform(np.pi*0.75, np.pi*1.25)
                rad = (R_max * (z_n**1.5) + R_min) * 1.05
                lesion_x.append(rad * np.cos(ph) * 0.8)
                lesion_y.append(rad * np.sin(ph))
                lesion_z.append(height_u * z_n)
            elif target == 'ut_front':
                z_n = np.random.uniform(0.4, 0.9)
                ph = np.random.uniform(-np.pi*0.25, np.pi*0.25)
                rad = (R_max * (z_n**1.5) + R_min) * 1.05
                lesion_x.append(rad * np.cos(ph) * 0.8)
                lesion_y.append(rad * np.sin(ph))
                lesion_z.append(height_u * z_n)
            elif target == 'left_ovary':
                th = np.random.uniform(0, np.pi)
                ph = np.random.uniform(0, 2*np.pi)
                lesion_x.append(1.0 * np.sin(th) * np.cos(ph) - ov_center_x)
                lesion_y.append(1.0 * np.sin(th) * np.sin(ph) + ov_center_y)
                lesion_z.append(1.0 * np.cos(th) + ov_center_z)
            elif target == 'right_ovary':
                th = np.random.uniform(0, np.pi)
                ph = np.random.uniform(0, 2*np.pi)
                lesion_x.append(1.0 * np.sin(th) * np.cos(ph) + ov_center_x)
                lesion_y.append(1.0 * np.sin(th) * np.sin(ph) + ov_center_y)
                lesion_z.append(1.0 * np.cos(th) + ov_center_z)
            else:
                side = np.random.choice([-1, 1])
                t_val = np.random.uniform(0.1, 0.9)
                th_val = np.random.uniform(0, 2*np.pi)
                cx = x_cornu_l * (1 - t_val) + (-ov_center_x) * t_val if side < 0 else x_cornu_r * (1 - t_val) + ov_center_x * t_val
                cy = y_cornu + (ov_center_y - y_cornu) * t_val
                cz = z_cornu + 0.8 * np.sin(np.pi * t_val) * (1.0 - t_val*0.5)
                r_val = 0.25 - 0.1 * t_val
                lesion_x.append(cx + r_val * np.cos(th_val) * 1.2)
                lesion_y.append(cy + r_val * np.sin(th_val) * 1.2)
                lesion_z.append(cz)
                
            lesion_colors.append(np.random.uniform(0.6, 1.0))
            lesion_sizes.append(np.random.uniform(6, 12))
            
        # ---------------------------------------------
        # FUTURE PREDICTED LESIONS (Yellow/Translucent)
        # ---------------------------------------------
        f_lesion_x, f_lesion_y, f_lesion_z = [], [], []
        f_lesion_colors = []
        future_lesion_count = int(self.state.get('lesion_count', 0) * self.state.get('future_lesion_multiplier', 0.0))
        
        for _ in range(future_lesion_count):
            # Cluster centers (mostly spreading further out into the pelvis)
            target = np.random.choice(['ut_pouch', 'ut_front', 'left_ovary', 'right_ovary', 'tubes', 'pelvic_wall'], p=[0.3, 0.2, 0.1, 0.1, 0.1, 0.2])
            if target == 'pelvic_wall':
                th = np.random.uniform(np.pi/2, np.pi)
                ph = np.random.uniform(0, 2*np.pi)
                rad = 8.5 # Expanding beyond uterus
                f_lesion_x.append(rad * np.sin(th) * np.cos(ph))
                f_lesion_y.append(rad * np.sin(th) * np.sin(ph))
                f_lesion_z.append(3.0 * np.cos(th))
            elif target == 'ut_pouch':
                th = np.random.uniform(np.pi/2, np.pi)
                ph = np.random.uniform(-np.pi/4, np.pi/4) + np.pi
                rad = (a + b * np.cos(th)) * 1.15 * xy_scale
                f_lesion_x.append(rad * np.sin(th) * np.cos(ph))
                f_lesion_y.append(rad * np.sin(th) * np.sin(ph))
                f_lesion_z.append(half_height * np.cos(th))
            elif target == 'ut_front':
                th = np.random.uniform(0, np.pi/2)
                ph = np.random.uniform(-np.pi/4, np.pi/4)
                rad = (a + b * np.cos(th)) * 1.15 * xy_scale
                f_lesion_x.append(rad * np.sin(th) * np.cos(ph))
                f_lesion_y.append(rad * np.sin(th) * np.sin(ph))
                f_lesion_z.append(half_height * np.cos(th))
            elif target == 'left_ovary':
                th = np.random.uniform(0, np.pi)
                ph = np.random.uniform(0, 2*np.pi)
                f_lesion_x.append(1.2 * np.sin(th) * np.cos(ph) - ov_center_x)
                f_lesion_y.append(1.2 * np.sin(th) * np.sin(ph))
                f_lesion_z.append(1.2 * np.cos(th) + ov_center_z)
            elif target == 'right_ovary':
                th = np.random.uniform(0, np.pi)
                ph = np.random.uniform(0, 2*np.pi)
                f_lesion_x.append(1.2 * np.sin(th) * np.cos(ph) + ov_center_x)
                f_lesion_y.append(1.2 * np.sin(th) * np.sin(ph))
                f_lesion_z.append(1.2 * np.cos(th) + ov_center_z)
            else: # Tubes
                side = np.random.choice([-1, 1])
                t_val = np.random.uniform(0.1, 0.9)
                th_val = np.random.uniform(0, 2*np.pi)
                cx = x_cornu_l * (1 - t_val) + (-ov_center_x) * t_val if side < 0 else x_cornu_r * (1 - t_val) + ov_center_x * t_val
                cy = 0.4 * np.sin(np.pi * t_val)
                cz = z_cornu + (ov_center_z - z_cornu) * t_val
                r_val = 0.25 - 0.1 * t_val
                f_lesion_x.append(cx + r_val * np.cos(th_val) * 1.8)
                f_lesion_y.append(cy + r_val * np.sin(th_val) * 1.8)
                f_lesion_z.append(cz)
                
            f_lesion_colors.append(np.random.uniform(0.1, 0.4)) # Different color map range for future
            
        # Adhesions (web-like strands: uterus to ovaries)
        adhesion_lines = []
        if self.state['adhesions_present']:
            num_strands = min(20, max(8, self.state['lesion_count'] // 3))  # Scale with severity, cap for perf
            for _ in range(num_strands):
                th1 = np.random.uniform(np.pi/2, np.pi)
                ph1 = np.random.uniform(0, 2*np.pi)
                r1 = (a + b * np.cos(th1)) * 1.02 * xy_scale
                pt1 = np.array([r1 * np.sin(th1) * np.cos(ph1), r1 * np.sin(th1) * np.sin(ph1), half_height * np.cos(th1)])
                side = np.random.choice([-1, 1])
                th2 = np.random.uniform(0, np.pi)
                ph2 = np.random.uniform(0, 2*np.pi)
                pt2 = np.array([1.1 * np.sin(th2) * np.cos(ph2) + (side * ov_center_x), 1.1 * np.sin(th2) * np.sin(ph2), 1.1 * np.cos(th2) + ov_center_z])
                t_line = np.linspace(0, 1, 6)  # Fewer segments for perf
                for i in range(len(t_line) - 1):
                    pA = pt1 * (1 - t_line[i]) + pt2 * t_line[i]
                    pB = pt1 * (1 - t_line[i+1]) + pt2 * t_line[i+1]
                    pA = pA.copy()
                    pB = pB.copy()
                    pA[2] -= np.sin(t_line[i] * np.pi) * 1.2
                    pB[2] -= np.sin(t_line[i+1] * np.pi) * 1.2
                    adhesion_lines.append((pA, pB))
                
        # future_lesions: optional sizes (same length as coords) for consistency
        f_sizes = [8.0] * len(f_lesion_x) if f_lesion_x else []
        # Part annotations (reference-guided): label positions from geometry so system knows which part is what
        def _centroid(xx, yy, zz, offset=(0, 0, 0.9)):
            return (float(np.mean(xx)) + offset[0], float(np.mean(yy)) + offset[1], float(np.mean(zz)) + offset[2])
        part_labels = [
            ('Uterus', *_centroid(np.asarray(x_uterus), np.asarray(y_uterus), np.asarray(z_uterus))),
            ('Left Ovary', *_centroid(left_ovary[0], left_ovary[1], left_ovary[2], (0, 0, 0.6))),
            ('Right Ovary', *_centroid(right_ovary[0], right_ovary[1], right_ovary[2], (0, 0, 0.6))),
            ('Left Fallopian Tube', *_centroid(left_tube[0], left_tube[1], left_tube[2], (-0.3, 0, 0.4))),
            ('Right Fallopian Tube', *_centroid(right_tube[0], right_tube[1], right_tube[2], (0.3, 0, 0.4))),
        ]
        return {
            'uterus': (x_uterus, y_uterus, z_uterus, uterus_i, uterus_j, uterus_k),
            'uterus_cavity': cavity_mesh,
            'left_ovary': left_ovary,
            'right_ovary': right_ovary,
            'left_tube': left_tube,
            'right_tube': right_tube,
            'lesions': (lesion_x, lesion_y, lesion_z, lesion_colors),
            'lesion_sizes': lesion_sizes,
            'future_lesions': (f_lesion_x, f_lesion_y, f_lesion_z, f_lesion_colors),
            'future_lesion_sizes': f_sizes,
            'adhesions': adhesion_lines,
            'part_labels': part_labels,
        }

    def generate_temporal_progression(self, base_probability, base_stage, future_risk_5yr, num_steps=4):
        """
        Generates a sequence of digital twin states representing disease progression
        over time (current, 1yr, 3yr, 5yr).
        
        Uses a simple exponential growth model:
        P(t) = P(0) + (1 - P(0)) * (1 - exp(-lambda * t))
        where lambda is derived from the 5-year risk prediction.
        
        Returns:
            list of dicts, each containing state parameters at a time point.
        """
        time_points = [0, 1, 3, 5]  # years
        
        # Derive growth rate from 5-year risk
        if future_risk_5yr > 0 and future_risk_5yr < 1:
            # lambda such that P(5) = future_risk_5yr
            lam = -np.log(1 - future_risk_5yr) / 5.0
        else:
            lam = 0.1
        
        progression = []
        for t in time_points:
            # Probability increases over time
            prob_t = base_probability + (1 - base_probability) * (1 - np.exp(-lam * t))
            prob_t = min(0.99, prob_t)
            
            # Stage may increase at higher probabilities
            if prob_t > 0.85:
                stage_t = min(4, base_stage + 2)
            elif prob_t > 0.65:
                stage_t = min(4, base_stage + 1)
            else:
                stage_t = base_stage
            
            progression.append({
                'time_years': t,
                'probability': prob_t,
                'stage': stage_t,
                'inflammation': min(1.0, prob_t * 1.5),
                'lesion_count': int(prob_t * 100 * (stage_t + 1) / 5),
            })
        
        return progression

if __name__ == "__main__":
    twin = UterusDigitalTwin()
    twin.update_from_model_prediction(0.8, 3)
    data = twin.generate_3d_scatter_data(patient_seed=12345)
    print("Generated 3D nodes (deterministic).")
    
    prog = twin.generate_temporal_progression(0.7, 2, 0.85)
    for step in prog:
        print(f"  Year {step['time_years']}: P={step['probability']:.2f}, Stage={step['stage']}")
