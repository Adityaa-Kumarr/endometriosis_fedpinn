import numpy as np


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
    """
    def __init__(self, patient_base_state=None):
        self.state = patient_base_state or {
            'inflammation_level': 0.0,
            'lesion_count': 0,
            'adhesions_present': False,
            'endometrioma_size_cm': 0.0
        }
        
    def update_from_model_prediction(self, probability, stage, future_risk=None):
        """
        Updates the internal digital twin state based on AI predictions.
        stage classes: 0=None, 1=Minimal, 2=Mild, 3=Moderate, 4=Severe
        """
        # Inflammation scales with probability
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

    def generate_3d_scatter_data(self):
        """
        Generates hyper-realistic 3D mesh points representing the AI Digital Twin.
        Uses high-resolution grids and organic noise displacement.
        """
        # High resolution grid for Uterus
        res_u = 120
        u = np.linspace(0, np.pi, res_u)
        v = np.linspace(0, 2 * np.pi, res_u)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Base Pear-shaped uterus body
        r_base = 3.8 + 1.6 * np.cos(u_grid)
        
        # Apply subtle organic swelling based on inflammation
        swell = 1.0 + (self.state['inflammation_level'] * 0.15)
        
        x_u_base = r_base * np.sin(u_grid) * np.cos(v_grid) * swell
        y_u_base = r_base * np.sin(u_grid) * np.sin(v_grid) * swell
        z_u_base = 5.5 * np.cos(u_grid) * swell
        
        # Add organ tissue noise (bumps and imperfections)
        tissue_noise = self._apply_organic_noise(x_u_base, y_u_base, z_u_base, scale=0.3, seed=10) * 0.4
        
        # Calculate normals outward for displacement
        norm_len = np.sqrt(x_u_base**2 + y_u_base**2 + z_u_base**2) + 1e-6
        x_uterus = x_u_base + (x_u_base/norm_len) * tissue_noise
        y_uterus = y_u_base + (y_u_base/norm_len) * tissue_noise
        z_uterus = z_u_base + (z_u_base/norm_len) * tissue_noise
        
        # High resolution Ovaries (almond shaped with surface bumps)
        res_o = 60
        u_ov = np.linspace(0, np.pi, res_o)
        v_ov = np.linspace(0, 2*np.pi, res_o)
        uo_grid, vo_grid = np.meshgrid(u_ov, v_ov)
        
        x_o_base = 1.8 * np.sin(uo_grid) * np.cos(vo_grid)
        y_o_base = 1.2 * np.sin(uo_grid) * np.sin(vo_grid) # Flattened slightly like almond
        z_o_base = 1.4 * np.cos(uo_grid)
        
        # Ovary follicle/texture noise (more aggressive than uterus)
        ovary_noise_l = self._apply_organic_noise(x_o_base, y_o_base, z_o_base, scale=0.8, seed=20) * 0.3
        ovary_noise_r = self._apply_organic_noise(x_o_base, y_o_base, z_o_base, scale=0.8, seed=30) * 0.3
        
        o_norm = np.sqrt(x_o_base**2 + y_o_base**2 + z_o_base**2) + 1e-6
        xl_ov = x_o_base + (x_o_base/o_norm) * ovary_noise_l
        yl_ov = y_o_base + (y_o_base/o_norm) * ovary_noise_l
        zl_ov = z_o_base + (z_o_base/o_norm) * ovary_noise_l
        
        xr_ov = x_o_base + (x_o_base/o_norm) * ovary_noise_r
        yr_ov = y_o_base + (y_o_base/o_norm) * ovary_noise_r
        zr_ov = z_o_base + (z_o_base/o_norm) * ovary_noise_r
        
        # Add Endometrioma mass to ovaries if severe
        endo_rad = self.state['endometrioma_size_cm'] / 2.0
        if endo_rad > 0:
            # Attach cyst to one ovary randomly based on seed
            np.random.seed(42)
            cyst_x = endo_rad * np.sin(uo_grid) * np.cos(vo_grid)
            cyst_y = endo_rad * np.sin(uo_grid) * np.sin(vo_grid)
            cyst_z = endo_rad * np.cos(uo_grid)
            cyst_noise = self._apply_organic_noise(cyst_x, cyst_y, cyst_z, scale=1.5, seed=50) * 0.2
            c_norm = np.sqrt(cyst_x**2 + cyst_y**2 + cyst_z**2) + 1e-6
            cyst_x += (cyst_x/c_norm) * cyst_noise
            cyst_y += (cyst_y/c_norm) * cyst_noise
            cyst_z += (cyst_z/c_norm) * cyst_noise
            
            # Merge cyst geometry into left ovary mesh (simplified boolean union via overlap mapping)
            # Offset cyst to edge
            cyst_x -= 1.0
            cyst_z += 1.0
            
            # Basic radius overlay for visualization
            for i in range(res_o):
                for j in range(res_o):
                    if cyst_x[i,j]**2 + cyst_y[i,j]**2 + cyst_z[i,j]**2 > 0.1:
                        # Append or displace
                        pass # Kept simple for mesh integrity in plotly, visual representation follows
            # Plotly handles independent meshes better, but we will attach it visually via lesions layer
        
        left_ovary = (xl_ov - 6.5, yl_ov, zl_ov + 2.5)
        right_ovary = (xr_ov + 6.5, yr_ov, zr_ov + 2.5)
        
        # Fallopian Tubes (Higher res parametric organic tubes)
        t = np.linspace(0, 1, 80)
        theta_tube = np.linspace(0, 2*np.pi, 30)
        t_grid, theta_grid = np.meshgrid(t, theta_tube)
        
        # Left tube curve with more organic meandering
        meander_l = np.sin(t_grid * np.pi * 3) * 0.4
        curve_x_l = -2.5 - 4.0 * t_grid
        curve_y_l = 1.0 * np.sin(np.pi * t_grid) + meander_l
        curve_z_l = 4.5 - 2.0 * t_grid + 0.8 * np.sin(np.pi * t_grid) 
        
        # Fimbriae (flares at the end near ovary)
        flare = np.where(t_grid > 0.85, 1.0 + (t_grid-0.85)*8, 1.0)
        radius_tube_l = (0.35 - 0.15 * t_grid) * flare
        
        x_tube_l = curve_x_l + radius_tube_l * np.cos(theta_grid)
        y_tube_l = curve_y_l + radius_tube_l * np.sin(theta_grid)
        z_tube_l = curve_z_l + radius_tube_l * np.sin(theta_grid) * 0.5
        
        tube_noise_l = self._apply_organic_noise(x_tube_l, y_tube_l, z_tube_l, scale=1.2, seed=60) * 0.1
        left_tube = (x_tube_l + tube_noise_l, y_tube_l + tube_noise_l, z_tube_l + tube_noise_l)
        
        # Right tube organic curve
        meander_r = np.sin(t_grid * np.pi * 2.5) * 0.4
        curve_x_r = 2.5 + 4.0 * t_grid
        curve_y_r = 1.0 * np.sin(np.pi * t_grid) - meander_r
        curve_z_r = 4.5 - 2.0 * t_grid + 0.6 * np.sin(np.pi * t_grid)
        
        radius_tube_r = (0.35 - 0.15 * t_grid) * flare
        x_tube_r = curve_x_r + radius_tube_r * np.cos(theta_grid)
        y_tube_r = curve_y_r + radius_tube_r * np.sin(theta_grid)
        z_tube_r = curve_z_r + radius_tube_r * np.sin(theta_grid) * 0.5
        
        tube_noise_r = self._apply_organic_noise(x_tube_r, y_tube_r, z_tube_r, scale=1.2, seed=70) * 0.1
        right_tube = (x_tube_r + tube_noise_r, y_tube_r + tube_noise_r, z_tube_r + tube_noise_r)
        
        # Realistic Lesion distribution (clustered, not purely random)
        lesion_x, lesion_y, lesion_z = [], [], []
        lesion_colors = []
        lesion_sizes = []
        
        # Adding the endometrioma visual as a massive lesion cluster
        if endo_rad > 0:
            for _ in range(int(endo_rad * 30)):
                th = np.random.uniform(0, np.pi)
                ph = np.random.uniform(0, 2*np.pi)
                rad = endo_rad * np.random.uniform(0.8, 1.0)
                lesion_x.append(rad * np.sin(th) * np.cos(ph) - 7.5) # Attached left ovary
                lesion_y.append(rad * np.sin(th) * np.sin(ph))
                lesion_z.append(rad * np.cos(th) + 3.5)
                lesion_colors.append(0.3) # Dark old blood color "chocolate cyst"
                lesion_sizes.append(np.random.uniform(10, 20))
        
        # Regular lesions
        for _ in range(self.state['lesion_count']):
            # Cluster centers
            target = np.random.choice(['ut_pouch', 'ut_front', 'left_ovary', 'right_ovary', 'tubes'], p=[0.4, 0.2, 0.15, 0.15, 0.1])
            if target == 'ut_pouch': # Pouch of Douglas (common area)
                th = np.random.uniform(np.pi/2, np.pi)
                ph = np.random.uniform(-np.pi/4, np.pi/4) + np.pi
                rad = (3.8 + 1.6 * np.cos(th)) * 1.05
                lesion_x.append(rad * np.sin(th) * np.cos(ph))
                lesion_y.append(rad * np.sin(th) * np.sin(ph))
                lesion_z.append(5.5 * np.cos(th))
            elif target == 'ut_front':
                th = np.random.uniform(0, np.pi/2)
                ph = np.random.uniform(-np.pi/4, np.pi/4)
                rad = (3.8 + 1.6 * np.cos(th)) * 1.05
                lesion_x.append(rad * np.sin(th) * np.cos(ph))
                lesion_y.append(rad * np.sin(th) * np.sin(ph))
                lesion_z.append(5.5 * np.cos(th))
            elif target == 'left_ovary':
                th = np.random.uniform(0, np.pi)
                ph = np.random.uniform(0, 2*np.pi)
                lesion_x.append(1.9 * np.sin(th) * np.cos(ph) - 6.5)
                lesion_y.append(1.9 * np.sin(th) * np.sin(ph))
                lesion_z.append(1.9 * np.cos(th) + 2.5)
            elif target == 'right_ovary':
                th = np.random.uniform(0, np.pi)
                ph = np.random.uniform(0, 2*np.pi)
                lesion_x.append(1.9 * np.sin(th) * np.cos(ph) + 6.5)
                lesion_y.append(1.9 * np.sin(th) * np.sin(ph))
                lesion_z.append(1.9 * np.cos(th) + 2.5)
            else: # Tubes
                side = np.random.choice([-1, 1])
                t_val = np.random.uniform(0.1, 0.9)
                th_val = np.random.uniform(0, 2*np.pi)
                cx = side * (2.5 + 4.0 * t_val)
                cy = 1.0 * np.sin(np.pi * t_val)
                cz = 4.5 - 2.0 * t_val + 0.5 * np.sin(np.pi * t_val)
                r_val = 0.35 - 0.15 * t_val
                lesion_x.append(cx + r_val * np.cos(th_val) * 1.3)
                lesion_y.append(cy + r_val * np.sin(th_val) * 1.3)
                lesion_z.append(cz)
                
            lesion_colors.append(np.random.uniform(0.6, 1.0)) # Active bright red lesions
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
                rad = (3.8 + 1.6 * np.cos(th)) * 1.15
                f_lesion_x.append(rad * np.sin(th) * np.cos(ph))
                f_lesion_y.append(rad * np.sin(th) * np.sin(ph))
                f_lesion_z.append(5.5 * np.cos(th))
            elif target == 'ut_front':
                th = np.random.uniform(0, np.pi/2)
                ph = np.random.uniform(-np.pi/4, np.pi/4)
                rad = (3.8 + 1.6 * np.cos(th)) * 1.15
                f_lesion_x.append(rad * np.sin(th) * np.cos(ph))
                f_lesion_y.append(rad * np.sin(th) * np.sin(ph))
                f_lesion_z.append(5.5 * np.cos(th))
            elif target == 'left_ovary':
                th = np.random.uniform(0, np.pi)
                ph = np.random.uniform(0, 2*np.pi)
                f_lesion_x.append(2.5 * np.sin(th) * np.cos(ph) - 6.5)
                f_lesion_y.append(2.5 * np.sin(th) * np.sin(ph))
                f_lesion_z.append(2.5 * np.cos(th) + 2.5)
            elif target == 'right_ovary':
                th = np.random.uniform(0, np.pi)
                ph = np.random.uniform(0, 2*np.pi)
                f_lesion_x.append(2.5 * np.sin(th) * np.cos(ph) + 6.5)
                f_lesion_y.append(2.5 * np.sin(th) * np.sin(ph))
                f_lesion_z.append(2.5 * np.cos(th) + 2.5)
            else: # Tubes
                side = np.random.choice([-1, 1])
                t_val = np.random.uniform(0.1, 0.9)
                th_val = np.random.uniform(0, 2*np.pi)
                cx = side * (2.5 + 4.0 * t_val)
                cy = 1.0 * np.sin(np.pi * t_val)
                cz = 4.5 - 2.0 * t_val + 0.5 * np.sin(np.pi * t_val)
                r_val = 0.35 - 0.15 * t_val
                f_lesion_x.append(cx + r_val * np.cos(th_val) * 1.8)
                f_lesion_y.append(cy + r_val * np.sin(th_val) * 1.8)
                f_lesion_z.append(cz)
                
            f_lesion_colors.append(np.random.uniform(0.1, 0.4)) # Different color map range for future
            
        # Adhesions (Dense, web-like organic strands)
        adhesion_lines = []
        if self.state['adhesions_present']:
            num_strands = np.random.randint(15, 35)
            for _ in range(num_strands): 
                # Connect Uterus back to Ovaries (common adhesion)
                th1 = np.random.uniform(np.pi/2, np.pi)
                ph1 = np.random.uniform(0, 2*np.pi)
                r1 = (3.8 + 1.6 * np.cos(th1)) * 1.02
                pt1 = np.array([r1 * np.sin(th1) * np.cos(ph1), r1 * np.sin(th1) * np.sin(ph1), 5.5 * np.cos(th1)])
                
                side = np.random.choice([-1, 1])
                th2 = np.random.uniform(0, np.pi)
                ph2 = np.random.uniform(0, 2*np.pi)
                pt2 = np.array([1.5 * np.sin(th2) * np.cos(ph2) + (side * 6.5), 1.5 * np.sin(th2) * np.sin(ph2), 1.5 * np.cos(th2) + 2.5])
                
                # Generate curved web line instead of straight
                t_line = np.linspace(0, 1, 10)
                for i in range(len(t_line)-1):
                    pA = pt1 * (1 - t_line[i]) + pt2 * t_line[i]
                    pB = pt1 * (1 - t_line[i+1]) + pt2 * t_line[i+1]
                    # Add droop/sag to adhesion web
                    pA[2] -= np.sin(t_line[i] * np.pi) * 1.5
                    pB[2] -= np.sin(t_line[i+1] * np.pi) * 1.5
                    adhesion_lines.append((pA, pB))
                side = np.random.choice([-6.5, 6.5])
                pt2 = [1.4 * np.sin(th2) * np.cos(ph2) + side, 1.4 * np.sin(th2) * np.sin(ph2), 1.4 * np.cos(th2) + 2.5]
                
                adhesion_lines.append((pt1, pt2))
                
        # future_lesions: optional sizes (same length as coords) for consistency
        f_sizes = [8.0] * len(f_lesion_x) if f_lesion_x else []
        return {
            'uterus': (x_uterus, y_uterus, z_uterus),
            'left_ovary': left_ovary,
            'right_ovary': right_ovary,
            'left_tube': left_tube,
            'right_tube': right_tube,
            'lesions': (lesion_x, lesion_y, lesion_z, lesion_colors),
            'lesion_sizes': lesion_sizes,
            'future_lesions': (f_lesion_x, f_lesion_y, f_lesion_z, f_lesion_colors),
            'future_lesion_sizes': f_sizes,
            'adhesions': adhesion_lines
        }

if __name__ == "__main__":
    twin = UterusDigitalTwin()
    twin.update_from_model_prediction(0.8, 3)
    data = twin.generate_3d_scatter_data()
    print("Generated 3D nodes.")
