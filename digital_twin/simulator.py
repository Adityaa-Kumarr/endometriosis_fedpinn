import numpy as np
import os

# =============================================================================
# HELPER: PURE NUMPY 3D NOISE (To avoid external dependencies)
# =============================================================================

def _hash3(x, y, z, seed=0):
    n = np.sin(x * 12.9898 + y * 78.233 + z * 45.164 + seed) * 43758.5453
    return n - np.floor(n)

def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)

def _pnoise3_numpy(x, y, z, scale, seed=42):
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

def get_organic_noise(x, y, z, scale=0.5, octaves=4, seed=42):
    out = np.zeros_like(x)
    for o in range(octaves):
        f = 2 ** o
        out += _pnoise3_numpy(x, y, z, scale * f, seed + o * 17) / f
    return out

class UterusDigitalTwin:
    """
    Simulates the physiological state of a uterus for a Digital Twin dashboard.
    Outputs metrics and visual parameters representing Endometriosis progression.
    
    Anatomically Realistic (v3):
    - Refined pear shape with cornual horns
    - Cervical barrel extension
    - Bezier-curved fallopian tubes with fimbriae
    - Textured ovaries with follicles
    """
    def __init__(self, patient_base_state=None):
        self.state = patient_base_state or {
            'inflammation_level': 0.0,
            'lesion_count': 0,
            'adhesions_present': False,
            'endometrioma_size_cm': 0.0,
            'future_lesion_multiplier': 0.0
        }
        
    def update_from_model_prediction(self, probability, stage, future_risk=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state['inflammation_level'] = min(1.0, probability * 1.5)
        
        if future_risk is not None:
            self.state['future_lesion_multiplier'] = min(5.0, future_risk * 2.5) 
        else:
            self.state['future_lesion_multiplier'] = 0.0
        
        # Stage logic for lesion/cyst parameters
        if stage == 0:
            self.state['lesion_count'], self.state['adhesions_present'], self.state['endometrioma_size_cm'] = 0, False, 0.0
        elif stage == 1:
            self.state['lesion_count'], self.state['adhesions_present'], self.state['endometrioma_size_cm'] = np.random.randint(2, 8), False, 0.0
        elif stage == 2:
            self.state['lesion_count'], self.state['adhesions_present'], self.state['endometrioma_size_cm'] = np.random.randint(12, 25), False, np.random.uniform(0.1, 1.5)
        elif stage == 3:
            self.state['lesion_count'], self.state['adhesions_present'], self.state['endometrioma_size_cm'] = np.random.randint(30, 60), True, np.random.uniform(1.5, 4.0)
        elif stage == 4:
            self.state['lesion_count'], self.state['adhesions_present'], self.state['endometrioma_size_cm'] = np.random.randint(60, 120), True, np.random.uniform(4.0, 10.0)
            
        return self.state

    def generate_3d_scatter_data(self, reference_shape=None, resolution=1.0, patient_seed=None):
        """Generates refined 3D anatomical data for the UI."""
        res = int(80 * resolution)
        rng = np.random.RandomState(patient_seed if patient_seed is not None else 42)
        
        # 1. Uterus Body (FIX 1-3, 5)
        z_v = np.linspace(0, 1, res)
        theta_v = np.linspace(0, 2 * np.pi, res)
        zg, tg = np.meshgrid(z_v, theta_v, indexing='ij')

        # Radius profile
        z_knots = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
        r_knots = [0.35, 0.55, 0.90, 2.00, 2.40, 2.50, 1.80]
        radius_z = np.interp(zg, z_knots, r_knots)
        
        # Cornual horns
        radius_z += 0.25 * np.exp(-((zg - 0.85)**2) / 0.002) * np.abs(np.cos(tg))
        # AP Flattening
        flatten_factor = np.interp(zg, [0.0, 0.5, 1.0], [0.95, 0.80, 0.72])

        ux, uy, uz = radius_z * np.cos(tg), radius_z * np.sin(tg) * flatten_factor, 8.0 * zg

        # Noise
        noise_scale = np.interp(zg, [0.0, 1.0], [0.8, 2.5])
        u_noise = get_organic_noise(ux, uy, uz, scale=1.0, seed=patient_seed or 42)
        ux += np.cos(tg) * u_noise * 0.04 * radius_z * noise_scale
        uy += np.sin(tg) * u_noise * 0.04 * radius_z * noise_scale

        # 2. Cervix (FIX 4)
        zc_v = np.linspace(-3.0, 0.0, res // 2)
        zcg, tcg = np.meshgrid(zc_v, theta_v, indexing='ij')
        radius_c = np.interp(zcg, [-3.0, -1.5, 0.0], [0.35, 1.2, 0.55])
        cx, cy, cz = radius_c * np.cos(tcg), radius_c * np.sin(tcg) * 0.95, zcg
        
        # 3. Appendages (FIX 6-7)
        idx_c = int(res * 0.88)
        p_lc = np.array([ux[idx_c, res//2], uy[idx_c, res//2], uz[idx_c, res//2]])
        p_rc = np.array([ux[idx_c, 0], uy[idx_c, 0], uz[idx_c, 0]])
        p_lo, p_ro = np.array([-5.5, 0.5, 5.5]), np.array([5.5, 0.5, 5.5])
        
        def gen_tube(p_start, p_end, side):
            t = np.linspace(0, 1, 50)
            p_ctrl = p_start + np.array([-4.0 if side=='left' else 4.0, 0.0, 1.5])
            curve = np.outer((1-t)**2, p_start) + np.outer(2*(1-t)*t, p_ctrl) + np.outer(t**2, p_end)
            r_tube = np.interp(t, [0, 0.9, 1.0], [0.18, 0.35, 0.8])
            th = np.linspace(0, 2*np.pi, 20)
            tx = curve[:, 0][:, None] + r_tube[:, None] * np.cos(th)
            ty = curve[:, 1][:, None] + r_tube[:, None] * np.sin(th)
            tz = curve[:, 2][:, None] + r_tube[:, None] * np.sin(th) * 0.2
            return tx.T, ty.T, tz.T

        lx_t, ly_t, lz_t = gen_tube(p_lc, p_lo, 'left')
        rx_t, ry_t, rz_t = gen_tube(p_rc, p_ro, 'right')

        def gen_ov(pos):
            u, v = np.linspace(0, np.pi, 30), np.linspace(0, 2*np.pi, 30)
            ug, vg = np.meshgrid(u, v)
            ox, oy, oz = 1.5 * np.sin(ug)*np.cos(vg), 1.0*np.sin(ug)*np.sin(vg), 2.0*np.cos(ug)
            noise = get_organic_noise(ox, oy, oz, scale=1.5, seed=789)
            return ox + (ox/1.5)*noise*0.12 + pos[0], oy + (oy/1.0)*noise*0.12 + pos[1], oz + (oz/2.0)*noise*0.12 + pos[2]

        lx_ov, ly_ov, lz_ov = gen_ov(p_lo)
        rx_ov, ry_ov, rz_ov = gen_ov(p_ro)

        # Lesions
        lesion_x, lesion_y, lesion_z = [], [], []
        n_les = self.state.get('lesion_count', 0)
        for _ in range(int(n_les)):
            iz, it = rng.randint(0, res), rng.randint(0, res)
            lesion_x.append(ux[iz, it] * 1.02)
            lesion_y.append(uy[iz, it] * 1.02)
            lesion_z.append(uz[iz, it])

        # Broad Ligament
        def gen_lig(ut_edge, px_wall):
            w_v = np.linspace(0, 1, 10)
            zg_lig, wg = np.meshgrid(np.arange(res), w_v, indexing='ij')
            lx = ut_edge[0][:, None] * (1 - wg) + px_wall * wg
            ly = ut_edge[1][:, None] * (1 - wg) + 0.0 * wg
            lz = ut_edge[2][:, None]
            ly += get_organic_noise(lx, ly, lz, scale=0.5, seed=10) * 0.2
            return lx, ly, lz

        lig_l_x, lig_l_y, lig_l_z = gen_lig(np.array([ux[:, res//2], uy[:, res//2], uz[:, res//2]]), -11.0)
        lig_r_x, lig_r_y, lig_r_z = gen_lig(np.array([ux[:, 0], uy[:, 0], uz[:, 0]]), 11.0)

        # Part labels
        def _centroid(xx, yy, zz):
            return (float(np.mean(xx)), float(np.mean(yy)), float(np.mean(zz)))

        part_labels = [
            ('Uterus Body', *_centroid(ux, uy, uz)),
            ('Cervix', *_centroid(cx, cy, cz)),
            ('Left Ovary', *_centroid(lx_ov, ly_ov, lz_ov)),
            ('Right Ovary', *_centroid(rx_ov, ry_ov, rz_ov)),
        ]

        return {
            'uterus': (ux, uy, uz),
            'cervix': (cx, cy, cz),
            'left_ovary': (lx_ov, ly_ov, lz_ov),
            'right_ovary': (rx_ov, ry_ov, rz_ov),
            'left_tube': (lx_t, ly_t, lz_t),
            'right_tube': (rx_t, ry_t, rz_t),
            'lesions': (lesion_x, lesion_y, lesion_z, [0.9]*len(lesion_x)),
            'lesion_sizes': [8.0]*len(lesion_x),
            'future_lesions': ([], [], [], []),
            'adhesions': [],
            'broad_ligament_l': (lig_l_x, lig_l_y, lig_l_z),
            'broad_ligament_r': (lig_r_x, lig_r_y, lig_r_z),
            'part_labels': part_labels
        }

    def generate_temporal_progression(self, base_probability, base_stage, future_risk_5yr, num_steps=4):
        time_points = [0, 1, 3, 5]
        lam = -np.log(1 - future_risk_5yr) / 5.0 if 0 < future_risk_5yr < 1 else 0.1
        progression = []
        for t in time_points:
            p_t = min(0.99, base_probability + (1 - base_probability) * (1 - np.exp(-lam * t)))
            s_t = base_stage + (2 if p_t > 0.85 else 1 if p_t > 0.65 else 0)
            progression.append({
                'time_years': t, 'probability': p_t, 'stage': min(4, s_t),
                'inflammation': min(1.0, p_t * 1.5),
                'lesion_count': int(p_t * 100 * (min(4, s_t) + 1) / 5)
            })
        return progression

if __name__ == "__main__":
    twin = UterusDigitalTwin()
    data = twin.generate_3d_scatter_data()
    print("V3 Model Geometry Generated Successfully.")
