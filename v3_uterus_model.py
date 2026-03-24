import numpy as np
import plotly.graph_objects as go
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

# =============================================================================
# GEOMETRY GENERATORS
# =============================================================================

def generate_medically_accurate_uterus(res=100):
    # Parametric ranges
    z_v = np.linspace(0, 1, res)
    theta_v = np.linspace(0, 2 * np.pi, res)
    zg, tg = np.meshgrid(z_v, theta_v, indexing='ij')

    # FIX 1: Radius Profile (Multi-term polynomial / interp)
    z_knots = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    r_knots = [0.35, 0.55, 0.90, 2.00, 2.40, 2.50, 1.80]
    radius_z = np.interp(zg, z_knots, r_knots)

    # FIX 2: Cornual Horns (Shoulder bumps)
    # Bulge where fallopian tubes attach (z ~ 0.85)
    cornual_amp = 0.25 * np.exp(-((zg - 0.85)**2) / 0.002) * np.abs(np.cos(tg))
    radius_z += cornual_amp

    # FIX 3: Anterior-Posterior Flattening
    flatten_z_knots = [0.0, 0.5, 1.0]
    flatten_v_knots = [0.95, 0.80, 0.72]
    flatten_factor = np.interp(zg, flatten_z_knots, flatten_v_knots)

    # Main Body Height
    height = 8.0

    # Base Coordinates
    x = radius_z * np.cos(tg)
    y = radius_z * np.sin(tg) * flatten_factor
    z = height * zg

    # FIX 5: Anatomically Shaped Organic Noise
    # Higher freq at fundus, lower at cervix
    noise_scale = np.interp(zg, [0.0, 1.0], [0.8, 2.5])
    noise_amp = 0.04 * radius_z # Amplitude scales with local radius
    
    noise = get_organic_noise(x, y, z, scale=1.0, seed=123) # Seeded noise base
    # Modulate noise by height-dependent scale
    x += np.cos(tg) * noise * noise_amp * noise_scale
    y += np.sin(tg) * noise * noise_amp * noise_scale
    
    return x, y, z

def generate_cervix(res=50):
    # FIX 4: Add cervix as a separate parametric cylinder (barrel shaped)
    z_v = np.linspace(-3.0, 0.0, res)
    theta_v = np.linspace(0, 2 * np.pi, res)
    zg, tg = np.meshgrid(z_v, theta_v, indexing='ij')

    # Radius profile for barrel shape
    # Peak at 1.2 mid-cervix, 0.35 at os
    z_knots = [-3.0, -1.5, 0.0]
    r_knots = [0.35, 1.2, 0.55] # Matches uterine start at 0.55
    radius_z = np.interp(zg, z_knots, r_knots)

    flatten_factor = 0.95 # Less flattened than uterus

    x = radius_z * np.cos(tg)
    y = radius_z * np.sin(tg) * flatten_factor
    z = zg

    # Noise for consistency
    noise = get_organic_noise(x, y, z, scale=0.8, seed=456)
    x += np.cos(tg) * noise * 0.02
    y += np.sin(tg) * noise * 0.02

    return x, y, z

def generate_fallopian_tube(side='left', p_start=None, p_end=None, res=60):
    # FIX 6: Realistic Fallopian Tubes with 3-point Bezier
    t = np.linspace(0, 1, res)
    
    # Point 1: Control Point lateral + superior
    offset_x = -4.0 if side == 'left' else 4.0
    p_ctrl = p_start + np.array([offset_x, 0.0, 1.5])
    
    # Bezier Spline
    curve = np.outer((1-t)**2, p_start) + \
            np.outer(2*(1-t)*t, p_ctrl) + \
            np.outer(t**2, p_end)
    
    # Tube Radius
    r_base = np.interp(t, [0, 0.9, 1.0], [0.18, 0.35, 0.8])
    
    theta = np.linspace(0, 2*np.pi, 20)
    tube_x = np.zeros((20, res))
    tube_y = np.zeros((20, res))
    tube_z = np.zeros((20, res))
    
    for i in range(res):
        rad = r_base[i]
        tube_x[:, i] = curve[i, 0] + rad * np.cos(theta)
        tube_y[:, i] = curve[i, 1] + rad * np.sin(theta)
        tube_z[:, i] = curve[i, 2] + rad * np.sin(theta) * 0.2

    # Fimbriae
    f_parts = []
    p_last = curve[-1]
    num_fingers = 10
    for i in range(num_fingers):
        angle = (i / num_fingers) * 2 * np.pi
        length = 0.5
        direction = np.array([np.cos(angle), np.sin(angle), -0.5]) 
        direction /= np.linalg.norm(direction)
        f_parts.append((p_last, p_last + direction * length))

    return tube_x, tube_y, tube_z, f_parts

def generate_ovary(side='left', pos=None, res=40):
    # FIX 7: Textured ellipsoids (3x2x1.5 cm)
    u = np.linspace(0, np.pi, res)
    v = np.linspace(0, 2 * np.pi, res)
    ug, vg = np.meshgrid(u, v)

    a, b, c = 1.5, 1.0, 2.0
    x = a * np.sin(ug) * np.cos(vg)
    y = b * np.sin(ug) * np.sin(vg)
    z = c * np.cos(ug)

    noise = get_organic_noise(x, y, z, scale=1.5, seed=789)
    x += (x / a) * noise * 0.12
    y += (y / b) * noise * 0.12
    z += (z / c) * noise * 0.12

    x, y, z = x + pos[0], y + pos[1], z + pos[2]

    # Follicles
    follicles = []
    for _ in range(4):
        f_u, f_v = np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)
        f_pos = np.array([a * np.sin(f_u) * np.cos(f_v), b * np.sin(f_u) * np.sin(f_v), c * np.cos(f_u)])
        follicles.append((f_pos + pos, np.random.uniform(0.2, 0.4)))

    return x, y, z, follicles

def generate_broad_ligament(ux, uy, uz, pelvic_wall_x=12.0):
    # FIX 8: Broad ligament
    res_z = ux.shape[0]
    mid = ux.shape[1] // 2
    
    # Left
    ut_edge_l = np.array([ux[:, mid], uy[:, mid], uz[:, mid]])
    w_v = np.linspace(0, 1, 10)
    zg, wg = np.meshgrid(np.arange(res_z), w_v, indexing='ij')
    lig_l_x = ut_edge_l[0][:, None] * (1 - wg) + (-pelvic_wall_x) * wg
    lig_l_y = ut_edge_l[1][:, None] * (1 - wg) + 0.0 * wg
    lig_l_z = ut_edge_l[2][:, None]
    lig_l_y += get_organic_noise(lig_l_x, lig_l_y, lig_l_z, scale=0.5, seed=10) * 0.2
    
    # Right
    ut_edge_r = np.array([ux[:, 0], uy[:, 0], uz[:, 0]])
    lig_r_x = ut_edge_r[0][:, None] * (1 - wg) + pelvic_wall_x * wg
    lig_r_y = ut_edge_r[1][:, None] * (1 - wg) + 0.0 * wg
    lig_r_z = ut_edge_r[2][:, None]
    lig_r_y += get_organic_noise(lig_r_x, lig_r_y, lig_r_z, scale=0.5, seed=20) * 0.2

    return (lig_l_x, lig_l_y, lig_l_z), (lig_r_x, lig_r_y, lig_r_z)

# =============================================================================
# MAIN RENDERING
# =============================================================================

def build_model():
    ux, uy, uz = generate_medically_accurate_uterus()
    cx, cy, cz = generate_cervix()
    
    idx_cornua = int(ux.shape[0] * 0.88)
    p_lc = np.array([ux[idx_cornua, ux.shape[1]//2], uy[idx_cornua, ux.shape[1]//2], uz[idx_cornua, ux.shape[1]//2]])
    p_rc = np.array([ux[idx_cornua, 0], uy[idx_cornua, 0], uz[idx_cornua, 0]])
    p_lo, p_ro = np.array([-5.5, 0.5, 5.5]), np.array([5.5, 0.5, 5.5])
    
    lx_t, ly_t, lz_t, l_f = generate_fallopian_tube('left', p_lc, p_lo)
    rx_t, ry_t, rz_t, r_f = generate_fallopian_tube('right', p_rc, p_ro)
    lx_ov, ly_ov, lz_ov, l_foll = generate_ovary('left', p_lo)
    rx_ov, ry_ov, rz_ov, r_foll = generate_ovary('right', p_ro)
    lig_l, lig_r = generate_broad_ligament(ux, uy, uz)

    fig = go.Figure()
    
    # Traces
    fig.add_trace(go.Surface(x=ux, y=uy, z=uz, backgroundcolor='rgb(220,120,150)', opacity=0.30, name='Uterus Body', showscale=False))
    # Approximation of cavity
    fig.add_trace(go.Surface(x=ux*0.6, y=uy*0.3, z=uz, backgroundcolor='rgb(160,40,55)', opacity=0.85, name='Endometrial Cavity', showscale=False))
    fig.add_trace(go.Surface(x=cx, y=cy, z=cz, backgroundcolor='rgb(230,150,170)', opacity=0.60, name='Cervix', showscale=False))
    
    fig.add_trace(go.Surface(x=lx_t, y=ly_t, z=lz_t, backgroundcolor='rgb(255,190,80)', opacity=0.90, name='Left Tube', showscale=False))
    fig.add_trace(go.Surface(x=rx_t, y=ry_t, z=rz_t, backgroundcolor='rgb(255,190,80)', opacity=0.90, name='Right Tube', showscale=False))
    
    fimb_x, fimb_y, fimb_z = [], [], []
    for f in (l_f + r_f): fimb_x.extend([f[0][0], f[1][0], None]); fimb_y.extend([f[0][1], f[1][1], None]); fimb_z.extend([f[0][2], f[1][2], None])
    fig.add_trace(go.Scatter3d(x=fimb_x, y=fimb_y, z=fimb_z, mode='lines', line=dict(color='rgb(255,160,60)', width=4), name='Fimbriae'))

    fig.add_trace(go.Surface(x=lx_ov, y=ly_ov, z=lz_ov, backgroundcolor='rgb(220,175,130)', opacity=0.92, name='Left Ovary', showscale=False))
    fig.add_trace(go.Surface(x=rx_ov, y=ry_ov, z=rz_ov, backgroundcolor='rgb(220,175,130)', opacity=0.92, name='Right Ovary', showscale=False))
    
    foll_x, foll_y, foll_z = [], [], []
    for f in (l_foll + r_foll): foll_x.append(f[0][0]); foll_y.append(f[0][1]); foll_z.append(f[0][2])
    fig.add_trace(go.Scatter3d(x=foll_x, y=foll_y, z=foll_z, mode='markers', marker=dict(size=6, color='rgb(240,210,160)', opacity=0.85), name='Follicles'))

    fig.add_trace(go.Surface(x=lig_l[0], y=lig_l[1], z=lig_l[2], backgroundcolor='rgb(255,180,200)', opacity=0.12, name='Left Ligament', showscale=False))
    fig.add_trace(go.Surface(x=lig_r[0], y=lig_r[1], z=lig_r[2], backgroundcolor='rgb(255,180,200)', opacity=0.12, name='Right Ligament', showscale=False))

    fig.update_layout(
        title="Uterine Digital Twin — Anatomical 3D Model",
        scene=dict(
            xaxis=dict(title='Lateral (cm)'), yaxis=dict(title='Anterior-Posterior (cm)'), zaxis=dict(title='Superior (cm)'),
            bgcolor='rgb(10, 10, 20)', camera=dict(eye=dict(x=1.5, y=-2.0, z=0.8))
        ),
        template="plotly_dark",
        updatemenus=[dict(type="buttons", direction="down", buttons=[
            dict(label="Show All", method="restyle", args=[{"visible": [True]*11}]),
            dict(label="Uterus Only", method="restyle", args=[{"visible": [True, True, True] + [False]*8}]),
            dict(label="Appendages Only", method="restyle", args=[{"visible": [False]*3 + [True]*8}])
        ], x=0.05, y=1.1)]
    )

    return fig

if __name__ == "__main__":
    fig = build_model()
    fig.show()
