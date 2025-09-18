import taichi as ti
import numpy as np
from pathlib import Path
import csv, json, time




# ──────────────────────────────────────────────────────────────────────────────
# Globals / constants
# ──────────────────────────────────────────────────────────────────────────────
PI = 3.141592653589793

ti.init(arch=ti.gpu)

# Simulation parameters
dim         = 2
n_particles = 20000
n_grid      = 128
dx          = 1.0 / n_grid
inv_dx      = float(n_grid)
dt          = 2e-4

# Material (Neo-Hookean)
p_rho     = 1.0
p_vol     = (dx * 0.5) ** 2
p_mass    = p_rho * p_vol
E         = 5.65e4
nu        = 0.185
mu_0      = E / (2 * (1 + nu))
lambda_0  = E * nu / ((1 + nu) * (1 - 2 * nu))

# Floor & domain
floor_level    = 0.0
floor_friction = 0.4  # param kept; current logic clamps velocities

# ──────────────────────────────────────────────────────────────────────────────
# Fixed 1-link arm with vertically oscillating base
# ──────────────────────────────────────────────────────────────────────────────
L_arm  = 0.12                   # length of the single rigid link
theta0 = 0.0                    # fixed orientation (0 => straight down)
y0, base_x = 0.40, 0.50         # mean base position
A, ω       = 0.075, 0.50        # vertical oscillation amplitude & angular freq
base_y     = y0
time_t     = 0.0
Time_period = 2 * PI / ω * 2    # run two periods

# Roller/contact
roller_radius         = 0.025
roller_center         = ti.Vector.field(dim, dtype=ti.f32, shape=1)
roller_velocity       = ti.Vector.field(dim, dtype=ti.f32, shape=1)
contact_force_vec     = ti.Vector.field(dim, dtype=ti.f32, shape=1)

# ──────────────────────────────────────────────────────────────────────────────
# MPM fields
# ──────────────────────────────────────────────────────────────────────────────
x      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
v      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
F      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
C      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
J      = ti.field(dtype=ti.f32, shape=n_particles)
grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

# Soft body initial placement
half_radius   = 0.2
soft_center_x = 0.5

# ──────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ──────────────────────────────────────────────────────────────────────────────
@ti.func
def neo_hookean_stress(F_i: ti.template()):
    J_det = F_i.determinant()
    FinvT = F_i.inverse().transpose()
    return mu_0 * (F_i - FinvT) + lambda_0 * ti.log(J_det) * FinvT

@ti.func
def quadratic_weights(fx: ti.template()):
    # MPM quadratic B-spline weights in 2D; returns three vectors
    return (0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2)

# ──────────────────────────────────────────────────────────────────────────────
# Initialization
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def init_mpm():
    for p in range(n_particles):
        # Disk sampling
        u     = ti.random()
        r     = half_radius * ti.sqrt(u)
        theta = ti.random() * PI
        x[p]  = ti.Vector([soft_center_x + r * ti.cos(theta),
                           floor_level    + r * ti.sin(theta)])
        v[p] = ti.Vector.zero(ti.f32, dim)
        F[p] = ti.Matrix.identity(ti.f32, dim)
        J[p] = 1.0
        C[p] = ti.Matrix.zero(ti.f32, dim, dim)

    # Place rollers at FK
    base = ti.Vector([base_x, base_y])
    tip  = base + ti.Vector([ti.sin(theta0), -ti.cos(theta0)]) * L_arm
    roller_center[0]   = tip
    roller_velocity[0] = ti.Vector.zero(ti.f32, dim)
    contact_force_vec[0] = ti.Vector.zero(ti.f32, dim)

    
# ──────────────────────────────────────────────────────────────────────────────
# P2G
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def p2g():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, dim)
        grid_m[I] = 0.0

    for p in range(n_particles):
        Xp   = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx   = Xp - base.cast(ti.f32)

        w0, w1, w2 = quadratic_weights(fx)
        w = ti.static([w0, w1, w2])

        stress = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * neo_hookean_stress(F[p])
        affine = stress + p_mass * C[p]

        for i, j in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([i, j])
            dpos = (offs.cast(ti.f32) - fx) * dx
            wt   = w[i].x * w[j].y
            momentum = p_mass * v[p] + affine @ dpos
            grid_v[base + offs] += wt * momentum
            grid_m[base + offs] += wt * p_mass

# ──────────────────────────────────────────────────────────────────────────────
# Grid updates + contact (normal-only with rollers, floor/walls clamp)
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def apply_grid_forces_and_detect():
    contact_force_vec[0] = ti.Vector.zero(ti.f32, dim)

    for I in ti.grouped(grid_m):
        m = grid_m[I]
        if m > 0:
            v_old = grid_v[I] / m
            v_new = v_old + dt * ti.Vector([0.0, -9.8])  # gravity
            pos   = I.cast(ti.f32) * dx

            # Right roller contact (normal component)
            rel_r = pos - roller_center[0]
            if rel_r.norm() < roller_radius:
                rv     = roller_velocity[0]
                n      = rel_r.normalized()
                v_norm = n * n.dot(rv)
                v_tan  = v_old - n * (n.dot(v_old))
                v_new  = v_tan + v_norm
                delta_v = v_new - v_old
                f_imp   = m * delta_v / dt
                contact_force_vec[0] += f_imp

            

            # Floor clamp
            if pos.y < floor_level + dx:
                if v_new.y < 0:
                    v_new.y = 0
                v_new.x = 0

            # Walls clamp
            if pos.x < dx:
                v_new.x = 0
            if pos.x > 1.0 - dx:
                v_new.x = 0

            grid_v[I] = v_new * m

# ──────────────────────────────────────────────────────────────────────────────
# G2P + boundary clamps
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def g2p():
    for p in range(n_particles):
        Xp   = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx   = Xp - base.cast(ti.f32)

        w0, w1, w2 = quadratic_weights(fx)
        w = ti.static([w0, w1, w2])

        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)

        for i, j in ti.static(ti.ndrange(3, 3)):
            offs   = ti.Vector([i, j])
            dpos   = (offs.cast(ti.f32) - fx) * dx
            wt     = w[i].x * w[j].y
            gv     = grid_v[base + offs] / grid_m[base + offs]
            new_v += wt * gv
            new_C += 4.0 * inv_dx * wt * gv.outer_product(dpos)

        # Update particle state
        v[p] = new_v
        C[p] = new_C
        x[p] += dt * new_v

        # Floor & walls clamps
        if x[p].y < floor_level:
            x[p].y = floor_level
            v[p].y = 0
            v[p].x = 0

        if x[p].x < dx:
            x[p].x = dx
            v[p].x = 0
        if x[p].x > 1.0 - dx:
            x[p].x = 1.0 - dx
            v[p].x = 0
        if x[p].y > 1.0 - dx:
            x[p].y = 1.0 - dx
            v[p].y = 0

        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * new_C) @ F[p]
        J[p] = F[p].determinant()

# ──────────────────────────────────────────────────────────────────────────────
# Arm/base dynamics (numpy host code)
# ──────────────────────────────────────────────────────────────────────────────
def update_base_and_arm():
    global time_t, base_y

    # Time & base vertical update
    time_t += dt * 15
    base_y = y0 + A * np.cos(ω * time_t)


    base = np.array([base_x, base_y], dtype=np.float32)
    ee_o = roller_center[0].to_numpy()
    ee_n = base + np.array([np.sin(theta0), -np.cos(theta0)], dtype=np.float32) * L_arm
    rv   = (ee_n - ee_o) / dt

    roller_center[0]   = ee_n.tolist()
    roller_velocity[0] = rv.tolist()

# ──────────────────────────────────────────────────────────────────────────────
# Robust numpy helpers
# ──────────────────────────────────────────────────────────────────────────────
def hexrgb(h: int):
    return ((h >> 16) & 255, (h >> 8) & 255, h & 255)

def safe_minmax_norm(a, eps=1e-8):
    a = np.asarray(a, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    r = float(np.ptp(a))
    if not np.isfinite(r) or r < eps:
        return np.zeros_like(a)
    amin = float(np.min(a))
    out = (a - amin) / (r + eps)
    return np.clip(out, 0.0, 1.0)

def pack_rgb_array(rgb_arr):
    a = np.nan_to_num(rgb_arr, nan=0.0, posinf=255.0, neginf=0.0)
    a = np.clip(a, 0, 255).astype(np.uint32)
    return (a[:, 0] << 16) + (a[:, 1] << 8) + (a[:, 2])

# Tissue palettes
tissue_palettes = {
    "muscle":   {"low": np.array([139,   0,   0], dtype=np.float32),
                 "mid": np.array([205,  92,  92], dtype=np.float32),
                 "high":np.array([255, 182, 193], dtype=np.float32)},
    "organic":  {"low": np.array([102,  51,   0], dtype=np.float32),
                 "mid": np.array([153, 102,  51], dtype=np.float32),
                 "high":np.array([255, 224, 189], dtype=np.float32)},
    "vascular": {"low": np.array([120,   0,   0], dtype=np.float32),
                 "mid": np.array([200,   0,   0], dtype=np.float32),
                 "high":np.array([255,  99,  71], dtype=np.float32)},
    "surgical": {"low": np.array([178, 102, 102], dtype=np.float32),
                 "mid": np.array([255, 160, 160], dtype=np.float32),
                 "high":np.array([255, 230, 230], dtype=np.float32)},
    "classic":  {"low": np.array(hexrgb(0xC74A4A), dtype=np.float32),
                 "mid": np.array(hexrgb(0xF59A9A), dtype=np.float32),
                 "high":np.array(hexrgb(0xF7C6C6), dtype=np.float32)},
}
current_palette = "muscle"
pal = tissue_palettes[current_palette]
TISSUE_LOW, TISSUE_MID, TISSUE_HIGH = pal["low"], pal["mid"], pal["high"]

# Arm draw palette
ARM_BODY    = 0x1F2937
ARM_EDGE    = 0x374151
JOINT_COLOR = 0xFF7A59
ROLLER_EDGE = 0x222222

def draw_arm(base_x, base_y, theta0, L, ee_center):
    base_pt = np.array([base_x, base_y], dtype=np.float32)
    tip     = ee_center[0].to_numpy()

    # single fixed link (base -> tip)
    j2 = base_pt + np.array([np.sin(theta0), -np.cos(theta0)], dtype=np.float32) * L

    # shadow
    off = np.array([0.003, -0.003], dtype=np.float32)
    gui.line(begin=base_pt + off, end=j2 + off, radius=3, color=0x151515)
    gui.circle(base_pt + off, radius=5, color=0x151515)

    # main body
    gui.line(begin=base_pt, end=j2, radius=3, color=ARM_BODY)
    gui.circle(base_pt, radius=5, color=JOINT_COLOR)

    # edge highlight
    lit = np.array([+0.0015, +0.0015], dtype=np.float32)
    gui.line(begin=base_pt + lit, end=j2 + lit, radius=1, color=ARM_EDGE)

    # roller ring at the tip
    rr = int(roller_radius * 1024)
    gui.circle(tip, radius=rr + 2, color=ROLLER_EDGE)
    gui.circle(tip, radius=rr,     color=0xAAAAAA)

    
def force_mag(obj):
    """Safe magnitude from Taichi field, ndarray, list, or scalar."""
    if hasattr(obj, 'to_numpy'):
        arr = obj.to_numpy()
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1:
            return float(np.linalg.norm(arr))
        return float(np.linalg.norm(arr[0]))
    try:
        return float(np.linalg.norm(np.asarray(obj, dtype=np.float32)))
    except Exception:
        return float(obj)



import imageio

class Recorder:
    def __init__(self, output_file='output.gif', fps=30):
        self.output_file = output_file
        self.fps = fps
        self.frames = []

    def add_frame(self, frame):
        """
        Add a frame to the recorder.
        frame should be a NumPy array (e.g., from GUI.get_image() or a screenshot).
        """
        self.frames.append(frame)

    def save(self):
        """Write the accumulated frames to the video file."""
        imageio.mimsave(self.output_file, self.frames, fps=self.fps)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
init_mpm()
gui = ti.GUI('MPM + Single Passive Arm (Attached & Outward)', res=(1024, 1024))


# recorder       = Recorder(output_file='simulation_dual_arm.gif', fps=30)
        

while gui.running and time_t < Time_period:
    # physics substepsr
    for _ in range(15):
        p2g()
        apply_grid_forces_and_detect()
        g2p()
        update_base_and_arm()

    # particle positions as float32, sanitized
    x_np = np.nan_to_num(x.to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # position-only shading
    t_pos = safe_minmax_norm(x_np[:, 1])[:, None]
    t_lo  = np.clip(2.0 * t_pos, 0.0, 1.0)
    t_hi  = np.clip(2.0 * (t_pos - 0.5), 0.0, 1.0)
    base01 = (1.0 - t_lo) * TISSUE_LOW + t_lo * TISSUE_MID
    base   = (1.0 - t_hi) * base01     + t_hi * TISSUE_HIGH

    centroid = np.mean(x_np, axis=0, keepdims=True)
    edge_w   = safe_minmax_norm(np.linalg.norm(x_np - centroid, axis=1))[:, None]
    base     = base * (1.0 - 0.15 * edge_w)

    colors_uint32 = pack_rgb_array(base)
    shadow_offset = np.array([0.0025, -0.0025], dtype=np.float32)
    gui.circles(x_np + shadow_offset, radius=2.2, color=0x151515)
    gui.circles(x_np,                 radius=1.8, color=colors_uint32)

    # contact halo
    ee   = roller_center[0].to_numpy()
    cf   = force_mag(contact_force_vec[0])
    halo_r = int(np.clip(6 + 0.004 * cf, 6, 22))
    gui.circle(ee, radius=halo_r, color=0xFFFFFF)

    # arms
    draw_arm(base_x, base_y, theta0, L_arm, roller_center)

    # HUD
    gui.text(f'Time: {time_t:.3f} s',
             pos=(0.02, 0.95), color=0xFFFFFF)
    gui.text(f'Force(L): {float(np.linalg.norm(contact_force_vec[0])):.2f} N',
             pos=(0.02, 0.92), color=0xFFFFFF)
    
    # gui.text('Lighting: top-left | Tissue: stress-mapped | Arm: matte graphite',
    #          pos=(0.02, 0.92), color=0xC0C0C0)


    frame = gui.get_image()
    frame = (np.array(frame) * 255).astype(np.uint8)  # Convert float32 to uint8
    frame = np.rot90(frame, k=1)  # Rotate 90 degrees counterclockwise
        
    # recorder.add_frame(frame)

    gui.show()


# recorder.save()