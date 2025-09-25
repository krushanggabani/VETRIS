import taichi as ti
import numpy as np
from math import pi as PI
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

# ----------------------------------------------------------------------------
# Simulation parameters
# ----------------------------------------------------------------------------
dim         = 2
n_particles = 20000
n_grid      = 128
dx          = 1.0 / n_grid
inv_dx      = float(n_grid)
dt          = 2e-5

# Domain / gravity / floor
floor_level    = 0.0
floor_friction = 0.4

# ----------------------------------------------------------------------------
# Collider: switch to SQUARE (AABB)
# ----------------------------------------------------------------------------
USE_SQUARE = True          # <--- square collider on
BOX_HX, BOX_HY = 0.03, 0.03  # half-sizes of the square (meters)

# We reuse roller_center/velocity as the collider's center/velocity.
roller_center   = ti.Vector.field(dim, dtype=ti.f32, shape=1)
roller_velocity = ti.Vector.field(dim, dtype=ti.f32, shape=1)
contact_force   = ti.Vector.field(dim, dtype=ti.f32, shape=1)
square_half     = ti.Vector.field(dim, dtype=ti.f32, shape=1)  # (hx, hy)

# ----------------------------------------------------------------------------
# MPM fields
# ----------------------------------------------------------------------------
x      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
v      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
C      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
FE     = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
FN     = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)

p_rho  = 1.0
p_vol  = (dx * 0.5) ** 2
p_mass = p_rho * p_vol

grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

# ----------------------------------------------------------------------------
# Viscoelastic params (Silly Rubber split)
# ----------------------------------------------------------------------------
mu_E      = ti.field(dtype=ti.f32, shape=())
lambda_E  = ti.field(dtype=ti.f32, shape=())
mu_N      = ti.field(dtype=ti.f32, shape=())
lambda_N  = ti.field(dtype=ti.f32, shape=())
nu_d      = ti.field(dtype=ti.f32, shape=())
nu_v      = ti.field(dtype=ti.f32, shape=())

@ti.kernel
def set_material_params(E_eq: ti.f32, nu_eq: ti.f32,
                        E_ne: ti.f32, nu_ne: ti.f32,
                        nu_dev: ti.f32, nu_vol: ti.f32):
    mu_E[None]     = E_eq / (2.0 * (1.0 + nu_eq))
    lambda_E[None] = E_eq * nu_eq / ((1.0 + nu_eq) * (1.0 - 2.0 * nu_eq))
    mu_N[None]     = E_ne / (2.0 * (1.0 + nu_ne))
    lambda_N[None] = E_ne * nu_ne / ((1.0 + nu_ne) * (1.0 - 2.0 * nu_ne))
    nu_d[None]     = nu_dev
    nu_v[None]     = nu_vol

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@ti.func
def quadratic_weights(fx: ti.types.vector(dim, ti.f32)):
    return (0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2)

@ti.func
def principal_kirchhoff_from_F(F: ti.types.matrix(dim, dim, ti.f32), mu: ti.f32, lam: ti.f32) -> ti.types.matrix(dim, dim, ti.f32):
    U, Sig, V = ti.svd(F)
    s0, s1 = Sig[0, 0], Sig[1, 1]
    e0 = ti.log(ti.max(s0, 1e-10))
    e1 = ti.log(ti.max(s1, 1e-10))
    tr_eps = e0 + e1
    t0 = 2.0 * mu * e0 + lam * tr_eps
    t1 = 2.0 * mu * e1 + lam * tr_eps
    u0 = ti.Vector([U[0, 0], U[1, 0]], dt=ti.f32)
    u1 = ti.Vector([U[0, 1], U[1, 1]], dt=ti.f32)
    Tau = t0 * (u0.outer_product(u0)) + t1 * (u1.outer_product(u1))
    return Tau

@ti.func
def visco_correct_FN(FN_tr: ti.types.matrix(dim, dim, ti.f32)) -> ti.types.matrix(dim, dim, ti.f32):
    U, Sig, V = ti.svd(FN_tr)
    s0, s1 = Sig[0, 0], Sig[1, 1]
    e0_tr = ti.log(ti.max(s0, 1e-10))
    e1_tr = ti.log(ti.max(s1, 1e-10))
    d = 2.0
    alpha = 2.0 * mu_N[None] / ti.max(nu_d[None], 1e-10)
    beta  = (2.0 * (2.0 * mu_N[None] + lambda_N[None]) / (9.0 * ti.max(nu_v[None], 1e-10))) - (2.0 * mu_N[None]) / (ti.max(nu_d[None], 1e-10) * d)
    A = 1.0 / (1.0 + dt * alpha)
    B = (dt * beta) / (1.0 + dt * (alpha + d * beta))
    tr_eps_tr = e0_tr + e1_tr
    e0_new = A * (e0_tr - B * tr_eps_tr)
    e1_new = A * (e1_tr - B * tr_eps_tr)
    s0_new, s1_new = ti.exp(e0_new), ti.exp(e1_new)
    Sig_new = ti.Matrix([[s0_new, 0.0], [0.0, s1_new]], dt=ti.f32)
    return U @ Sig_new @ V.transpose()

# --- SLS mapping + presets ----------------------------------------------------
def set_sls(E0: float, Einf: float, nu: float, tau_dev: float, tau_vol: float):
    E2 = max(E0 - Einf, 1e-6)
    muN = E2 / (2.0 * (1.0 + nu))
    KN  = E2 / (3.0 * (1.0 - 2.0 * nu))
    eta_dev = 2.0 * muN * tau_dev
    zeta    = KN * tau_vol
    set_material_params(E_eq=float(Einf), nu_eq=float(nu),
                        E_ne=float(E2),   nu_ne=float(nu),
                        nu_dev=float(eta_dev), nu_vol=float(zeta))

TISSUE_PRESETS = {
    'fat_soft':   dict(E0=15e3,  Einf=5e3,  nu=0.48, tau_dev=0.6,  tau_vol=0.6),
    'muscle_rel': dict(E0=60e3,  Einf=20e3, nu=0.48, tau_dev=0.12, tau_vol=0.12),
    'liver_gen':  dict(E0=12e3,  Einf=4e3,  nu=0.49, tau_dev=0.25, tau_vol=0.25),
    'fascia':     dict(E0=120e3, Einf=40e3, nu=0.47, tau_dev=0.08, tau_vol=0.08),
}

# ----------------------------------------------------------------------------
# Initialization (soft rectangle)
# ----------------------------------------------------------------------------
# Soft rectangle dimensions
SOFT_W, SOFT_H = 0.40, 0.30  # width, height
soft_half_w    = SOFT_W * 0.5
soft_height    = SOFT_H      # placed on floor in [0, H]

time_t = ti.field(dtype=ti.f32, shape=())
base_x = 0.50
base_y = ti.field(dtype=ti.f32, shape=())
y0     = SOFT_W+BOX_HX/2 + 0.05
A_osc  = 0.035
omega  = 2.0
L_arm  = 0.12
theta0 = 0.0

# Logging
time_hist, indent_hist, force_hist = [], [], []
indent_ref_y = None  # for square: y_top0 (just touching when y_bottom == y_top0)

@ti.kernel
def init_scene():
    # Soft body: rectangle centered at (soft_center_x, floor + H/2)
    soft_center_x = 0.5
    for p in range(n_particles):
        rx = (ti.random() * 2.0 - 1.0) * soft_half_w
        ry = ti.random() * soft_height  # from floor up to H
        x[p]  = ti.Vector([soft_center_x + rx, floor_level + ry])


        # u  = ti.random()
        # r  = soft_height * ti.sqrt(u)
        # th = ti.random() * PI
        # x[p] = ti.Vector([soft_center_x + r * ti.cos(th),
        #                   floor_level    + r * ti.sin(th)])
        
        v[p]  = ti.Vector.zero(ti.f32, dim)
        C[p]  = ti.Matrix.zero(ti.f32, dim, dim)
        FE[p] = ti.Matrix.identity(ti.f32, dim)
        FN[p] = ti.Matrix.identity(ti.f32, dim)

    # Collider center/vel
    base_y[None] = y0
    time_t[None] = 0.0
    base = ti.Vector([base_x, base_y[None]])
    tip  = base + ti.Vector([ti.sin(theta0), -ti.cos(theta0)]) * L_arm
    roller_center[0]   = tip
    roller_velocity[0] = ti.Vector.zero(ti.f32, dim)
    contact_force[0]   = ti.Vector.zero(ti.f32, dim)

    # Square half-sizes
    square_half[0] = ti.Vector([BOX_HX, BOX_HY], dt=ti.f32)

# ----------------------------------------------------------------------------
# P2G
# ----------------------------------------------------------------------------
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

        tau_E = principal_kirchhoff_from_F(FE[p], mu_E[None], lambda_E[None])
        tau_N = principal_kirchhoff_from_F(FN[p], mu_N[None], lambda_N[None])
        tau   = tau_E + tau_N
        F_total = FE[p] @ FN[p]
        FinvT = F_total.inverse().transpose()
        P = tau @ FinvT

        stress = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * P
        affine = stress + p_mass * C[p]

        for i, j in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([i, j])
            dpos = (offs.cast(ti.f32) - fx) * dx
            wt   = w[i].x * w[j].y
            momentum = p_mass * v[p] + affine @ dpos
            grid_v[base + offs] += wt * momentum
            grid_m[base + offs] += wt * p_mass

# ----------------------------------------------------------------------------
# Grid update + contact (square AABB)
# ----------------------------------------------------------------------------
@ti.kernel
def grid_update_and_contact():
    contact_force[0] = ti.Vector.zero(ti.f32, dim)

    for I in ti.grouped(grid_m):
        m = grid_m[I]
        if m > 0:
            v_old = grid_v[I] / m
            v_new = v_old + dt * ti.Vector([0.0, -9.8])
            pos   = I.cast(ti.f32) * dx

            if USE_SQUARE:
                rel = pos - roller_center[0]
                hx, hy = square_half[0].x, square_half[0].y
                inside = (ti.abs(rel.x) < hx) and (ti.abs(rel.y) < hy)
                if inside:
                    # choose face with smallest remaining distance to exit
                    dxp = hx - ti.abs(rel.x)
                    dyp = hy - ti.abs(rel.y)
                    # normal points outward (+/- x or +/- y)
                    n = ti.Vector([0.0, 0.0])
                    if dxp < dyp:
                        # x-face
                        sx = 1.0
                        if rel.x < 0: sx = -1.0
                        n = ti.Vector([sx, 0.0])
                    else:
                        sy = 1.0
                        if rel.y < 0: sy = -1.0
                        n = ti.Vector([0.0, sy])
                    rv = roller_velocity[0]
                    v_t = v_old - n * (n.dot(v_old))
                    v_n = n * n.dot(rv)
                    v_new = v_t + v_n
                    dv = v_new - v_old
                    imp = m * dv / dt
                    contact_force[0] += imp
            else:
                # (fallback) circular collider — not used when USE_SQUARE is True
                pass

            # Floor / walls / ceiling clamps
            if pos.y < floor_level + dx:
                if v_new.y < 0: v_new.y = 0
                v_new.x = 0
            if pos.x < dx or pos.x > 1.0 - dx:
                v_new.x = 0
            if pos.y > 1.0 - dx:
                v_new.y = 0

            grid_v[I] = v_new * m

# ----------------------------------------------------------------------------
# G2P + strain update
# ----------------------------------------------------------------------------
@ti.kernel
def g2p_and_strain_update():
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
            gv     = grid_v[base + offs] / (grid_m[base + offs] + 1e-12)
            new_v += wt * gv
            new_C += 4.0 * inv_dx * wt * gv.outer_product(dpos)

        v[p] = new_v
        C[p] = new_C
        x[p] += dt * new_v

        if x[p].y < floor_level:
            x[p].y = floor_level
            v[p]   = ti.Vector.zero(ti.f32, dim)
        if x[p].x < dx:
            x[p].x = dx; v[p].x = 0
        if x[p].x > 1.0 - dx:
            x[p].x = 1.0 - dx; v[p].x = 0
        if x[p].y > 1.0 - dx:
            x[p].y = 1.0 - dx; v[p].y = 0

        I2 = ti.Matrix.identity(ti.f32, dim)
        F_pred = (I2 + dt * new_C)
        FE_tr = F_pred @ FE[p]
        FN_tr = F_pred @ FN[p]
        FE[p] = FE_tr
        FN[p] = visco_correct_FN(FN_tr)

# ----------------------------------------------------------------------------
# Collider kinematics (vertical oscillation via base point)
# ----------------------------------------------------------------------------
time_scale = 5.0

def update_collider():
    t = float(time_t[None])
    time_t[None] = t + dt * time_scale
    by = y0 + A_osc * np.cos(omega * time_t[None])
    base = np.array([base_x, by], dtype=np.float32)

    ee_o = roller_center[0].to_numpy()
    ee_n = base + np.array([np.sin(theta0), -np.cos(theta0)], dtype=np.float32) * L_arm
    rv   = (ee_n - ee_o) / dt

    roller_center[0]   = ee_n.tolist()
    roller_velocity[0] = rv.tolist()
    base_y[None] = by

# ----------------------------------------------------------------------------
# Optional RPIC damping
# ----------------------------------------------------------------------------
rpic_iters = 0

@ti.kernel
def rpic_damping_step(shrink: ti.f32):
    for p in range(n_particles):
        S = 0.5 * (C[p] + C[p].transpose())
        R = 0.5 * (C[p] - C[p].transpose())
        C[p] = R + shrink * S

# ----------------------------------------------------------------------------
# Logging & plots (indentation uses square bottom face)
# ----------------------------------------------------------------------------
def compute_indent_reference():
    # initial top surface y
    x_np = x.to_numpy()
    y_top0 = np.max(x_np[:, 1])
    if USE_SQUARE:
        return float(y_top0)  # just touching when (center_y - BOX_HY) == y_top0
    else:
        # legacy (not used when USE_SQUARE True)
        return float(y_top0 + 0.0)

def log_signals():
    t  = float(time_t[None])
    c  = roller_center[0].to_numpy()
    cf = float(np.linalg.norm(contact_force[0].to_numpy()))
    if USE_SQUARE:
        y_bottom = float(c[1]) - BOX_HY
        d = max(0.0, indent_ref_y - y_bottom)
    else:
        d = 0.0
    time_hist.append(t)
    indent_hist.append(d)
    force_hist.append(cf)

def plot_signals():
    """Plot indentation(t) and |F_contact|(t) on two separate figures."""
    t = np.asarray(time_hist, dtype=float)
    d = np.asarray(indent_hist, dtype=float)
    f = np.asarray(force_hist, dtype=float)

    plt.figure()
    plt.plot(d, f)
    plt.xlabel("indentation [m]")
    plt.ylabel("|contact force| [N]")
    plt.title("Indentation vs Contact Force")
    plt.tight_layout()

    plt.show()
# ----------------------------------------------------------------------------
# GUI helpers
# ----------------------------------------------------------------------------
def draw_square(gui, center, half, color=0xFFFFFF, radius=2):
    cx, cy = float(center[0]), float(center[1])
    hx, hy = float(half[0]), float(half[1])
    p1 = (cx - hx, cy - hy)
    p2 = (cx + hx, cy - hy)
    p3 = (cx + hx, cy + hy)
    p4 = (cx - hx, cy + hy)
    gui.line(begin=p1, end=p2, radius=radius, color=color)
    gui.line(begin=p2, end=p3, radius=radius, color=color)
    gui.line(begin=p3, end=p4, radius=radius, color=color)
    gui.line(begin=p4, end=p1, radius=radius, color=color)

# ----------------------------------------------------------------------------
# Main loops
# ----------------------------------------------------------------------------
def run_sls(frames=300, preset: str = 'muscle_rel'):
    global indent_ref_y
    gui = ti.GUI('Viscoelastic Tissue (Square Indenter, Rect Soft Body)', res=(1024, 1024))
    p = TISSUE_PRESETS.get(preset, TISSUE_PRESETS['muscle_rel'])
    set_sls(**p)
    init_scene()
    indent_ref_y = compute_indent_reference()

    frames_left = frames
    while gui.running and frames_left > 0:
        for _ in range(15):
            p2g()
            grid_update_and_contact()
            g2p_and_strain_update()
            if rpic_iters > 0:
                for _k in range(rpic_iters):
                    rpic_damping_step(0.0)
            update_collider()

        log_signals()

        xp = x.to_numpy()
        gui.circles(xp, radius=1.8, color=0xDD6666)

        c  = roller_center[0].to_numpy()
        cf = np.linalg.norm(contact_force[0].to_numpy())
        draw_square(gui, c, (BOX_HX, BOX_HY), color=0xFFFFFF, radius=2)

        d_now = indent_hist[-1]
        gui.text(f'Time: {float(time_t[None]):.3f}s', pos=(0.02, 0.95), color=0xFFFFFF)
        gui.text(f'|F_contact|: {cf:.2f} N',         pos=(0.02, 0.92), color=0xFFFFFF)
        gui.text(f'Indent: {d_now*1e3:.2f} mm',      pos=(0.02, 0.89), color=0xFFFFFF)

        gui.show()
        frames_left -= 1

    plot_signals()

if __name__ == '__main__':
    run_sls(frames=999999, preset='muscle_rel')
