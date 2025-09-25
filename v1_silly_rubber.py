import taichi as ti
import numpy as np
from math import pi as PI
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Taichi init
# ----------------------------------------------------------------------------
ti.init(arch=ti.gpu)

# ----------------------------------------------------------------------------
# Simulation parameters (keep your defaults; tweak as needed)
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

# Rigid roller (contact)
roller_radius   = 0.025
roller_center   = ti.Vector.field(dim, dtype=ti.f32, shape=1)
roller_velocity = ti.Vector.field(dim, dtype=ti.f32, shape=1)
contact_force   = ti.Vector.field(dim, dtype=ti.f32, shape=1)

# ----------------------------------------------------------------------------
# MPM fields
# ----------------------------------------------------------------------------
x      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)  # position
v      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)  # velocity
C      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)  # APIC affine

# Finite strain viscoelasticity (Silly Rubber): F = F_E at equilibrium, and F_N for non-equilibrated part
FE     = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)  # equilibrated elastic part
FN     = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)  # non-equilibrated elastic part

# Mass / volume (constant here)
p_rho  = 1.0
p_vol  = (dx * 0.5) ** 2
p_mass = p_rho * p_vol

# Grid
grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

# ----------------------------------------------------------------------------
# Viscoelastic material parameters (Hencky strain + principal-space projection)
#   Total Kirchhoff stress: tau = tau_E(FE) + tau_N(FN)
#   tau_*(principal) = 2 mu_* * eps + lambda_* * tr(eps) * 1
#   Corrector for FN principal Hencky strain per Fang et al. (2019)
# ----------------------------------------------------------------------------
mu_E      = ti.field(dtype=ti.f32, shape=())  # equilibrated Lamé (shear)
lambda_E  = ti.field(dtype=ti.f32, shape=())
mu_N      = ti.field(dtype=ti.f32, shape=())  # non-eq Lamé (shear)
lambda_N  = ti.field(dtype=ti.f32, shape=())
nu_d      = ti.field(dtype=ti.f32, shape=())  # deviatoric viscosity
nu_v      = ti.field(dtype=ti.f32, shape=())  # volumetric viscosity


# Tissue-like presets (Pa, s). Adjust to match your target curves.
TISSUE_PRESETS = {
    'fat_soft':   dict(E0=15e3,  Einf=5e3,  nu=0.48, tau_dev=0.6,  tau_vol=0.6),
    'muscle_rel': dict(E0=60e3,  Einf=20e3, nu=0.48, tau_dev=0.12, tau_vol=0.12),
    'liver_gen':  dict(E0=12e3,  Einf=4e3,  nu=0.49, tau_dev=0.25, tau_vol=0.25),
    'fascia':     dict(E0=120e3, Einf=40e3, nu=0.47, tau_dev=0.08, tau_vol=0.08),
}

# ----------------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------------
half_radius   = 0.2
soft_center_x = 0.5

time_t   = ti.field(dtype=ti.f32, shape=())
base_x   = 0.50
base_y   = ti.field(dtype=ti.f32, shape=())
y0       = 0.33
A_osc    = 0.02
omega    = 2.0
L_arm    = 0.12
theta0   = 0.0

Time_period = 2 * PI / omega * 1
time_scale = 5.0  # lower driver speed to avoid violent contact impulses

# --- NEW: logging buffers (Python lists) -------------------------------------
time_hist   = []
indent_hist = []
force_hist  = []

# Reference “just-touching” height (set after init_scene)
indent_ref_y = None  # y_top0 + roller_radius

@ti.kernel
def set_material_params(E_eq: ti.f32, nu_eq: ti.f32,
                        E_ne: ti.f32, nu_ne: ti.f32,
                        nu_dev: ti.f32, nu_vol: ti.f32):
    # Convert (E, nu) -> (mu, lambda)
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
def mat_logdiag(sig: ti.types.vector(dim, ti.f32)) -> ti.types.vector(dim, ti.f32):
    # principal Hencky strain eps = log(sigma)
    ret = ti.Vector.zero(ti.f32, dim)
    for k in ti.static(range(dim)):
        ret[k] = ti.log(ti.max(sig[k], 1e-10))
    return ret

@ti.func
def mat_expdiag(eps: ti.types.vector(dim, ti.f32)) -> ti.types.vector(dim, ti.f32):
    # inverse: sigma = exp(eps)
    ret = ti.Vector.zero(ti.f32, dim)
    for k in ti.static(range(dim)):
        ret[k] = ti.exp(eps[k])
    return ret

@ti.func
def principal_kirchhoff_from_F(F: ti.types.matrix(dim, dim, ti.f32), mu: ti.f32, lam: ti.f32) -> ti.types.matrix(dim, dim, ti.f32):
    # 2D explicit version to avoid dynamic matrix indexing issues
    U, Sig, V = ti.svd(F)
    s0 = Sig[0, 0]
    s1 = Sig[1, 1]
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
    # 2D explicit principal-space corrector for FN
    U, Sig, V = ti.svd(FN_tr)
    s0 = Sig[0, 0]
    s1 = Sig[1, 1]
    e0_tr = ti.log(ti.max(s0, 1e-10))
    e1_tr = ti.log(ti.max(s1, 1e-10))

    # coefficients
    d = 2.0
    alpha = 2.0 * mu_N[None] / ti.max(nu_d[None], 1e-10)
    beta  = (2.0 * (2.0 * mu_N[None] + lambda_N[None]) / (9.0 * ti.max(nu_v[None], 1e-10))) - (2.0 * mu_N[None]) / (ti.max(nu_d[None], 1e-10) * d)
    A = 1.0 / (1.0 + dt * alpha)
    B = (dt * beta) / (1.0 + dt * (alpha + d * beta))

    tr_eps_tr = e0_tr + e1_tr
    e0_new = A * (e0_tr - B * tr_eps_tr)
    e1_new = A * (e1_tr - B * tr_eps_tr)

    s0_new = ti.exp(e0_new)
    s1_new = ti.exp(e1_new)

    Sig_new = ti.Matrix([[s0_new, 0.0], [0.0, s1_new]], dt=ti.f32)
    FN_new = U @ Sig_new @ V.transpose()
    return FN_new

# ----------------------------------------------------------------------------
# SLS-style tissue convenience (maps desired tissue params to Silly Rubber fields)
# ----------------------------------------------------------------------------
# Standard Linear Solid (Zener): E0 (instantaneous), Einf (equilibrium), tau_dev, tau_vol, nu.
# Mapping used: E2 = E0 - Einf (Maxwell spring). With target time-constants
#   tau_dev ≈ η_dev / (2 μ_N) and tau_vol ≈ ζ / K_N, where
#   μ_N = E2 / [2(1+nu)],   K_N = E2 / [3(1-2nu)].
# Then η_dev = 2 μ_N tau_dev and ζ = K_N tau_vol.

def set_sls(E0: float, Einf: float, nu: float, tau_dev: float, tau_vol: float):
    E2 = max(E0 - Einf, 1e-6)
    muN = E2 / (2.0 * (1.0 + nu))
    KN  = E2 / (3.0 * (1.0 - 2.0 * nu))
    eta_dev = 2.0 * muN * tau_dev
    zeta    = KN * tau_vol
    set_material_params(E_eq=float(Einf), nu_eq=float(nu),
                        E_ne=float(E2),   nu_ne=float(nu),
                        nu_dev=float(eta_dev), nu_vol=float(zeta))




@ti.kernel
def init_scene():
    for p in range(n_particles):
        u  = ti.random()
        r  = half_radius * ti.sqrt(u)
        th = ti.random() * PI
        x[p] = ti.Vector([soft_center_x + r * ti.cos(th),
                          floor_level    + r * ti.sin(th)])
        v[p] = ti.Vector.zero(ti.f32, dim)
        C[p] = ti.Matrix.zero(ti.f32, dim, dim)
        FE[p] = ti.Matrix.identity(ti.f32, dim)
        FN[p] = ti.Matrix.identity(ti.f32, dim)

    base_y[None] = y0
    time_t[None] = 0.0
    base = ti.Vector([base_x, base_y[None]])
    tip  = base + ti.Vector([ti.sin(theta0), -ti.cos(theta0)]) * L_arm
    roller_center[0]   = tip
    roller_velocity[0] = ti.Vector.zero(ti.f32, dim)
    contact_force[0]   = ti.Vector.zero(ti.f32, dim)

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

        # Total Kirchhoff stress tau = tau_E(FE) + tau_N(FN)
        tau_E = principal_kirchhoff_from_F(FE[p], mu_E[None], lambda_E[None])
        tau_N = principal_kirchhoff_from_F(FN[p], mu_N[None], lambda_N[None])
        tau   = tau_E + tau_N
        # First Piola: P = tau * F^{-T}; here use F_total ~ FE (since plasticity off). For robustness, use FE.
        # Use total F for mapping: P = tau * F_total^{-T}
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
# Grid update + contact
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

            # Roller contact (normal-only reflection toward roller velocity)
            rel = pos - roller_center[0]
            r   = rel.norm()
            if r < roller_radius:
                n = rel / (r + 1e-8)
                rv = roller_velocity[0]
                v_t = v_old - n * (n.dot(v_old))
                v_n = n * n.dot(rv)  # impose roller normal component
                v_new = v_t + v_n
                dv = v_new - v_old
                imp = m * dv / dt
                contact_force[0] += imp

            # Floor clamp
            if pos.y < floor_level + dx:
                if v_new.y < 0: v_new.y = 0
                v_new.x = 0
            # Walls
            if pos.x < dx or pos.x > 1.0 - dx:
                v_new.x = 0
            # Ceiling
            if pos.y > 1.0 - dx:
                v_new.y = 0

            grid_v[I] = v_new * m

# ----------------------------------------------------------------------------
# G2P + Strain update (predictor-corrector for FE, FN)
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

        # Update particle kinematics
        v[p] = new_v
        C[p] = new_C
        x[p] += dt * new_v

        # Boundary clamps (particles)
        if x[p].y < floor_level:
            x[p].y = floor_level
            v[p]   = ti.Vector.zero(ti.f32, dim)
        if x[p].x < dx:
            x[p].x = dx; v[p].x = 0
        if x[p].x > 1.0 - dx:
            x[p].x = 1.0 - dx; v[p].x = 0
        if x[p].y > 1.0 - dx:
            x[p].y = 1.0 - dx; v[p].y = 0

        # Predictor: F_tr_* = (I + dt * grad v) * F_*^n  (Eq. 4)
        I2 = ti.Matrix.identity(ti.f32, dim)
        F_pred = (I2 + dt * new_C)
        FE_tr = F_pred @ FE[p]
        FN_tr = F_pred @ FN[p]

        # Corrector: FE no plasticity -> FE_{n+1} = FE_tr
        FE[p] = FE_tr
        # Corrector: project FN_tr in principal space per Eq. (8)
        FN[p] = visco_correct_FN(FN_tr)

# ----------------------------------------------------------------------------
# Roller kinematics (numpy-host) — vertical base oscillation
# ----------------------------------------------------------------------------


def update_roller():
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
# Optional RPIC transfer-based damping (unconditional stable)
# ----------------------------------------------------------------------------
# For simplicity, we expose a knob rpic_iters; if >0, we apply a lightweight
# APIC->RPIC style momentum smoothing by shrinking the symmetric part of C.

rpic_iters = 0

@ti.kernel
def rpic_damping_step(shrink: ti.f32):
    # shrink symmetric part of affine C -> emulates viscosity-like damping
    for p in range(n_particles):
        S = 0.5 * (C[p] + C[p].transpose())
        R = 0.5 * (C[p] - C[p].transpose())
        C[p] = R + shrink * S


# ----------------------------------------------------------------------------
# Utilities: indent reference and logging
# ----------------------------------------------------------------------------
def compute_indent_reference():
    """Compute initial top surface y and return y_top0 + roller_radius."""
    x_np = x.to_numpy()
    y_top0 = np.max(x_np[:, 1])
    return float(y_top0 + roller_radius)

def log_signals():
    """Append time, indentation, and force magnitude for current frame."""
    t  = float(time_t[None])
    ee = roller_center[0].to_numpy()
    cf = float(np.linalg.norm(contact_force[0].to_numpy()))
    d  = max(0.0, indent_ref_y - float(ee[1]))
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
# Simple GUI loop (colors omitted here; user can plug their palette/renderer)
# ----------------------------------------------------------------------------

# def run(frames=100):
#     gui = ti.GUI('Silly Rubber (explicit viscoelastic MPM)', res=(1024, 1024))

#     # Example parameters close to your old elastic values
#     set_material_params(E_eq=5.65e4, nu_eq=0.185,
#                         E_ne=5.65e4, nu_ne=0.185,
#                         nu_dev=1.0e1,  # increase for stronger rate resistance
#                         nu_vol=1.0e1)
#     init_scene()
#     indent_ref_y = compute_indent_reference()

#     while gui.running and time_t > Time_period:
#         for _ in range(15):
#             p2g()
#             grid_update_and_contact()
#             g2p_and_strain_update()
#             if rpic_iters > 0:
#                 for _k in range(rpic_iters):
#                     rpic_damping_step(0.0)  # 0 removes symmetric part fully (RPIC); try 0.25..0.75 for mild damping
#             update_roller()

#         # draw particles
#         xp = x.to_numpy()
#         gui.circles(xp, radius=1.8, color=0xDD6666)

#         # draw roller halo scaled by |contact_force|
#         ee = roller_center[0].to_numpy()
#         cf = np.linalg.norm(contact_force[0].to_numpy())
#         halo_r = int(np.clip(6 + 0.004 * cf, 6, 22))
#         gui.circle(ee, radius=halo_r, color=0xFFFFFF)

#         gui.text(f'Time: {float(time_t[None]):.3f}s', pos=(0.02, 0.95), color=0xFFFFFF)
#         gui.text(f'|F_contact|: {cf:.2f} N', pos=(0.02, 0.92), color=0xFFFFFF)
#         gui.show()



def run_sls(preset: str = 'muscle_rel'):
    """Run with a tissue-like SLS preset (near-incompressible viscoelastic).
    Choose from TISSUE_PRESETS or pass your own params via set_sls(...).
    """
    global indent_ref_y

    gui = ti.GUI('Viscoelastic Tissue (SLS on Silly Rubber)', res=(1024, 1024))
    p = TISSUE_PRESETS.get(preset, TISSUE_PRESETS['muscle_rel'])
    set_sls(**p)
    init_scene()
    indent_ref_y = compute_indent_reference()


    while gui.running and time_t[None] < Time_period:
        for _ in range(15):
            p2g()
            grid_update_and_contact()
            g2p_and_strain_update()
            if rpic_iters > 0:
                for _k in range(rpic_iters):
                    rpic_damping_step(0.0)
            update_roller()

        log_signals()

        xp = x.to_numpy()
        gui.circles(xp, radius=1.8, color=0xDD6666)
        ee = roller_center[0].to_numpy()
        cf = np.linalg.norm(contact_force[0].to_numpy())
        gui.circle(ee, radius=roller_radius*1024, color=0xFFFFFF)

        d_now = indent_hist[-1]
        gui.text(f'Time: {float(time_t[None]):.3f}s', pos=(0.02, 0.95), color=0xFFFFFF)
        gui.text(f'|F_contact|: {cf:.2f} N', pos=(0.02, 0.92), color=0xFFFFFF)
        gui.text(f'Indent: {d_now*1e3:.2f} mm',      pos=(0.02, 0.89), color=0xFFFFFF)


        gui.show()

    plot_signals()

if __name__ == '__main__':
    # Use SLS presets for viscoelastic soft tissue behavior
    run_sls(preset='muscle_rel')
