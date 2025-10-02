import taichi as ti
import numpy as np

from vetris.engine.massagers.massager import MassagerWrapper

PI = 3.141592653589793
@ti.data_oriented
class mpmengine:
    def __init__(self, cfg):

        # simulation parameters
        self.cfg = cfg.engine.mpm
        self.dim = 2
        self.n_particles = 20000
        self.n_grid = 128
        self.dx = 1.0 / self.n_grid
        self.inv_dx = float(self.n_grid)
        # self.dt = 2e-4

        self.dt = cfg.engine.dt

        # Material (Neo-Hookean)
        self.p_rho     = 1.0
        self.p_vol     = (self.dx * 0.5) ** 2
        self.p_mass    = self.p_rho * self.p_vol
        self.E         = self.cfg.youngs_modulus
        self.nu        = self.cfg.poisson_ratio
        self.mu_0      = self.E / (2 * (1 + self.nu))
        self.lambda_0  = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        # --- Viscosity & rate-dependence (Kelvin–Voigt + simple power-law) ---
        self.eta_shear  = float(getattr(self.cfg, "shear_viscosity",5.0))   # Pa·s-ish in code units
        self.eta_bulk   = float(getattr(self.cfg, "bulk_viscosity",  50.0))    # bulk (ζ) for volumetric damping

        # Optional nonlinear shear-thickening/thinning:  eta_eff = eta0 * (1 + k * |D|^n)
        self.rate_k     = float(getattr(self.cfg, "rate_k", 0.0))             # 0 disables nonlinearity
        self.rate_n     = float(getattr(self.cfg, "rate_n", 1.0))


        # Floor & domain
        self.floor_level    = 0.0
        self.floor_friction = 0.4  # param kept; current logic clamps velocities



        self.massager = MassagerWrapper(cfg)
        self.massager_type = self.massager.type


        # MPM fields
        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.n_particles)
        self.J = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.grid_v = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))

        # For logging / analysis
        self.green_E = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.n_particles)  # 2x2 Green strain
        self.principal_stretch = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)  # [λ1, λ2]
        self.principal_log_strain = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)  # [log λ1, log λ2]
        self.I1_C = ti.field(dtype=ti.f32, shape=self.n_particles)   # tr(C) invariant, C = F^T F
        self.eqv_strain_dev = ti.field(dtype=ti.f32, shape=self.n_particles)  # scalar dev. Green strain magnitude

        self.disp_mag = ti.field(dtype=ti.f32, shape=self.n_particles)  # |Δx| per particle per substep
        self.max_disp = ti.field(dtype=ti.f32, shape=())                # global max |Δx|
        self.max_eqv_strain = ti.field(dtype=ti.f32, shape=())  

        # --- add in __init__ after diagnostics fields ---
        self.disp_vec = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)

        self.indentation = ti.field(dtype=ti.f32, shape=())               # δ (m)
        self.max_eqv_strain_contact = ti.field(dtype=ti.f32, shape=())     # max eqv strain near contact
        self.max_disp_contact = ti.field(dtype=ti.f32, shape=())           # max disp along contact normal
        self.contact_pad = 2.0 * self.dx                                   # capture a thin band around roller


        # ----- Stability monitoring (host-visible) -----
        self.unstable       = ti.field(dtype=ti.i32, shape=())
        self.nan_flag       = ti.field(dtype=ti.i32, shape=())
        self.inf_flag       = ti.field(dtype=ti.i32, shape=())
        self.min_J          = ti.field(dtype=ti.f32, shape=())
        self.max_J          = ti.field(dtype=ti.f32, shape=())
        self.max_speed      = ti.field(dtype=ti.f32, shape=())

        # thresholds (tweak if needed; you can also plumb these from cfg)
        self.J_min_thresh   = 1e-6     # inverted/near-singular if below
        self.J_max_thresh   = 10.0     # unphysically large volume change
        self.vmax_thresh    = 250.0    # max allowed particle speed (code units)
        self.domain_lo      = -1e-3    # small tolerance below domain
        self.domain_hi      = 1.0 + 1e-3



        # Soft body initial placement
        self.half_radius = 0.2
        self.soft_center_x = 0.5

        # initialize particles and rollers
        self.init_mpm()
        self.massager.initialize()

    @ti.func
    def neo_hookean_stress(self,F_i):
        J_det = F_i.determinant()
        FinvT = F_i.inverse().transpose()
        return self.mu_0 * (F_i - FinvT) + self.lambda_0 * ti.log(J_det) * FinvT


    @ti.func
    def _eta_eff(self, D_norm: ti.f32) -> ti.f32:
        # power-law multiplier; set rate_k=0 to disable
        return ti.cast(self.eta_shear, ti.f32) * (1.0 + ti.cast(self.rate_k, ti.f32) * ti.pow(ti.max(D_norm, 0.0), ti.cast(self.rate_n, ti.f32)))

    @ti.func
    def viscous_PK1(self, F_i: ti.template(), C_i: ti.template()):
        # Objective Newtonian: tau = 2*eta*D + zeta*tr(D) I ;   P = J * tau * F^{-T}
        D = 0.5 * (C_i + C_i.transpose())
        D_norm = ti.sqrt(2.0*(D[0,0]*D[0,0] + D[1,1]*D[1,1]) + 4.0*(D[0,1]*D[0,1]))
        D_norm = ti.min(D_norm, 5e3)

        eta  = self._eta_eff(D_norm)
        zeta = ti.cast(self.eta_bulk, ti.f32)
        I    = ti.Matrix.identity(ti.f32, self.dim)
        tau  = 2.0 * eta * D + zeta * (D[0,0] + D[1,1]) * I

        J = F_i.determinant()
        P = ti.Matrix.zero(ti.f32, self.dim, self.dim)  # default

        if J > 1e-3:
            FinvT = F_i.inverse().transpose()
            P = J * (tau @ FinvT)
        return P

        
        

    @ti.func
    def quadratic_weights(self, fx):
        # MPM quadratic B-spline weights in 2D; returns three vectors
        return (
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        )
    
    def reset(self):
        pass

        
    # ──────────────────────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────────────────────
    @ti.kernel
    def init_mpm(self):
        
        self.unstable[None]  = 0

        for p in range(self.n_particles):
            # Disk sampling
            u     = ti.random()
            r     = self.half_radius * ti.sqrt(u)
            theta = ti.random() * PI
            self.x[p]  = ti.Vector([self.soft_center_x + r * ti.cos(theta),
                            self.floor_level    + r * ti.sin(theta)])
            self.v[p] = ti.Vector.zero(ti.f32, self.dim)
            self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
            self.J[p] = 1.0
            self.C[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)

            self.disp_mag[p] = 0.0



    # ──────────────────────────────────────────────────────────────────────────────
    # P2G
    # ──────────────────────────────────────────────────────────────────────────────
    @ti.kernel
    def p2g(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
            self.grid_m[I] = 0.0

        for p in range(self.n_particles):
            Xp   = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx   = Xp - base.cast(ti.f32)

            w0, w1, w2 = self.quadratic_weights(fx)
            w = ti.static([w0, w1, w2])

            # stress = (-self.dt * self.p_vol * 4.0 * self.inv_dx * self.inv_dx) * self.neo_hookean_stress(self.F[p])


            P_elastic = self.neo_hookean_stress(self.F[p])
            P_visc    = self.viscous_PK1(self.F[p], self.C[p]) # uses last-updated ∇v proxy


            stress = (-self.dt * self.p_vol * 4.0 * self.inv_dx * self.inv_dx) * (P_elastic + P_visc)
            affine = stress + self.p_mass * self.C[p]


            affine = stress + self.p_mass * self.C[p]

            for i, j in ti.static(ti.ndrange(3, 3)):
                offs = ti.Vector([i, j])
                dpos = (offs.cast(ti.f32) - fx) * self.dx
                wt   = w[i].x * w[j].y
                momentum = self.p_mass * self.v[p] + affine @ dpos
                self.grid_v[base + offs] += wt * momentum
                self.grid_m[base + offs] += wt * self.p_mass

    # ──────────────────────────────────────────────────────────────────────────────
    # Grid updates + contact (normal-only with rollers, floor/walls clamp)
    # ──────────────────────────────────────────────────────────────────────────────
    @ti.kernel
    def apply_grid_forces_and_detect(self):

        self.massager.massager.reset_contact_forces()

        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0:
                v_old = self.grid_v[I] / m
                v_new = v_old + self.dt * ti.Vector([0.0, -9.8])  # gravity
                pos   = I.cast(ti.f32) * self.dx

                v_new = self.massager.massager.update_contact_info(m,pos,v_old,v_new)
                
                # Floor clamp
                if pos.y < self.floor_level + self.dx:
                    if v_new.y < 0:
                        v_new.y = 0
                    v_new.x = 0

                # Walls clamp
                if pos.x < self.dx:
                    v_new.x = 0
                if pos.x > 1.0 - self.dx:
                    v_new.x = 0

                self.grid_v[I] = v_new * m

    # ──────────────────────────────────────────────────────────────────────────────
    # G2P + boundary clamps
    # ──────────────────────────────────────────────────────────────────────────────
    @ti.kernel
    def g2p(self):
        for p in range(self.n_particles):
            Xp   = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx   = Xp - base.cast(ti.f32)

            w0, w1, w2 = self.quadratic_weights(fx)
            w = ti.static([w0, w1, w2])

            new_v = ti.Vector.zero(ti.f32, self.dim)
            new_C = ti.Matrix.zero(ti.f32, self.dim, self.dim)

            for i, j in ti.static(ti.ndrange(3, 3)):
                offs   = ti.Vector([i, j])
                dpos   = (offs.cast(ti.f32) - fx) * self.dx
                wt     = w[i].x * w[j].y

                den = self.grid_m[base + offs]
                gv  = ti.Vector.zero(ti.f32, self.dim)
                if den > 0:
                    gv = self.grid_v[base + offs] / den

                # gv     = self.grid_v[base + offs] / self.grid_m[base + offs]
                new_v += wt * gv
                new_C += 4.0 * self.inv_dx * wt * gv.outer_product(dpos)


            old_x = self.x[p]
            # Update particle state
            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * new_v

            self.disp_vec[p] = self.x[p] - old_x
            self.disp_mag[p] = (self.x[p] - old_x).norm()  # per-substep deformation magnitude



            # Floor & walls clamps
            if self.x[p].y < self.floor_level:
                self.x[p].y = self.floor_level
                self.v[p].y = 0
                self.v[p].x = 0

            if self.x[p].x < self.dx:
                self.x[p].x = self.dx
                self.v[p].x = 0
            if self.x[p].x > 1.0 - self.dx:
                self.x[p].x = 1.0 - self.dx
                self.v[p].x = 0
            if self.x[p].y > 1.0 - self.dx:
                self.x[p].y = 1.0 - self.dx
                self.v[p].y = 0

            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) + self.dt * new_C) @ self.F[p]
            self.J[p] = self.F[p].determinant()


    # ──────────────────────────────────────────────────────────────────────────────
    @ti.kernel
    def reduce_max_disp(self):
        self.max_disp[None] = 0.0
        for p in range(self.n_particles):
            ti.atomic_max(self.max_disp[None], self.disp_mag[p])

    @ti.kernel
    def reduce_max_eqv_strain(self):
        self.max_eqv_strain[None] = 0.0
        for p in range(self.n_particles):
            ti.atomic_max(self.max_eqv_strain[None], self.eqv_strain_dev[p])

    @ti.kernel
    def reduce_indentation(self, radius:ti.f32):
        # max penetration of “occupied” grid nodes inside the roller
        self.indentation[None] = 0.0
        center = self.massager.massager.roller_center[0]  # Taichi field
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                pos = I.cast(ti.f32) * self.dx
                pen = radius - (pos - center).norm()  # >0 means overlap
                if pen > 0:
                    ti.atomic_max(self.indentation[None], pen)

    @ti.kernel
    def reduce_max_eqv_strain_near_contact(self, radius: ti.f32, pad: ti.f32):
        self.max_eqv_strain_contact[None] = 0.0
        center = self.massager.massager.roller_center[0]
        r = radius + pad
        r2 = r * r
        for p in range(self.n_particles):
            d = self.x[p] - center
            if d.dot(d) <= r2:
                ti.atomic_max(self.max_eqv_strain_contact[None], self.eqv_strain_dev[p])

    @ti.kernel
    def reduce_max_disp_along_contact_normal(self, radius: ti.f32, pad: ti.f32):
        self.max_disp_contact[None] = 0.0
        center = self.massager.massager.roller_center[0]
        r = radius + pad
        r2 = r * r
        for p in range(self.n_particles):
            rel = self.x[p] - center
            if rel.dot(rel) <= r2:
                n = rel.normalized() if rel.norm() > 1e-12 else ti.Vector([0.0, 1.0])
                d_along = self.disp_vec[p].dot(n)
                ti.atomic_max(self.max_disp_contact[None], ti.max(0.0, d_along))  # inward only

    @ti.kernel
    def compute_strain_from_F(self):
        for p in range(self.n_particles):
            Fp = self.F[p]
            C  = Fp.transpose() @ Fp                       # right Cauchy–Green
            E  = 0.5 * (C - ti.Matrix.identity(ti.f32, self.dim))
            self.green_E[p] = E

            # invariants and principal stretches (2D, symmetric C)
            trC  = C[0, 0] + C[1, 1]
            detC = C.determinant()
            disc = ti.sqrt(ti.max(0.0, trC * trC - 4.0 * detC))
            lC1  = 0.5 * (trC + disc)                      # eigenvalues of C
            lC2  = 0.5 * (trC - disc)

            # principal stretches λi = sqrt(eigs(C))
            lam1 = ti.sqrt(ti.max(lC1, 1e-20))
            lam2 = ti.sqrt(ti.max(lC2, 1e-20))
            self.principal_stretch[p] = ti.Vector([lam1, lam2])
            self.principal_log_strain[p] = ti.Vector([ti.log(lam1), ti.log(lam2)])

            self.I1_C[p] = trC

            # Deviatoric Green–Lagrange strain magnitude (2D)
            # E_dev = E - (tr(E)/2) I, ||E_dev||_F used as a scalar measure
            trE   = E[0, 0] + E[1, 1]
            Em    = trE * 0.5
            E00d  = E[0, 0] - Em
            E11d  = E[1, 1] - Em
            E01   = E[0, 1]
            # Frobenius norm of deviatoric part in 2D
            eqv   = ti.sqrt(2.0 * (E00d * E00d + E11d * E11d) + 4.0 * (E01 * E01))
            self.eqv_strain_dev[p] = eqv


    def run(self):
        self.p2g()
        self.apply_grid_forces_and_detect()
        self.g2p()

        self.massager.step(1)
        self.compute_strain_from_F()

        # global (kept for compatibility)
        self.reduce_max_disp()
        self.reduce_max_eqv_strain()

        # Indentation
        r = float(self.massager.massager.roller_radius)
        self.reduce_indentation(r)
        self.reduce_max_eqv_strain_near_contact(r, self.contact_pad)
        self.reduce_max_disp_along_contact_normal(r, self.contact_pad)

        # === stability check & early exit ===
        if self.check_stability():
            # Best-effort: jump time to the end so callers’ while-loops exit
            try:
                # self.massager.massager.time_t = float(self.massager.massager.Time_period)
                # print("[unstable]", self.get_stability_stats())
                print("Unstable")
            except Exception:
                pass
            return False
        return True
    
    
    def get_deformation(self):
        # Extract deformation metrics for logging/analysis
        return {
        "max_disp": float(self.max_disp[None]),
        "max_eqv_strain_dev": float(self.max_eqv_strain[None]),
        "indentation": float(self.indentation[None])*1000,                        # NEW
        "max_eqv_strain_contact": float(self.max_eqv_strain_contact[None])*1000,  # NEW
        "max_disp_contact": float(self.max_disp_contact[None]),              # NEW
    }
        # return {
        #     "green_strain": self.green_E.to_numpy()[:self.n_particles],
        #     "principal_stretch": self.principal_stretch.to_numpy()[:self.n_particles],
        #     "principal_log_strain": self.principal_log_strain.to_numpy()[:self.n_particles],
        #     "I1_C": self.I1_C.to_numpy()[:self.n_particles],
        #     "eqv_strain_dev": self.eqv_strain_dev.to_numpy()[:self.n_particles],
        # }
    
    def get_state(self):
        # Extract state for logging: total volume, mean position, etc.
        positions = self.x.to_numpy()[:self.n_particles]
        velocities = self.v.to_numpy()[:self.n_particles]

        time = self.massager.massager.time_t
        contact_force = self.massager.massager.contact_force[0].to_numpy()
        deformation = self.get_deformation()


        return {
            "positions": positions,
            "velocities": velocities,
            "time": time,
            "contact_force": contact_force,
            "deformation": deformation
        }
    

        # ──────────────────────────────────────────────────────────────────────────────
    # Stability checks
    # ──────────────────────────────────────────────────────────────────────────────
    @ti.kernel
    def _reset_stability(self):
        self.unstable[None]  = 0
        self.nan_flag[None]  = 0
        self.inf_flag[None]  = 0
        self.min_J[None]     = 1e9
        self.max_J[None]     = -1e9
        self.max_speed[None] = 0.0

    @ti.kernel
    def _scan_particles_for_stability(self, lo: ti.f32, hi: ti.f32):
        for p in range(self.n_particles):
            x = self.x[p]
            v = self.v[p]
            Jp = self.J[p]

            # track extrema
            ti.atomic_min(self.min_J[None], Jp)
            ti.atomic_max(self.max_J[None], Jp)
            ti.atomic_max(self.max_speed[None], v.norm())

            # NaN checks: (val != val) is true only for NaN
            if not (x[0] == x[0]) or not (x[1] == x[1]) or not (v[0] == v[0]) or not (v[1] == v[1]) or not (Jp == Jp):
                self.nan_flag[None] = 1

            # crude Inf/overflow guard
            if ti.abs(x[0]) > 1e30 or ti.abs(x[1]) > 1e30 or ti.abs(v[0]) > 1e30 or ti.abs(v[1]) > 1e30 or ti.abs(Jp) > 1e30:
                self.inf_flag[None] = 1

            # escaped domain → mark unstable
            if (x[0] < lo) or (x[1] < lo) or (x[0] > hi) or (x[1] > hi):
                self.unstable[None] = 1

    @ti.kernel
    def _apply_stability_thresholds(self, jmin: ti.f32, jmax: ti.f32, vmax: ti.f32):
        if self.nan_flag[None] == 1 or self.inf_flag[None] == 1:
            self.unstable[None] = 1
        if self.min_J[None] < jmin or self.max_J[None] > jmax:
            self.unstable[None] = 1
        if self.max_speed[None] > vmax:
            self.unstable[None] = 1

    def check_stability(self) -> bool:
        """Run all stability checks and return True if unstable."""
        self._reset_stability()
        self._scan_particles_for_stability(self.domain_lo, self.domain_hi)
        self._apply_stability_thresholds(self.J_min_thresh, self.J_max_thresh, self.vmax_thresh)
        return bool(self.unstable[None])

    def get_stability_stats(self):
        return {
            "unstable":      int(self.unstable[None]),
            "nan_flag":      int(self.nan_flag[None]),
            "inf_flag":      int(self.inf_flag[None]),
            "min_J":         float(self.min_J[None]),
            "max_J":         float(self.max_J[None]),
            "max_speed":     float(self.max_speed[None]),
            "J_min_thresh":  float(self.J_min_thresh),
            "J_max_thresh":  float(self.J_max_thresh),
            "vmax_thresh":   float(self.vmax_thresh),
        }

        
        