import taichi as ti
import numpy as np

from vetris.engine.massagers.massager import massager

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
        self.dt = 2e-4

        # Material (Neo-Hookean)
        self.p_rho     = 1.0
        self.p_vol     = (self.dx * 0.5) ** 2
        self.p_mass    = self.p_rho * self.p_vol
        self.E         = self.cfg.youngs_modulus
        self.nu        = self.cfg.poisson_ratio
        self.mu_0      = self.E / (2 * (1 + self.nu))
        self.lambda_0  = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        # Floor & domain
        self.floor_level    = 0.0
        self.floor_friction = 0.4  # param kept; current logic clamps velocities



        self.massager = massager(cfg)
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

            stress = (-self.dt * self.p_vol * 4.0 * self.inv_dx * self.inv_dx) * self.neo_hookean_stress(self.F[p])
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
                gv     = self.grid_v[base + offs] / self.grid_m[base + offs]
                new_v += wt * gv
                new_C += 4.0 * self.inv_dx * wt * gv.outer_product(dpos)

            # Update particle state
            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * new_v

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
        self.massager.step()
        self.compute_strain_from_F()


    @property    
    def extract_state(self):
        # Extract state for logging: total volume, mean position, etc.
        positions = self.x.to_numpy()[:self.particle_count]
        velocities = self.v.to_numpy()[:self.particle_count]
        masses = self.particle_mass.to_numpy()[:self.particle_count]
        volumes = self.particle_volume.to_numpy()[:self.particle_count]
        total_volume = np.sum(volumes)
        mean_position = np.mean(positions, axis=0) if positions.shape[0] > 0 else np.zeros(self.dim)
        mean_velocity = np.mean(velocities, axis=0) if velocities.shape[0] > 0 else np.zeros(self.dim)

        return {
            "step": self.step,
            "time": self.time,
            "total_volume": total_volume,
            "mean_position": mean_position,
            "mean_velocity": mean_velocity,
            "particle_count": self.particle_count
        }
        
        