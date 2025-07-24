import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import os


def setup_logging_dirs(base_dir="logs"):
    os.makedirs(os.path.join(base_dir, "positions"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "velocities"), exist_ok=True)
    return base_dir

@ti.data_oriented
class MPM_Simulator:
    def __init__(
        self,
        n_particles=20000,
        n_grid=128,
        dim=2,
        dt=2e-4,
        E=5e4,
        nu=0.2,
        rho=1.0,
        floor_friction=0.4,
        floor_level=0.0,
    ):
        # Simulation parameters
        self.dim = dim
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.dx = 1.0 / n_grid
        self.inv_dx = float(n_grid)
        self.dt = dt

        # Material parameters
        self.E = E
        self.nu = nu
        self.mu0 = E / (2 * (1 + nu))
        self.lambda0 = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.rho = rho
        self.p_vol = (self.dx * 0.5) ** 2
        self.p_mass = rho * self.p_vol

        # Floor
        self.floor_level = floor_level
        self.floor_friction = floor_friction

        # Arm parameters
        self.L1, self.L2 = 0.12, 0.10
        self.theta1 = np.array([0.0], dtype=np.float32)
        self.theta2 = np.array([0.0], dtype=np.float32)
        self.dtheta1 = np.zeros(1, dtype=np.float32)
        self.dtheta2 = np.zeros(1, dtype=np.float32)
        self.theta1_rest = self.theta1.copy()
        self.theta2_rest = self.theta2.copy()
        self.k1, self.k2 = 1.0, 1.0
        self.b1, self.b2 = 0.5, 0.5
        self.I1 = self.L1**2 / 12.0
        self.I2 = self.L2**2 / 12.0
        self.base_x = 0.5
        self.y0 = 0.4
        self.A = 0.1
        self.omega = 0.5
        self.base_y = self.y0
        self.time_t = 0.0
        self.roller_radius = 0.025

        # Initialize Taichi and fields
        ti.init(arch=ti.vulkan)
        self._alloc_fields()
        self.logs = setup_logging_dirs()

    def _alloc_fields(self):
        # Particle fields
        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.n_particles)
        self.J = ti.field(dtype=ti.f32, shape=self.n_particles)

        # Grid fields
        self.grid_v = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))

        # Roller & contact
        self.roller_center = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.roller_velocity = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.contact_force_vec = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)

    @ti.func
    def neo_hookean_stress(self, F_i):
        J_det = F_i.determinant()
        FinvT = F_i.inverse().transpose()
        return self.mu0 * (F_i - FinvT) + self.lambda0 * ti.log(J_det) * FinvT

    @ti.kernel
    def init_mpm(self):
        for p in range(self.n_particles):
            u = ti.random()
            r = 0.2 * ti.sqrt(u)
            theta = ti.random() * 3.141592653589793
            self.x[p] = ti.Vector([0.5 + r * ti.cos(theta), self.floor_level + r * ti.sin(theta)])
            self.v[p] = [0.0, 0.0]
            self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
            self.J[p] = 1.0
            self.C[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)

        # Place roller at rest via FK
        base = ti.Vector([self.base_x, self.base_y])
        j2 = base + ti.Vector([ti.sin(self.theta1[0]), -ti.cos(self.theta1[0])]) * self.L1
        ee = j2 + ti.Vector([ti.sin(self.theta1[0] + self.theta2[0]), -ti.cos(self.theta1[0] + self.theta2[0])]) * self.L2
        self.roller_center[0] = ee
        self.roller_velocity[0] = ti.Vector.zero(ti.f32, self.dim)
        self.contact_force_vec[0] = ti.Vector.zero(ti.f32, self.dim)

    @ti.kernel
    def p2g(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
            self.grid_m[I] = 0.0
        for p in range(self.n_particles):
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - base.cast(ti.f32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx**2) * self.neo_hookean_stress(self.F[p])
            affine = stress + self.p_mass * self.C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                offs = ti.Vector([i, j])
                dpos = (offs.cast(ti.f32) - fx) * self.dx
                wt = w[i].x * w[j].y
                self.grid_v[base + offs] += wt * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offs] += wt * self.p_mass

    @ti.kernel
    def apply_grid_forces_and_detect(self):
        self.contact_force_vec[0] = ti.Vector.zero(ti.f32, self.dim)
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0:
                v_old = self.grid_v[I] / m
                v_new = v_old + self.dt * ti.Vector([0.0, -9.8])
                pos = I.cast(ti.f32) * self.dx
                rel = pos - self.roller_center[0]
                # Roller contact
                if rel.norm() < self.roller_radius:
                    rv = self.roller_velocity[0]
                    n = rel.normalized()
                    v_norm = n * n.dot(rv)
                    v_tan = v_old - n * n.dot(v_old)
                    v_new = v_tan + v_norm
                    delta_v = v_new - v_old
                    f_imp = m * delta_v / self.dt
                    self.contact_force_vec[0] += f_imp
                # Floor & walls
                if pos.y < self.floor_level + self.dx:
                    if v_new.y < 0: v_new.y = 0
                    v_new.x = 0
                if pos.x < self.dx or pos.x > 1 - self.dx:
                    v_new.x = 0
                self.grid_v[I] = v_new * m

    @ti.kernel
    def g2p(self):
        for p in range(self.n_particles):
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - base.cast(ti.f32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(ti.f32, self.dim)
            new_C = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            for i, j in ti.static(ti.ndrange(3, 3)):
                offs = ti.Vector([i, j])
                dpos = (offs.cast(ti.f32) - fx) * self.dx
                wt = w[i].x * w[j].y
                gv = self.grid_v[base + offs] / self.grid_m[base + offs]
                new_v += wt * gv
                new_C += 4 * self.inv_dx * wt * gv.outer_product(dpos)
            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * new_v
            # Boundaries
            if self.x[p].y < self.floor_level:
                self.x[p].y = self.floor_level
                self.v[p] = ti.Vector.zero(ti.f32, self.dim)
            if self.x[p].x < self.dx:
                self.x[p].x, self.v[p].x = self.dx, 0
            if self.x[p].x > 1 - self.dx:
                self.x[p].x, self.v[p].x = 1 - self.dx, 0
            if self.x[p].y > 1 - self.dx:
                self.x[p].y, self.v[p].y = 1 - self.dx, 0
            # Update deformation gradient
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) + self.dt * new_C) @ self.F[p]
            self.J[p] = self.F[p].determinant()

    def update_base_and_arm(self):
        self.time_t += self.dt * 15
        self.base_y = self.y0 + self.A * np.cos(self.omega * self.time_t)
        Fc = self.contact_force_vec[0].to_numpy()
        base = np.array([self.base_x, self.base_y], dtype=np.float32)
        j2 = base + np.array([np.sin(self.theta1[0]), -np.cos(self.theta1[0])]) * self.L1
        ee_old = self.roller_center[0].to_numpy()
        ee_new = j2 + np.array([np.sin(self.theta1[0] + self.theta2[0]), -np.cos(self.theta1[0] + self.theta2[0])])* self.L2
        rv = (ee_new - ee_old) / self.dt
        self.roller_center[0], self.roller_velocity[0] = ee_new.tolist(), rv.tolist()
        r1 = ee_new - base
        r2 = ee_new - j2
        tau1_c = r1[1] * Fc[0] - r1[0] * Fc[1]
        tau2_c = r2[1] * Fc[0] - r2[0] * Fc[1]
        tau1 = tau1_c - self.k1 * (self.theta1[0] - self.theta1_rest[0]) - self.b1 * self.dtheta1[0]
        tau2 = tau2_c - self.k2 * (self.theta2[0] - self.theta2_rest[0]) - self.b2 * self.dtheta2[0]
        alpha1 = tau1 / self.I1
        alpha2 = tau2 / self.I2
        self.dtheta1[0] += alpha1 * self.dt
        self.theta1[0] += self.dtheta1[0] * self.dt
        self.dtheta2[0] += alpha2 * self.dt
        self.theta2[0] += self.dtheta2[0] * self.dt

    def run(self, true_deformations: np.ndarray = None, cycles: int = 100, steps_per_cycle: int = 15):
        self.init_mpm()
        sim_deformations = []
        forces = [] 
        for c in range(cycles):
            prev_x = self.x.to_numpy().copy()
            for _ in range(steps_per_cycle):
                self.p2g()
                self.apply_grid_forces_and_detect()
                self.g2p()
                self.update_base_and_arm()
            curr_x = self.x.to_numpy()
            disp = np.linalg.norm(curr_x - prev_x, axis=1)
            max_disp = float(disp.max())
            sim_deformations.append(max_disp)
            Fc = self.contact_force_vec[0].to_numpy()

            forces.append(float(np.linalg.norm(Fc)))
            # Save logs
            np.save(os.path.join(self.logs, "positions", f"cycle_{c}.npy"), curr_x)
            np.save(os.path.join(self.logs, "velocities", f"cycle_{c}.npy"), self.v.to_numpy())

        # Write deformation log
        np.savetxt(os.path.join(self.logs, "deformations.csv"), sim_deformations, delimiter=',')

        # Plot if true data provided
        # if true_deformations is not None:
        self.plot_deformations(np.array(forces), np.array(sim_deformations))
        print(forces)
        return np.array(sim_deformations)

    @staticmethod
    def plot_deformations(forces,sim: np.ndarray):
        plt.figure()
        cycles = np.arange(len(sim))
        # plt.plot(cycles, true, label='True')
        plt.plot(forces, sim, label='Simulated')
        plt.xlabel('Cycle')
        plt.ylabel('Max Deformation')
        plt.legend()
        plt.title('True vs Simulated Deformations')
        plt.grid(True)
        plt.show()


sim = MPM_Simulator(n_particles=20000, n_grid=128, E=5e4, nu=0.2)
sim_def = sim.run(true_deformations=[], cycles=150, steps_per_cycle=15)