import taichi as ti
import numpy as np
import time

PI = 3.141592653589793

@ti.data_oriented
class straight_massager:
    def __init__(self, cfg):
        self.cfg = cfg

        self.dim = 2
        self.dt  = 2e-4
        

        self.L_arm  = 0.12
        self.theta0 = 0.0
        self.I      = self.L_arm**2 / 12.0
        self.k      = 1.0
        self.b      = 0.2

        # Initial joint states
        self.base    = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)


        # Moving base (vertical oscillation)
        self.y0, self.base_x = 0.325, 0.5
        self.A, self.omega = 0.025, 0.5
        self.base_y = self.y0
        self.time_t = 0.0
        self.Time_period = 2 * PI / self.omega * 2

        # Roller/contact
        self.roller_radius = 0.025
        self.roller_center = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.roller_velocity = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.contact_force = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)

        
    @ti.kernel
    def initialize(self):

        # Place rollers at FK
        base = ti.Vector([self.base_x, self.base_y])

        tip  = base + ti.Vector([ti.sin(self.theta0), -ti.cos(self.theta0)]) * self.L_arm
        self.roller_center[0] = tip
        self.roller_velocity[0] = ti.Vector.zero(ti.f32, self.dim)
        self.contact_force[0] = ti.Vector.zero(ti.f32, self.dim)

        


    def step(self):  
        global time_t, base_y
        # Time & base vertical update
        self.time_t += self.dt * 15
        self.base_y = self.y0 + self.A * np.cos(self.omega * self.time_t)

        # ---------- Right side ----------
        base  = np.array([self.base_x, self.base_y], dtype=np.float32)

        ee_o  = self.roller_center[0].to_numpy()
        ee_n  = base + np.array([np.sin(self.theta0), -np.cos(self.theta0)], dtype=np.float32) * self.L_arm
        rv    = (ee_n - ee_o) / self.dt

        self.roller_center[0]   = ee_n.tolist()
        self.roller_velocity[0] = rv.tolist()


    @ti.func
    def reset_contact_forces(self):
        self.contact_force[0] = ti.Vector.zero(ti.f32, self.dim)

    @ti.func
    def update_contact_info(self, m,pos, v_old, v_new):

        # Right roller contact (normal component)
        rel_r = pos - self.roller_center[0]
        if rel_r.norm() < self.roller_radius:
            rv     = self.roller_velocity[0]
            n      = rel_r.normalized()
            v_norm = n * n.dot(rv)
            v_tan  = v_old - n * (n.dot(v_old))
            v_new  = v_tan + v_norm
            delta_v = v_new - v_old
            f_imp   = m * delta_v / self.dt
            self.contact_force[0] += f_imp

        return v_new