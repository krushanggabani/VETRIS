import taichi as ti
import numpy as np
import time

PI = 3.141592653589793

@ti.data_oriented
class dual_arm_massager:
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.dim = 2
        self.dt  = 2e-4
        # 2-link passive arm params (single hand, duplicated left/right)
        self.L1, self.L2 = 0.12, 0.10
        self.I1     = self.L1**2 / 12.0
        self.I2     = self.L2**2 / 12.0
        self.k      = 1.0
        self.b      = 0.2
        self.slimit = np.array([5.0, 90.0], dtype=np.float32)  # deg, symmetric limits

        # Initial joint states
        self.thetaleft    = np.array([-5.0 * PI / 180, -5.0 * PI / 180], dtype=np.float32)
        self.thetaright   = np.array([ 5.0 * PI / 180,  5.0 * PI / 180], dtype=np.float32)
        self.dthetaleft   = np.array([0.0, 0.0], dtype=np.float32)
        self.dthetaright  = np.array([0.0, 0.0], dtype=np.float32)
        self.theta_rest_left  = self.thetaleft.copy()
        self.theta_rest_right = self.thetaright.copy()


        # Moving base (vertical oscillation)
        self.y0, self.base_x = 0.4, 0.5
        self.A, self.omega = 0.075, 0.5
        self.base_y = self.y0
        self.time_t = 0.0
        self.Time_period = 2 * PI / self.omega * 2

        # Roller/contact
        self.roller_radius = 0.025
        self.roller_center_right = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.roller_velocity_right = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.roller_center_left = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.roller_velocity_left = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.contact_force_vec_r = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.contact_force_vec_l = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)


    @ti.kernel
    def initialize(self):

        # Place rollers at FK
        base = ti.Vector([self.base_x, self.base_y])

        #  Right side
        j2_r = base + ti.Vector([ti.sin(self.thetaright[0]), -ti.cos(self.thetaright[0])]) * self.L1
        ee_r = j2_r + ti.Vector(
            [ti.sin(self.thetaright[0] + self.thetaright[1]), -ti.cos(self.thetaright[0] + self.thetaright[1])]
        ) * self.L2
        self.roller_center_right[0] = ee_r
        self.roller_velocity_right[0] = ti.Vector.zero(ti.f32, self.dim)
        self.contact_force_vec_r[0] = ti.Vector.zero(ti.f32, self.dim)

        # Left side  
        j2_l = base + ti.Vector([ti.sin(self.thetaleft[0]), -ti.cos(self.thetaleft[0])]) * self.L1
        ee_l = j2_l + ti.Vector(
            [ti.sin(self.thetaleft[0] + self.thetaleft[1]), -ti.cos(self.thetaleft[0] + self.thetaleft[1])]
        ) * self.L2
        self.roller_center_left[0] = ee_l
        self.roller_velocity_left[0] = ti.Vector.zero(ti.f32, self.dim)
        self.contact_force_vec_l[0] = ti.Vector.zero(ti.f32, self.dim)


    def step(self):  
        global time_t, base_y, thetaright, dthetaright, thetaleft, dthetaleft

        # Time & base vertical update
        self.time_t += self.dt * 15
        self.base_y = self.y0 + self.A * np.cos(self.omega * self.time_t)

        # ---------- Right side ----------
        Fc_r  = self.contact_force_vec_r[0].to_numpy()
        base  = np.array([self.base_x, self.base_y], dtype=np.float32)

        j2    = base + np.array([np.sin(self.thetaright[0]), -np.cos(self.thetaright[0])]) * self.L1
        ee_o  = self.roller_center_right[0].to_numpy()
        ee_n  = j2 + np.array([np.sin(self.thetaright[0] + self.thetaright[1]),
                            -np.cos(self.thetaright[0] + self.thetaright[1])]) * self.L2
        rv    = (ee_n - ee_o) / self.dt

        self.roller_center_right[0]   = ee_n.tolist()
        self.roller_velocity_right[0] = rv.tolist()

        r1     = ee_n - base
        r2     = ee_n - j2
        tau1_c = r1[1] * Fc_r[0] - r1[0] * Fc_r[1]
        tau2_c = r2[1] * Fc_r[0] - r2[0] * Fc_r[1]

        tau1 = tau1_c - self.k * (self.thetaright[0] - self.theta_rest_right[0]) - self.b * self.dthetaright[0]
        tau2 = tau2_c - self.k * (self.thetaright[1] - self.theta_rest_right[1]) - self.b * self.dthetaright[1]

        self.dthetaright[0] += (tau1 / self.I1) * self.dt
        self.thetaright[0]  += self.dthetaright[0] * self.dt
        self.dthetaright[1] += (tau2 / self.I2) * self.dt
        self.thetaright[1]  += self.dthetaright[1] * self.dt

        # Joint limits (deg → rad)
        lo, hi = self.slimit[0] * PI / 180.0, self.slimit[1] * PI / 180.0
        if self.thetaright[0] < lo: self.thetaright[0] = lo
        elif self.thetaright[0] > hi: self.thetaright[0] = hi
        if self.thetaright[1] < lo: self.thetaright[1] = lo
        elif self.thetaright[1] > hi: self.thetaright[1] = hi

        # ---------- Left side ----------
        Fc_l  = self.contact_force_vec_l[0].to_numpy()
        base  = np.array([self.base_x, self.base_y], dtype=np.float32)

        j2 = base + np.array([np.sin(self.thetaleft[0]), -np.cos(self.thetaleft[0])]) * self.L1
        ee_o = self.roller_center_left[0].to_numpy()  # FIX A: use left's own old pose
        ee_n = j2 + np.array(
            [np.sin(self.thetaleft[0] + self.thetaleft[1]), -np.cos(self.thetaleft[0] + self.thetaleft[1])]
        ) * self.L2
        rv = (ee_n - ee_o) / self.dt

        self.roller_center_left[0] = ti.Vector(list(ee_n))
        self.roller_velocity_left[0] = ti.Vector(list(rv))

        r1 = ee_n - base
        r2 = ee_n - j2
        tau1_c = r1[1] * Fc_l[0] - r1[0] * Fc_l[1]
        tau2_c = r2[1] * Fc_l[0] - r2[0] * Fc_l[1]

        tau1 = tau1_c - self.k * (self.thetaleft[0] - self.theta_rest_left[0]) - self.b * self.dthetaleft[0]
        tau2 = tau2_c - self.k * (self.thetaleft[1] - self.theta_rest_left[1]) - self.b * self.dthetaleft[1]

        self.dthetaleft[0] += (tau1 / self.I1) * self.dt
        self.thetaleft[0] += self.dthetaleft[0] * self.dt
        self.dthetaleft[1] += (tau2 / self.I2) * self.dt
        self.thetaleft[1] += self.dthetaleft[1] * self.dt

        # Asymmetric mirrored limits (kept as in your original)
        nlo, nhi = -self.slimit[0] * PI / 180.0, -self.slimit[1] * PI / 180.0
        if self.thetaleft[0] > nlo:
            self.thetaleft[0] = nlo
        elif self.thetaleft[0] < nhi:
            self.thetaleft[0] = nhi
        if self.thetaleft[1] > nlo:
            self.thetaleft[1] = nlo
        elif self.thetaleft[1] < nhi:
            self.thetaleft[1] = nhi

    @ti.func
    def reset_contact_forces(self):
        self.contact_force_vec_r[0] = ti.Vector.zero(ti.f32, 2)
        self.contact_force_vec_l[0] = ti.Vector.zero(ti.f32, 2)

    @ti.func
    def update_contact_info(self, m,pos, v_old, v_new):

        # Right roller contact (normal component)
        rel_r = pos - self.roller_center_right[0]
        if rel_r.norm() < self.roller_radius:
            rv     = self.roller_velocity_right[0]
            n      = rel_r.normalized()
            v_norm = n * n.dot(rv)
            v_tan  = v_old - n * (n.dot(v_old))
            v_new  = v_tan + v_norm
            delta_v = v_new - v_old
            f_imp   = m * delta_v / self.dt
            self.contact_force_vec_r[0] += f_imp

        # Left roller contact (normal component; FIX B: assign v_new too)
        rel_l = pos - self.roller_center_left[0]
        if rel_l.norm() < self.roller_radius:
            rv     = self.roller_velocity_left[0]
            n      = rel_l.normalized()
            v_norm = n * n.dot(rv)
            v_tan  = v_old - n * (n.dot(v_old))
            v_new  = v_tan + v_norm          # ← ensure same constraint as right
            delta_v = v_new - v_old
            f_imp   = m * delta_v / self.dt
            self.contact_force_vec_l[0] += f_imp

        return v_new