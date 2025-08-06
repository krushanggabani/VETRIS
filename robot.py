# robot.py
import math

class RobotArm2D:
    def __init__(self, base_pos=(0.5, 0.7), link_lengths=(0.3, 0.3), 
                 spring_constants=(100.0, 50.0), damping_constants=(5.0, 2.0), roller_radius=0.02):
        self.base_x, self.base_y = base_pos
        self.l1, self.l2 = link_lengths
        self.k1, self.k2 = spring_constants
        self.c1, self.c2 = damping_constants
        self.r = roller_radius  # roller radius at end-effector
        # Link masses (for dynamics calculations)
        self.m1 = 1.0
        self.m2 = 1.0
        # Moments of inertia about the joint (assuming uniform rod)
        self.I1 = (1/3) * self.m1 * (self.l1 ** 2)
        self.I2 = (1/3) * self.m2 * (self.l2 ** 2)
        # Initial joint angles (q1 from horizontal axis, q2 relative between links)
        # Default orientation: pointing downward
        self.q1 = -0.5 * math.pi  # -90 degrees
        self.q2 = 0.0           # link2 in line with link1
        self.q1_dot = 0.0
        self.q2_dot = 0.0
        # Target (desired) joint angles for springs (initialize to current angles)
        self.q1_target = self.q1
        self.q2_target = self.q2

    def forward_kinematics(self):
        """Return coordinates of base, joint (end of link1), and tip (roller center)."""
        base = (self.base_x, self.base_y)
        # Joint (link1 end) position
        joint_x = self.base_x + self.l1 * math.cos(self.q1)
        joint_y = self.base_y + self.l1 * math.sin(self.q1)
        joint = (joint_x, joint_y)
        # Tip (link2 end) position
        tip_x = joint_x + self.l2 * math.cos(self.q1 + self.q2)
        tip_y = joint_y + self.l2 * math.sin(self.q1 + self.q2)
        tip = (tip_x, tip_y)
        return base, joint, tip

    def get_tip_velocity(self):
        """Compute the linear velocity (vx, vy) of the roller (end-effector)."""
        # Use the Jacobian of the 2-link arm to get end-effector velocity
        s1 = math.sin(self.q1);  c1 = math.cos(self.q1)
        s12 = math.sin(self.q1 + self.q2);  c12 = math.cos(self.q1 + self.q2)
        # Partial derivatives of tip position w.rt q1 and q2
        # Tip pos: (base_x + l1*cos(q1) + l2*cos(q1+q2), base_y + l1*sin(q1) + l2*sin(q1+q2))
        # vx = ∂x/∂q1 * q1_dot + ∂x/∂q2 * q2_dot
        # vy = ∂y/∂q1 * q1_dot + ∂y/∂q2 * q2_dot
        # Compute these using chain rule:
        # ∂x/∂q1 = -l1*sin(q1) - l2*sin(q1+q2);   ∂x/∂q2 = - l2 * sin(q1+q2)
        # ∂y/∂q1 =  l1*cos(q1) + l2*cos(q1+q2);   ∂y/∂q2 =   l2 * cos(q1+q2)
        vx = - (self.l1 * s1 + self.l2 * s12) * self.q1_dot  -  (self.l2 * s12) * self.q2_dot
        vy =   (self.l1 * c1 + self.l2 * c12) * self.q1_dot  +  (self.l2 * c12) * self.q2_dot
        return (vx, vy)

    def step(self, dt, tau_ext1=0.0, tau_ext2=0.0):
        """
        Integrate the robot's motion by time step dt, applying external torques tau_ext1 and tau_ext2 on joint1 and joint2.
        tau_ext* are typically reaction torques from contact with the soft body.
        """
        # Spring-damper torques for compliance
        tau_spring1 = -self.k1 * (self.q1 - self.q1_target) - self.c1 * self.q1_dot
        tau_spring2 = -self.k2 * (self.q2 - self.q2_target) - self.c2 * self.q2_dot
        # Total torques on joints (including contact torques)
        tau1 = tau_spring1 + tau_ext1
        tau2 = tau_spring2 + tau_ext2
        # Equations of motion (simplified decoupled inertia for each joint)
        # Note: In a real 2-link system, tau2 also affects joint1 due to reaction forces.
        # Here we approximate by treating them independently for compliance simulation.
        q1_ddot = tau1 / self.I1
        q2_ddot = tau2 / self.I2
        # Integrate angular velocities and angles
        self.q1_dot += q1_ddot * dt
        self.q2_dot += q2_ddot * dt
        self.q1 += self.q1_dot * dt
        self.q2 += self.q2_dot * dt
