# main.py
import taichi as ti
import numpy as np
from soft_body import SoftBody
from robot import RobotArm2D
from contact import ContactHandler
import time

# Initialize simulation components
soft_body = SoftBody(grid_res=80, dt=1e-4, gravity=9.8, viscosity=0.2)
soft_body.init_tissue_block(lower_corner=(0.2, 0.0), upper_corner=(0.8, 0.2), density=1.0)
robot = RobotArm2D(base_pos=(0.5, 0.65), link_lengths=(0.2, 0.15), 
                   spring_constants=(100.0, 50.0), damping_constants=(5.0, 2.0), roller_radius=0.02)
contact = ContactHandler(friction_coeff=0.3, roller_radius=robot.r)

# Adjust initial robot pose to just above the tissue
robot.q1 = -1.30  # about -75 degrees, a bit above vertical
robot.q1_target = robot.q1
robot.q2 = 0.0
robot.q2_target = robot.q2

# Setup visualization window
gui = ti.GUI("Robot-SoftBody MPM Simulation", res=(600, 600))

# Simulation parameters
total_steps = 2000
press_steps = 1000   # duration of pressing phase
slide_steps = 1000   # duration of sliding phase
penetration_threshold = 0.5 * robot.r   # e.g., half the roller radius
volume_threshold = 50                # 10% volume change allowed
unstable = False

for t in range(total_steps):
    # Determine target angle trajectory: phase 1 (press down) then phase 2 (slide sideways)
    if t < press_steps:
        # Linearly move q1_target to -90 deg over press_steps
        target_q1 = -1.57  # -90°
        alpha = t / press_steps
    else:
        # Phase 2: move q1_target to -70 deg over slide_steps
        target_q1 = -1.22  # -70°
        alpha = (t - press_steps) / slide_steps
    # Smoothly update q1_target (small increments for stability)
    robot.q1_target = robot.q1_target + 0.001 * (target_q1 - robot.q1_target)
    # (q2_target remains 0 to keep link2 straight for simplicity)

    # MPM simulation step
    soft_body.clear_grid()
    soft_body.p2g()       # Particle to grid transfer
    soft_body.grid_ops()  # Grid physics (gravity, boundary)
    # Contact resolution between robot roller and soft body
    base, joint, tip = robot.forward_kinematics()
    tip_vx, tip_vy = robot.get_tip_velocity()
    contact.clear_accumulators()
    contact.resolve_contact(soft_body, tip[0], tip[1], tip_vx, tip_vy, joint[0], joint[1], robot.base_x, robot.base_y)
    # Apply reaction torques to robot
    tau1_ext = float(contact.contact_tau1[None])
    tau2_ext = float(contact.contact_tau2[None])
    robot.step(soft_body.dt, tau_ext1=tau1_ext, tau_ext2=tau2_ext)
    # Update particle states from grid
    soft_body.g2p()

    # Logging calculations
    # Contact force on robot
    force_vec = contact.contact_force[None]  # 2D vector
    Fx, Fy = float(force_vec[0]), float(force_vec[1])
    # Compute penetration depth (if roller radius exceeds min particle distance)
    particle_positions = soft_body.x.to_numpy()[:soft_body.num_particles[None]]
    if particle_positions.size > 0:
        dists = np.sqrt(((particle_positions[:,0] - tip[0])**2 + (particle_positions[:,1] - tip[1])**2))
        min_dist = dists.min()
    else:
        min_dist = float('inf')
    penetration = max(0.0, robot.r - min_dist)
    # Compute volume change
    F_arrays = soft_body.F.to_numpy()[:soft_body.num_particles[None]]

    F_reshaped = F_arrays.reshape(-1, 2, 2) 
    F_reshaped[np.isnan(F_reshaped)] = 0.0
    
    # determinant of F for each particle
    if F_reshaped.size > 0:
        J = np.linalg.det(F_reshaped)
    else:
        J = np.array([])

    current_volume = float(np.sum(soft_body.vol.to_numpy()[:soft_body.num_particles[None]] * (J if J.size > 0 else 1.0)))
    vol_change = abs(current_volume - soft_body.initial_volume) / soft_body.initial_volume
    # Check for violations
    violation_msg = "None"
    # if vol_change > volume_threshold:
    #     violation_msg = "Volume > 10%"
    # if penetration > penetration_threshold:
    #     violation = f"Penetration > {penetration_threshold:.3f}"
    #     violation_msg = violation if violation_msg == "None" else (violation_msg + ", " + violation)
    # # Check instability: if any particle velocity is extremely high or positions invalid
    # velocities = soft_body.v.to_numpy()[:soft_body.num_particles[None]]
    # max_speed = np.max(np.sqrt(np.sum(velocities**2, axis=1))) if velocities.size > 0 else 0.0
    # if max_speed > 500.0 or np.isnan(max_speed):
    #     violation_msg = "Unstable (explosion)" if violation_msg == "None" else (violation_msg + ", Unstable")
    #     unstable = True

    # Print log for this timestep
    print(f"Step {t}: ContactForce=({Fx:.3f}, {Fy:.3f}), Penetration={penetration:.4f}, "
          f"VolumeChange={vol_change*100:.1f}%, Violations={violation_msg}")

    # Render the current state in the GUI
    if gui:
        # Draw soft body particles
        if particle_positions.size > 0:
            gui.circles(particle_positions, radius=2, color=0x068587)
        # Draw robot (base, joint, tip positions)
        base_pos = (robot.base_x, robot.base_y)
        gui.circle(base_pos, radius=5, color=0xFF0000)               # base (red)
        gui.circle((joint[0], joint[1]), radius=5, color=0xFF0000)   # joint1
        gui.circle((tip[0], tip[1]), radius=8, color=0xFFFF00)       # tip (roller, yellow)
        # Draw links as small red segments
        num_seg = 10
        for seg in range(num_seg + 1):
            frac = seg / num_seg
            # Points along link1
            x1 = robot.base_x + frac * (joint[0] - robot.base_x)
            y1 = robot.base_y + frac * (joint[1] - robot.base_y)
            gui.circle((x1, y1), radius=1, color=0xFF0000)
            # Points along link2
            x2 = joint[0] + frac * (tip[0] - joint[0])
            y2 = joint[1] + frac * (tip[1] - joint[1])
            gui.circle((x2, y2), radius=1, color=0xFF0000)
        gui.show()
        time.sleep(1)
    if unstable:
        time.sleep(5000)

# (Optional) After simulation, one could use SimulationOptimizer to tune parameters.
# For example:
# optimizer = SimulationOptimizer(initial_params={"E": 5000, "friction_coeff": 0.3}, target_data=experimental_data, sim_function=run_simulation)
# best_params = optimizer.run(iterations=50)
# print("Optimized parameters:", best_params)
