# contact.py
import taichi as ti

@ti.data_oriented
class ContactHandler:
    def __init__(self, friction_coeff=0.3, roller_radius=0.02):
        self.friction_coeff = friction_coeff
        self.roller_radius = roller_radius
        # Accumulators for contact forces and torques
        self.contact_force = ti.Vector.field(2, float, shape=())
        self.contact_tau1 = ti.field(float, shape=())
        self.contact_tau2 = ti.field(float, shape=())

    def clear_accumulators(self):
        # Reset accumulated force and torques to zero
        self.contact_force[None] = ti.Vector([0.0, 0.0])
        self.contact_tau1[None] = 0.0
        self.contact_tau2[None] = 0.0

    @ti.kernel
    def resolve_contact(self, sb: ti.template(), tip_x: ti.f32, tip_y: ti.f32, tip_vx: ti.f32, tip_vy: ti.f32, 
                        joint2_x: ti.f32, joint2_y: ti.f32, base_x: ti.f32, base_y: ti.f32):
        # Iterate over all grid nodes of the soft body
        for I in ti.grouped(sb.grid_v):
            # Compute world coordinates of this grid node (cell center)
            # (Domain [0,1] with sb.n cells, each cell size = sb.dx)
            ti_x = (I.x + 0.5) * sb.dx
            ti_y = (I.y + 0.5) * sb.dx
            # Vector from roller center to grid node
            dx = ti_x - tip_x
            dy = ti_y - tip_y
            dist = ti.sqrt(dx * dx + dy * dy)
            if dist < self.roller_radius:
                # Contact: grid node is inside roller radius
                # Compute unit normal from roller center to node (outward from roller)
                nx = dx / (dist + 1e-6)
                ny = dy / (dist + 1e-6)
                # Relative velocity of soft body at node w.rt roller
                # (roller velocity is (tip_vx, tip_vy))
                rel_vx = sb.grid_v[I].x - tip_vx
                rel_vy = sb.grid_v[I].y - tip_vy
                # Normal relative velocity (scalar projection)
                let_vn = rel_vx * nx + rel_vy * ny
                if let_vn < 0.0:  # material moving into the roller
                    # Save old grid velocity (for impulse calculation)
                    old_vx = sb.grid_v[I].x
                    old_vy = sb.grid_v[I].y
                    # Remove normal component of relative velocity (prevent penetration)
                    # New relative normal velocity = 0 (no penetration)
                    # Compute original tangential velocity components
                    rel_vt_x = rel_vx - let_vn * nx
                    rel_vt_y = rel_vy - let_vn * ny
                    # Apply Coulomb friction: limit tangential velocity
                    # Calculate magnitude of tangential relative velocity
                    vt_mag = ti.sqrt(rel_vt_x * rel_vt_x + rel_vt_y * rel_vt_y)
                    if vt_mag > 0.0:
                        # Maximum allowed tangential reduction based on friction coefficient
                        # Using |v_n| (magnitude of original normal relative velocity) for friction criterion
                        max_reduction = self.friction_coeff * (-let_vn)
                        if vt_mag <= max_reduction:
                            # Stick: cancel all tangential motion
                            rel_vt_x = 0.0
                            rel_vt_y = 0.0
                        else:
                            # Slip: reduce tangential velocity by friction impulse
                            # (reduce magnitude by max_reduction)
                            rel_vt_x *= (vt_mag - max_reduction) / vt_mag
                            rel_vt_y *= (vt_mag - max_reduction) / vt_mag
                    # Set new grid velocity = roller velocity + adjusted relative velocity
                    sb.grid_v[I].x = tip_vx + rel_vt_x
                    sb.grid_v[I].y = tip_vy + rel_vt_y
                    # Compute momentum impulse delivered to this grid node
                    # m * Δv = m * (v_new - v_old)
                    m_node = sb.grid_m[I]
                    new_vx = sb.grid_v[I].x
                    new_vy = sb.grid_v[I].y
                    imp_x = m_node * (new_vx - old_vx)
                    imp_y = m_node * (new_vy - old_vy)
                    # Force on robot is opposite of force on soft body
                    # (Impulse on robot = -impulse on node; Force = impulse / dt)
                    ti.atomic_add(self.contact_force[None][0], -imp_x / sb.dt)
                    ti.atomic_add(self.contact_force[None][1], -imp_y / sb.dt)
                    # Compute torques on joints due to this contact force
                    # r (base->tip) and r (joint2->tip) vectors:
                    # Torque = r_x * F_y - r_y * F_x
                    tau1 = (tip_x - base_x) * (-imp_y / sb.dt) - (tip_y - base_y) * (-imp_x / sb.dt)
                    tau2 = (tip_x - joint2_x) * (-imp_y / sb.dt) - (tip_y - joint2_y) * (-imp_x / sb.dt)
                    ti.atomic_add(self.contact_tau1[None], tau1)
                    ti.atomic_add(self.contact_tau2[None], tau2)
