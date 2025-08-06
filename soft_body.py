import taichi as ti

ti.init(arch=ti.cpu)

@ti.data_oriented
class SoftBody:
    def __init__(self, grid_res=80, dt=1e-4, gravity=9.8, viscosity=0.2):
        # Simulation parameters
        self.n = grid_res
        self.dt = dt
        self.gravity = gravity
        self.viscosity = viscosity
        # Material properties (elastic modulus and Poisson's ratio)
        self.E = 5e3  # Young's modulus
        self.nu = 0.1  # Poisson's ratio
        # Compute Lamé parameters for elasticity
        self.lambda_ = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))
        # Grid resolution and spacing
        self.inv_dx = float(self.n)
        self.dx = 1.0 / self.inv_dx
        # Maximum number of particles
        self.max_particles = 10000
        # Particle fields: position (x), velocity (v), deformation gradient (F), volume (vol), mass
        self.x = ti.Vector.field(2, float, shape=self.max_particles)
        self.v = ti.Vector.field(2, float, shape=self.max_particles)
        self.F = ti.Matrix.field(2, 2, float, shape=self.max_particles)
        self.vol = ti.field(float, shape=self.max_particles)
        self.mass = ti.field(float, shape=self.max_particles)
        # Grid fields: velocity (momentum) and mass
        self.grid_v = ti.Vector.field(2, float, shape=(self.n, self.n))
        self.grid_m = ti.field(float, shape=(self.n, self.n))
        # Number of active particles
        self.num_particles = ti.field(int, shape=())
        # Track initial total volume for volume change monitoring
        self.initial_volume = 0.0

    def init_tissue_block(self, lower_corner, upper_corner, density=1.0):
        '''Initialize a rectangular block of soft tissue particles.'''
        lx, ly = lower_corner
        ux, uy = upper_corner
        particle_positions = []
        spacing = self.dx  # place one particle per grid cell in the region
        i = 0
        while lx + (i + 0.5) * spacing <= ux:
            j = 0
            while ly + (j + 0.5) * spacing <= uy:
                x_pos = lx + (i + 0.5) * spacing
                y_pos = ly + (j + 0.5) * spacing
                particle_positions.append((x_pos, y_pos))
                j += 1
            i += 1
        num = len(particle_positions)
        self.num_particles[None] = num
        assert num <= self.max_particles, "Too many particles in block, increase max_particles"
        # Initialize particle properties
        for p in range(num):
            px, py = particle_positions[p]
            self.x[p] = ti.Vector([px, py])
            self.v[p] = ti.Vector([0.0, 0.0])
            self.F[p] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])  # undeformed (identity)
        # Assign particle volume and mass (assuming uniform density)
        part_vol = self.dx**2
        part_mass = part_vol * density
        for p in range(num):
            self.vol[p] = part_vol
            self.mass[p] = part_mass
        self.initial_volume = part_vol * num

    @ti.kernel
    def clear_grid(self):
        # Reset grid masses and velocities to zero before each particle-to-grid (P2G) transfer
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector([0.0, 0.0])
            self.grid_m[I] = 0.0

    @ti.kernel
    def p2g(self):
        # Particle-to-Grid transfer: scatter particle mass and momentum to grid
        for p in range(self.num_particles[None]):
            # Compute particle's cell index (lower-left corner) and local position (fx, fy) within cell
            ti_position = self.x[p] * self.inv_dx
            base = ti.cast(ti.floor(ti_position - 0.5), ti.i32)  # index of cell lower-left of particle
            frac = ti_position - base.cast(float)               # particle's local fractional offset in cell
            # Linear shape function weights for 2D (weight in x and y for cell nodes)
            wx = [1.0 - frac.x, frac.x]
            wy = [1.0 - frac.y, frac.y]
            # Particle properties
            Fp = self.F[p]
            mass_p = self.mass[p]
            vol_p = self.vol[p]
            # Compute particle stress (First Piola-Kirchhoff)
            # Determinant of F (J) for volume change
            J = Fp.determinant()
            # Neo-Hookean first Piola stress: P = mu*(F - F^{-T}) + lambda*log(J)*F^{-T}
            F_inv_T = Fp.inverse().transpose()
            P = self.mu * (Fp - F_inv_T) + self.lambda_ * ti.log(J) * F_inv_T
            # Loop over the 2x2 grid cell nodes around the particle
            for i_local in ti.static(range(2)):
                for j_local in ti.static(range(2)):
                    w_ij = wx[i_local] * wy[j_local]  # weight for this node
                    node_idx = base + ti.Vector([i_local, j_local])
                    # Add particle mass to grid node
                    ti.atomic_add(self.grid_m[node_idx], w_ij * mass_p)
                    # Compute gradient of shape function (∇N) for this node (for stress force)
                    gradN_x = 0.0
                    gradN_y = 0.0
                    if j_local == 0:
                        gradN_x = -(1.0 - frac.y) if i_local == 0 else (1.0 - frac.y)
                    else:
                        gradN_x = -frac.y if i_local == 0 else frac.y
                    if i_local == 0:
                        gradN_y = -(1.0 - frac.x) if j_local == 0 else (1.0 - frac.x)
                    else:
                        gradN_y = -frac.x if j_local == 0 else frac.x
                    gradN = ti.Vector([gradN_x, gradN_y]) * self.inv_dx
                    # Momentum contribution from particle velocity
                    ti.atomic_add(self.grid_v[node_idx][0], w_ij * mass_p * self.v[p].x)
                    ti.atomic_add(self.grid_v[node_idx][1], w_ij * mass_p * self.v[p].y)
                    # Momentum contribution from particle stress (force * dt)
                    # Use affine momentum: -dt * vol * (P * ∇N) added to grid momentum
                    # (We distribute internal force to grid as equivalent momentum change)
                    ti.atomic_add(self.grid_v[node_idx][0], - self.dt * vol_p * (P @ gradN)[0])
                    ti.atomic_add(self.grid_v[node_idx][1], - self.dt * vol_p * (P @ gradN)[1])

    @ti.kernel
    def grid_ops(self):
        # Grid operations: update grid velocities with gravity, collisions, and boundaries
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                # Convert momentum to velocity
                self.grid_v[I] = self.grid_v[I] / self.grid_m[I]
                # Apply gravity to vertical velocity
                self.grid_v[I].y -= self.gravity * self.dt
                # Floor boundary (y=0): prevent penetration and apply friction
                if I.y < 1:  # bottom row of grid nodes
                    if self.grid_v[I].y < 0:
                        self.grid_v[I].y = 0  # no downward velocity into floor
                    # Friction: reduce horizontal velocity based on friction coefficient
                    if self.grid_v[I].x != 0:
                        # Compute friction impulse magnitude = viscosity * normal_vel (using viscosity as friction coeff)
                        normal_vel = self.grid_v[I].y
                        # Only apply friction if normal contact (normal_vel ~ 0 when in contact or just prevented)
                        # In static contact, normal_vel = 0 after above step, so we use prior normal momentum implicitly
                        friction_impulse = normal_vel * self.viscosity
                        if abs(self.grid_v[I].x) < abs(friction_impulse):
                            self.grid_v[I].x = 0.0  # static friction holds node
                        else:
                            sign = ti.select(self.grid_v[I].x > 0, 1.0, ti.select(self.grid_v[I].x < 0, -1.0, 0.0))

                            self.grid_v[I].x -= friction_impulse * sign
                # Side boundaries (x=0 and x=1): reflect horizontal velocity
                if I.x < 1 and self.grid_v[I].x < 0:
                    self.grid_v[I].x = 0
                if I.x > self.n - 2 and self.grid_v[I].x > 0:
                    self.grid_v[I].x = 0

    
    
    @ti.kernel
    def g2p(self):
        # Grid-to-Particle: interpolate grid velocity back to particles, update particles
        for p in range(self.num_particles[None]):
            # Find cell and interpolation weights as in P2G
            ti_position = self.x[p] * self.inv_dx
            base = ti.cast(ti.floor(ti_position - 0.5), ti.i32)
            frac = ti_position - base.cast(float)
            wx = [1.0 - frac.x, frac.x]
            wy = [1.0 - frac.y, frac.y]
            # Initialize new particle velocity and velocity gradient
            new_v = ti.Vector([0.0, 0.0])
            grad_v = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            # Interpolate velocity and compute velocity gradient from nearby grid nodes
            for i_local in ti.static(range(2)):
                for j_local in ti.static(range(2)):
                    w_ij = wx[i_local] * wy[j_local]
                    node_idx = base + ti.Vector([i_local, j_local])
                    # Grid node velocity
                    gv = self.grid_v[node_idx]
                    new_v += w_ij * gv
                    # Compute ∇N (same as in P2G) for velocity gradient
                    gradN_x = 0.0
                    gradN_y = 0.0
                    if j_local == 0:
                        gradN_x = -(1.0 - frac.y) if i_local == 0 else (1.0 - frac.y)
                    else:
                        gradN_x = -frac.y if i_local == 0 else frac.y
                    if i_local == 0:
                        gradN_y = -(1.0 - frac.x) if j_local == 0 else (1.0 - frac.x)
                    else:
                        gradN_y = -frac.x if j_local == 0 else frac.x
                    gradN = ti.Vector([gradN_x, gradN_y]) * self.inv_dx
                    grad_v += gv.outer_product(gradN)
            # Update particle velocity with damping (viscosity)
            self.v[p] = new_v * (1.0 - self.viscosity * self.dt)
            # Update particle position
            self.x[p] += self.dt * self.v[p]
            # Update deformation gradient F (Jacobian of deformation)
            self.F[p] = (ti.Matrix([[1.0, 0.0], [0.0, 1.0]]) + self.dt * grad_v) @ self.F[p]
