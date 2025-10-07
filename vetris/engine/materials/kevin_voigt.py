import taichi as ti

@ti.data_oriented
class KelvinVoigtMaterial:
    """
    Linear Kelvin–Voigt model in Green strain (finite strain wrapper):
      S = 2 μ dev(E) + λ tr(E) I  +  2 η_s dev(D) + ζ tr(D) I
      P = F @ S
    where:
      E = 0.5 (C - I),   C = FᵀF,   D = sym(Ċ) proxy from APIC affine C.
    """
    def __init__(self, cfg, dim: int = 2):
        self.dim = int(dim)

        # Scalar Taichi fields for GPU/device use
        self.mu_0      = ti.field(dtype=ti.f32, shape=())
        self.lambda_0  = ti.field(dtype=ti.f32, shape=())
        self.eta_shear = ti.field(dtype=ti.f32, shape=())
        self.eta_bulk  = ti.field(dtype=ti.f32, shape=())

        # Lamé parameters from (E, ν)
        E  = float(cfg.youngs_modulus)
        nu = float(cfg.poisson_ratio)
        mu  = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Assign to device fields
        self.mu_0[None]      = mu
        self.lambda_0[None]  = lam
        self.eta_shear[None] = float(getattr(cfg, "shear_viscosity", 5.0))
        self.eta_bulk[None]  = float(getattr(cfg, "bulk_viscosity", 50.0))

    # ---------------------- helper ops ----------------------
    @ti.func
    def I(self):
        return ti.Matrix.identity(ti.f32, self.dim)

    @ti.func
    def trace(self, M: ti.template()) -> ti.f32:
        """Compute matrix trace manually (no ti.trace)."""
        s = ti.cast(0.0, ti.f32)
        for i in ti.static(range(self.dim)):
            s += M[i, i]
        return s

    @ti.func
    def dev(self, A: ti.template()):
        """Deviatoric part of a tensor."""
        return A - (self.trace(A) / self.dim) * self.I()

    @ti.func
    def green_E(self, F: ti.template()):
        """Green strain tensor: E = 0.5 (C - I)."""
        C = F.transpose() @ F
        return 0.5 * (C - self.I())

    @ti.func
    def D_from_C(self, C_i: ti.template()):
        """Symmetric velocity-gradient proxy from APIC C."""
        return 0.5 * (C_i + C_i.transpose())

    # ---------------------- constitutive law ----------------------
    @ti.func
    def pk1_update(self, F_i: ti.template(), C_i: ti.template()):
        """Return First Piola–Kirchhoff stress."""
        E = self.green_E(F_i)
        trE = self.trace(E)
        D = self.D_from_C(C_i)
        trD = self.trace(D)

        # Elastic and viscous parts
        S_el = 2.0 * self.mu_0[None] * self.dev(E) + self.lambda_0[None] * trE * self.I()
        S_vi = 2.0 * self.eta_shear[None] * self.dev(D) + self.eta_bulk[None] * trD * self.I()

        # Convert 2nd PK → 1st PK
        return F_i @ (S_el + S_vi)

    @ti.func
    def pk1_components(self, F_i: ti.template(), C_i: ti.template()):
        """Return separate elastic and viscous PK1 for debugging."""
        E = self.green_E(F_i)
        D = self.D_from_C(C_i)
        trE = self.trace(E)
        trD = self.trace(D)

        S_el = 2.0 * self.mu_0[None] * self.dev(E) + self.lambda_0[None] * trE * self.I()
        S_vi = 2.0 * self.eta_shear[None] * self.dev(D) + self.eta_bulk[None] * trD * self.I()

        return F_i @ S_el, F_i @ S_vi
