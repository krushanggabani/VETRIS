import taichi as ti

@ti.data_oriented
class NeoHookeanKelvinVoigtMaterial:
    """Neo-Hookean + objective Newtonian dashpot (deviatoric + bulk)."""

    def __init__(self, cfg, dim: int = 2):
        self.dim = int(dim)

        # Make scalars as 0-D Taichi fields (safe inside @ti.func)
        self.mu_0     = ti.field(dtype=ti.f32, shape=())
        self.lambda_0 = ti.field(dtype=ti.f32, shape=())
        self.eta_shear = ti.field(dtype=ti.f32, shape=())
        self.eta_bulk  = ti.field(dtype=ti.f32, shape=())
        self.rate_k    = ti.field(dtype=ti.f32, shape=())
        self.rate_n    = ti.field(dtype=ti.f32, shape=())

        # Read from cfg (assumes cfg is your MPM sub-config)
        E  = float(cfg.youngs_modulus)
        nu = float(cfg.poisson_ratio)
        
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        self.mu_0[None]     = mu
        self.lambda_0[None] = lam
        self.eta_shear[None] = float(getattr(cfg, "shear_viscosity", 5.0))
        self.eta_bulk[None]  = float(getattr(cfg, "bulk_viscosity", 50.0))
        self.rate_k[None]    = float(getattr(cfg, "rate_k", 0.0))
        self.rate_n[None]    = float(getattr(cfg, "rate_n", 1.0))

        

    @ti.func
    def neo_hookean_stress(self, F_i: ti.template()):
        J_det = F_i.determinant()
        FinvT = F_i.inverse().transpose()
        # Use field scalars
        return self.mu_0[None] * (F_i - FinvT) + self.lambda_0[None] * ti.log(J_det) * FinvT

    @ti.func
    def _eta_eff(self, D_norm: ti.f32) -> ti.f32:
        # eta_eff = eta * (1 + k * |D|^n)  (k==0 disables nonlinearity)
        return self.eta_shear[None] * (1.0 + self.rate_k[None] * ti.pow(ti.max(D_norm, 0.0), self.rate_n[None]))

    @ti.func
    def viscous_PK1(self, F_i: ti.template(), C_i: ti.template()):
        # Objective Newtonian: tau = 2*eta*D + zeta*tr(D) I ;   P = J * tau * F^{-T}
        D = 0.5 * (C_i + C_i.transpose())
        D_norm = ti.sqrt(2.0 * (D[0, 0] * D[0, 0] + D[1, 1] * D[1, 1]) + 4.0 * (D[0, 1] * D[0, 1]))
        D_norm = ti.min(D_norm, 5e3)

        eta  = self._eta_eff(D_norm)
        zeta = self.eta_bulk[None]
        I    = ti.Matrix.identity(ti.f32, self.dim)
        tau  = 2.0 * eta * D + zeta * (D[0, 0] + D[1, 1]) * I

        J = F_i.determinant()
        P = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        if J > 1e-3:
            FinvT = F_i.inverse().transpose()
            P = J * (tau @ FinvT)
        return P

    @ti.func
    def pk1_update(self, F_i: ti.template(), C_i: ti.template()):
        P_elastic = self.neo_hookean_stress(F_i)
        P_visc    = self.viscous_PK1(F_i, C_i) 

        return P_elastic + P_visc
