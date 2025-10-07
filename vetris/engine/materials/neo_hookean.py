import taichi as ti


@ti.data_oriented
class NeoHookeanMaterial:

    def __init__(self, cfg):
        
        self.E         = cfg.youngs_modulus
        self.nu        = cfg.poisson_ratio

        # set Lamé parameters
        self.mu_0      = self.E / (2 * (1 + self.nu))
        self.lambda_0  = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))


    @ti.func
    def neo_hookean_stress(self,F_i):
        J_det = F_i.determinant()
        FinvT = F_i.inverse().transpose()
        return self.mu_0 * (F_i - FinvT) + self.lambda_0 * ti.log(J_det) * FinvT
