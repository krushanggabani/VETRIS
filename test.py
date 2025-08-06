import taichi as ti

ti.init()

@ti.func
def sign(x):
    # returns 1 if x>0, -1 if x<0, 0 if x==0
    return ti.select(x > 0, 1.0, ti.select(x < 0, -1.0, 0.0))

@ti.kernel
def demo():
    for i in range(5):
        x = ti.cast(i - 2, ti.f32)  # values: -2, -1, 0, 1, 2
        s = sign(x)
        print("x =", x, "sign =", s)

demo()
