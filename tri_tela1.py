import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# center_of_gravity = np.array([0., 2., 0.])
# center_of_gravity2 = np.array([0., 6., 0.])
G = 1.
dt = 0.01


def gravity(position, center_of_gravity, mass):
    to_center = center_of_gravity - position
    r = np.linalg.norm(to_center)
    result = to_center * ((G * mass) / (r**3))
    return result

class Satelites:
    def __init__(self, position, velocity, mass, color) -> None:
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.color = color

# s1 = Satelites(np.array([0., 0., 0.]), np.array([0, 0, 2.0]), 'semi', np.array([1., 0., 0]))
s1 = Satelites(np.array([0., 5., 0.]), np.array([1, 1, 1.0]), 0, np.array([1., 0., 0]))
s2 = Satelites(np.array([0., 0., 5.]), np.array([0., 0, 0.0]), 10, np.array([0., 0., 1.]))
s3 = Satelites(np.array([0., 10, 0.]), np.array([0., 0, 0.0]), 10, np.array([0., 1., 0.]))

planets = [s1, s2, s3]
iters = 100000
coords_s = np.ndarray([len(planets), 3, iters])

def v(x, dt, vn, planets, me):
    sum_gravity = 0
    for p in planets:
        if p != me:
            sum_gravity += gravity(x, p.position, p.mass)
    return vn + sum_gravity * dt

def a(v, dt, xn, planets, me):
    sum_gravity = 0
    for p in planets:
        if p != me:
            sum_gravity += gravity(xn, p.position, p.mass)
    return sum_gravity # + gravity(xn + v*dt) * dt

def anim(planets, i = None):
    s = planets[0]
    xn = s.position.copy()
    vn = s.velocity.copy()
    s.position = xn + dt * v(xn, dt, vn, planets, s)
    s.velocity = vn + dt * a(vn, 0, xn, planets, s)
    # for s in planets:
    #     xn = s.position.copy()
    #     vn = s.velocity.copy()
    #     s.position = xn + dt * v(xn, dt, vn, planets, s)
    #     s.velocity = vn + dt * a(vn, 0, xn, planets, s)
    if i != None:
        for index, s in enumerate(planets):
            coords_s[index, :, i] = s.position

for i in range(iters):
    anim(planets, i)

fig = plt.figure(figsize=(8,6))
bx = fig.add_subplot(111, projection='3d')
bx.set(xlim=(-7, 8), ylim=(-3, 12))
for i in range(len(coords_s)):
    bx.plot(coords_s[i, 0, :], coords_s[i, 1, :], coords_s[i, 2, :], color = planets[i].color)


for p in planets:
    bx.scatter(p.position[0], p.position[1], p.position[2])

fig.show()
input()
