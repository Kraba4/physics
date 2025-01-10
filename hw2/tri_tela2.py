import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# center_of_gravity = np.array([0., 2., 0.])
# center_of_gravity2 = np.array([0., 6., 0.])
# gsgsgs
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
s1 = Satelites(np.array([0., 5., 0.]), np.array([1, 1, 0.0]), 0, np.array([1., 0., 0]))
s2 = Satelites(np.array([0., 0., 0.]), np.array([0., 0, 0.0]), 10, np.array([0., 0., 1.]))
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


fig = plt.figure(figsize=(8,6))
bx = fig.add_subplot(111)
bx.set(xlim=(-7, 11), ylim=(-3, 15))

c_a1 = plt.Circle(s1.position, 0.1, color=s1.color)
bx.add_artist(c_a1)

# for i in range(len(coords_s)):
#     bx.plot(coords_s[i, 0, :], coords_s[i, 1, :], coords_s[i, 2, :], color = planets[i].color)

for p in planets[1:]:
    bx.scatter(p.position[0], p.position[1])

coords_x = [s1.position[0]]
coords_y = [s1.position[1]] 
tr,  = bx.plot(coords_x, coords_y, color = np.array([0.9, 0.1, 0]))
def anim(i):
    global c_a1, planets, coords_x, coords_y, tr, bx
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
    c_a1.center = s.position[:2]
    coords_x.append(s.position[0])
    coords_y.append(s.position[1])
    tr.set_xdata(coords_x)
    tr.set_ydata(coords_y)
    return tr, c_a1

ani = FuncAnimation(fig, anim,
                               frames=1000, interval=0.5, blit=True)
fig.show()
input()
