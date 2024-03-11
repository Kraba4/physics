import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider


G = 1.
dt = 0.5


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

# s1 = Satelites(np.array([0., 0., 0.]), np.array([-1.4, 0, 0.0]), 0, np.array([1., 0., 0]))
s2 = Satelites(np.array([0., 0., 0.]), np.array([4, 0, 0.0]), 50, np.array([0., 0., 1.]))
s3 = Satelites(np.array([0., 320, 0.]), np.array([0., 0, 0.0]), 5000, np.array([0., 1., 0.]))

planets = [ s2, s3]

iters = 10000
coords_s = np.ndarray([len(planets), 3, iters])

def calc_center_mass(planets):
    center_of_mass = np.zeros_like(planets[0].position)
    sum_of_mass = 0
    for p in planets:
        center_of_mass += p.position * p.mass
        sum_of_mass += p.mass
    center_of_mass /= sum_of_mass
    return center_of_mass

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

lagrange_points = np.ndarray([5, 3])

# for i in range(iters):
#     anim(planets, i)

fig = plt.figure(figsize=(8,6))
bx = fig.add_subplot(111)
bx.set(xlim=(-600, 600), ylim=(-200, 900))
# c_a1 = plt.Circle(s1.position, 1, color=s1.color)
# ax.add_artist(c_a1)
c_a2 = plt.Circle(s2.position, 5, color=s2.color)
bx.add_artist(c_a2)
c_a3 = plt.Circle(s3.position, 30, color=s3.color)
bx.add_artist(c_a3)

lp = []
for i in range(5):
    lp.append(plt.Circle(lagrange_points[i, :], 2, color=np.array([0., 0., 0.])))
    bx.add_artist(lp[i])

# lp, = bx.scatter(lagrange_points[:, 0], lagrange_points[:, 1], c='black')

def anim(i):
    global c_a2, c_a3, planets, lp
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
    c_a2.center = s2.position
    c_a3.center = s3.position
    center_mass = calc_center_mass(planets)
    ax_forward = s2.position - center_mass
    ax_forward = ax_forward / np.linalg.norm(ax_forward)
    ax_right = np.cross(ax_forward, np.array([0,0,1.0]))
    ax_right = ax_right / np.linalg.norm(ax_right)
    R = np.linalg.norm(s2.position - s3.position)
    alpha = s2.mass / (s3.mass + s2.mass)
    beta = (s3.mass - s2.mass) / (s3.mass + s2.mass)
    r1 = np.array([R * (1 - (alpha/3)**(1/3)), 0])
    r2 = np.array([R * (1 + (alpha/3)**(1/3)), 0])
    r3 = np.array([-R * (1 + (5/12)*alpha), 0])
    r4 = (R/2*beta, (3**(1/2)) * R / 2)
    r5 = (R/2*beta, -(3**(1/2)) * R / 2)
    lagrange_points[0, :] = center_mass + ax_forward * r1[0] + ax_right * r1[1]
    lagrange_points[1, :] = center_mass + ax_forward * r2[0] + ax_right * r2[1]
    lagrange_points[2, :] = center_mass + ax_forward * r3[0] + ax_right * r3[1]
    lagrange_points[3, :] = center_mass + ax_forward * r4[0] + ax_right * r4[1]
    lagrange_points[4, :] = center_mass + ax_forward * r5[0] + ax_right * r5[1]
    for i in range(5):
        lp[i].center = lagrange_points[i, :]
    return c_a2, c_a3, lp[0], lp[1], lp[2], lp[3], lp[4]

anim = FuncAnimation(fig, anim,
                               frames=100000, interval=1, blit=True)
fig.show()

input()
