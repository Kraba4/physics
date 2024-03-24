import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.widgets import Slider

class Constraint:
    def __init__(self, index1, index2, dist) -> None:
        self.index1 = index1
        self.index2 = index2
        self.dist = dist
        self.error = [0.8, 0.95, 0.975, 1.0]
        self.e = 3

iters = 10000
fig = plt.figure(figsize=(8,6))
bx = fig.add_subplot(111, projection='3d')

particles = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            position = np.array([i, j, k], dtype='float')
            position -= 0.5
            particles.append(position)

particles = np.array(particles)
alpha = 1
rotateMatrix = np.array([[1 ,            0,              0],
                         [0, np.cos(alpha), -np.sin(alpha)],
                         [0, np.sin(alpha), np.cos(alpha)]])
particles = particles @ rotateMatrix
particlesOld = particles.copy()

lines = []
constraints = []
def addConstraints(i, j):
    constraints.append(Constraint(i, j, np.linalg.norm(particles[i] - particles[j])))
    line = Line3D([particles[i, 0], particles[j, 0]], [particles[i, 1], particles[j, 1]], [particles[i, 2], particles[j, 2]],
                    color = np.array([0., 0.8, 0.]))
    lines.append(line)
    bx.add_artist(line)

for i in range(len(particles)):
    for j in range(i+1, len(particles)):
        constraints.append(Constraint(i, j, np.linalg.norm(particles[i] - particles[j])))
        line = Line3D([particles[i, 0], particles[j, 0]], [particles[i, 1], particles[j, 1]], [particles[i, 2], particles[j, 2]],
                      color = np.array([0., 0.8, 0.]))
        lines.append(line)
        bx.add_artist(line)

ticks = np.arange(-2.0, 2.0, 0.5)
r = 1.5
bx.set(xlim=(-r, r), ylim=(-r, r), zlim=(-r, r))
bx.set_aspect('equal')
dots = bx.scatter(particles[:, 0], particles[:, 1], particles[:, 2], c='black', s=100)
a = np.array([0, 0, -1])
timeStep = 0.05
def verlet():
    global particles, particlesOld, a, timeStep
    for i in range(len(particles)):
        temp = particles[i].copy()
        particles[i] += particles[i] - particlesOld[i] + a * timeStep * timeStep
        temp[0] -= 0.005
        particlesOld[i] = temp


low = np.array([-r, -r, -r])
up = np.array([r, r, r])
NUM_ITER = 5
def satisfyConstraints():
    flags = [True] * len(constraints)
    for i in range(NUM_ITER):
        for p in particles:
            p[0] = min(max(p[0], low[0]), up[0])
            p[1] = min(max(p[1], low[1]), up[1])
            p[2] = min(max(p[2], low[2]), up[2])
        for j, c in enumerate(constraints):
            p1 = particles[c.index1]
            p2 = particles[c.index2]
            dist = np.linalg.norm(p2 - p1)
            diff = c.dist - dist
            dir = (p2 - p1) / dist
            p1 -= dir * diff * 0.5
            p2 += dir * diff * 0.5
             

        
def anim(i):
    global dots, lines
    verlet()
    satisfyConstraints()
    for i in range(len(constraints)):
        p1 = particles[constraints[i].index1]
        p2 = particles[constraints[i].index2]
        lines[i].set_data_3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
    
    dots._offsets3d = (particles[:, 0], particles[:, 1], particles[:, 2])
    return dots, lines

ani = FuncAnimation(fig, anim,
                               frames=100000, interval=1, blit=False)
# ani.save('rigid_body.gif')
plt.show()

