import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.widgets import Slider
# from joblib import Parallel, delayed

class Constraint:
    def __init__(self, index1, index2, dist) -> None:
        self.index1 = index1
        self.index2 = index2
        self.dist = dist

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
velocities = np.zeros_like(particles)
velocities = np.array([1.5, 0, 0])
alpha = 1
rotateMatrix = np.array([[1 ,            0,              0],
                         [0, np.cos(alpha), -np.sin(alpha)],
                         [0, np.sin(alpha), np.cos(alpha)]])
particles = particles @ rotateMatrix
tempParticles = particles.copy()

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
timeStep = 0.1

def update_velocities(dt):
    global velocities
    velocities += dt * a

damp_coef = 0.993
def damp_velocities():
    global velocities
    velocities *= damp_coef

def update_temp_particles(dt):
    global tempParticles
    tempParticles = particles + dt * velocities

low = np.array([-r, -r, -r])
up = np.array([r, r, r])

def process_collisions(i):
    global tempParticles
    p = tempParticles[i]
    pn = np.empty_like(p)
    pn[0] = min(max(p[0], low[0]), up[0])
    pn[1] = min(max(p[1], low[1]), up[1])
    pn[2] = min(max(p[2], low[2]), up[2])
    return pn - tempParticles[i]

def process_constraints(i):
    c = constraints[i]
    p1 = tempParticles[c.index1]
    p2 = tempParticles[c.index2]
    dist = np.linalg.norm(p2 - p1)
    diff = c.dist - dist
    dir = (p2 - p1) / dist
    p1d = -dir * diff * 0.5
    p2d = dir * diff * 0.5
    return p1d, p2d

NUM_ITER = 15
def satisfyConstraints():
    global velocities, particles
    miniDt = timeStep / NUM_ITER
    for k in range(NUM_ITER):
        update_velocities(miniDt)
        damp_velocities()
        update_temp_particles(miniDt)
        # точек мало и запуск паралленых процессов медленнее, чем если в одном решить
        # results1 = Parallel(n_jobs=4)(delayed(process_collisions)(i) for i in range(len(tempParticles)))
        # results2 = Parallel(n_jobs=4)(delayed(process_constraints)(i) for i in range(len(constraints)))

        # psevdo parallel
        results1 = [process_collisions(i) for i in range(len(tempParticles))]
        results2 = [process_constraints(i) for i in range(len(constraints))]

        # можно конечно все суммы сделать параллельными за O(log) но я не хочу страдать (не ставьте 2)
        for i in range(len(results1)):
            tempParticles[i] += results1[i]
        
        averageCoef = 8.0 # без него симуляция взрывается
        for i in range(len(results2)):
            ind1d, ind2d = results2[i]
            c = constraints[i]
            tempParticles[c.index1] += ind1d / averageCoef
            tempParticles[c.index2] += ind2d / averageCoef
        velocities = (tempParticles - particles) / miniDt
        particles = tempParticles

def anim(i):
    global dots, lines
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

