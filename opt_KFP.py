import numpy as np
import time

from opt import nlp_solve, reset_x0
from dynamics_map import DynObstacle, ProbabilityMap
from KFP import KFP

map = ProbabilityMap()

t = np.linspace(0, 2*np.pi, 60)
x = np.sin(t+2.9)*2 + 2.4
# y = np.sin(t+2.9)*0 + 2.5
y = np.ones(x.shape[0])*2
x = x.reshape([1, -1])
y = y.reshape([1, -1])
trj = np.concatenate([x,y], axis=0)
obs = DynObstacle(trj)
map.add_obstacle(obs)

sim_dt = 0.1
def sim_run(T=50):
    t = 0
    x_init = np.array([1, 0, 0, 0])
    x_goal = np.array([2.5, 4])
    p = np.zeros(4+6*10+2)
    
    while t<T:
        map.map_update()

        p[:4] = x_init
        p[4:4+6*10] = map.get_pred()
        p[4+6*10:] = x_goal
        t1 = time.time()
        reset_x0()
        X, traj = nlp_solve(p)
        t2 = time.time()
        print(t2-t1)
        print(map._KFPs[0]._Q)
        x_init = X[2:6]

        t += sim_dt
        data = {
            'map': map,
            'traj': traj
        }
        yield data
sim = sim_run()

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_axes([0.03, 0.05, 0.94, 0.9])
line, = ax.plot([], [])
robot, = ax.plot([], [], 'o')
im = ax.imshow(map.grid_map, origin='lower', extent=(0,5,0,5), vmin=0, vmax=1)

def update_plot(*arg):
    data = next(sim)
    map = data['map']
    traj = data['traj']

    im.set_data(map.grid_map)
    line.set_data(traj[0,:], traj[1,:])
    robot.set_data(traj[0,0], traj[1,0])
    return [im, line, robot]

ani = animation.FuncAnimation(fig, update_plot, interval=100, save_count=30, blit=True, repeat=False)
# ani.save('./gif/animation.gif', writer='imagemagick')
plt.show()