import numpy as np
import time

from KFP import KFP

from opt import nlp_solve, reset_x0

class Gaussian(object):
    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov
        self._dim = self._mean.shape[-1]
        self._k = 1/( np.power(2*np.pi, self._dim/2)* np.sqrt(np.linalg.norm(self._cov)) )
    
    def reset(self, mean, cov):
        self._mean = mean
        self._cov = cov
        self._dim = self._mean.shape[-1]
        self._k  = 1/( np.power(2*np.pi, self._dim/2)*np.sqrt(np.linalg.norm(self._cov)) )
    
    def __call__(self, x):
        e = x-self._mean
        
        ee = e*(np.linalg.inv(self._cov)@e)
        ee = np.sum(ee, axis=0)
        return self._k*np.exp(-ee/2)

class DynObstacle():
    def __init__(self, trj:np.array, dt=0.1):
        self._dt = dt
        self._trj = trj
        self._total_steps = trj.shape[-1]
        
        self._step = 0
        self.pos = trj[:,[0]]

    def update(self):
        self.pos = self._trj[:, [self._step]]
        
        self._step += 1
        if self._step >= self._total_steps:
            self._step = 0

class ProbabilityMap(object):
    def __init__(self, size=(5,5)):
        self._size = size
        self._obstacles = []
        self._KFPs = []

        self.grid_map = np.zeros([100, 100])
        x = np.linspace(0, self._size[0], num=100)
        y = np.linspace(0, self._size[1], num=100)
        X, Y = np.meshgrid(x, y)
        X = X.reshape([1, -1])
        Y = Y.reshape([1, -1])
        self._XY = np.concatenate([X, Y], axis=0)
    
    def add_obstacle(self, obs):
        self._obstacles.append(obs)
        
        dt = 0.1
        state_dim = 6
        input_dim = 0
        obs_dim = 2
        F = np.array([[1, 0, dt,  0,  0,  0],
                      [0, 1,  0, dt,  0,  0],
                      [0, 0,  1,  0, dt,  0],
                      [0, 0,  0,  1,  0, dt],
                      [0, 0,  0,  0, 1,   0],
                      [0, 0,  0,  0, 0,   1]])
        G = np.zeros(1)
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]])
        Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        R = np.diag([0.01, 0.01])
        X0 = np.array([0,0,0,0,0,0]).reshape(-1,1)
        X0[:2] = obs.pos
        P0 = np.diag([1,1,1,1,1,1])
        kfp = KFP(state_dim, input_dim, obs_dim, F, G, H, Q, R, X0, P0)
        self._KFPs.append(kfp)

    def get_pred(self):
        X, P = self._KFPs[0].predict(N=10)
        p = np.zeros(6*10)
        for i in range(10):
            p[i*6:2+i*6] = X[:2,i]
            # p[2+i*6:6+i*6] = np.array([1,0 ,0,1])
            p[2+i*6:6+i*6] = P[:2,:2,i].T.reshape(-1)
            # print(P[:2,:2,i])
        return p
            
    def map_update(self):
        self.grid_map = np.zeros([100, 100])
        for obs,kfp in zip(self._obstacles, self._KFPs):
            obs.update()
            kfp.propagate()
            kfp.update(obs.pos)
            print(kfp._P[:2,:2])
            distri = Gaussian(kfp._X[:2], kfp._P[:2,:2])
            self.grid_map += distri(self._XY).reshape([100, 100])

map = ProbabilityMap()

t = np.linspace(0, 2*np.pi, 60)
x = np.sin(t+2.9)*2 + 2.4
y = np.ones(x.shape[0])*2
x = x.reshape([1, -1])
y = y.reshape([1, -1])
trj = np.concatenate([x,y], axis=0)
obs = DynObstacle(trj)
map.add_obstacle(obs)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

sim_dt = 0.1
def sim_run(T=50):
    t = 0
    x_init = np.array([1,0,0,0])
    x_goal = np.array([2.5, 4])
    p = np.zeros(4+6*10+2)
    
    while t<T:
        map.map_update()

        p[:4] = x_init
        p[4:4+6*10] = map.get_pred()
        p[4+6*10:] = x_goal
        t1 = time.time()
        reset_x0() # hard reboot
        X, traj = nlp_solve(p)
        t2 = time.time()
        print(t2-t1)
        x_init = X[2:6]

        t += sim_dt
        data = {
            'map': map,
            'traj': traj
        }
        yield data

sim = sim_run()

fig = plt.figure()
ax = fig.add_axes([0.03, 0.05, 0.94, 0.9])
line, = ax.plot([], [])
robot, = ax.plot([],[],'o')
im = ax.imshow(map.grid_map, origin='lower', extent=(0,5, 0,5), vmin=0, vmax=1)

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
