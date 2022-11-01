import numpy as np

# from opt import nlp_solve

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
        self.pos = trj[:,0]
        self.pos_cov = np.array([[0.1, 0],
                                  [0, 0.1]])
        self.distribution = Gaussian(self.pos.reshape([-1,1]), self.pos_cov)

    def update(self):
        self.pos = self._trj[:, self._step]
        self.distribution.reset(self.pos.reshape([-1, 1]), self.pos_cov)
        
        self._step += 1
        if self._step >= self._total_steps:
            self._step = 0

class ProbabilityMap(object):
    def __init__(self, size=(5,5)):
        self._size = size
        self._obstacles = []

        self.grid_map = np.zeros([100, 100])

        x = np.linspace(0, self._size[0], num=100)
        y = np.linspace(0, self._size[1], num=100)
        X, Y = np.meshgrid(x, y)
        X = X.reshape([1, -1])
        Y = Y.reshape([1, -1])
        self._XY = np.concatenate([X, Y], axis=0)
    
    def add_obstacle(self, obs):
        self._obstacles.append(obs)

    def map_update(self):
        self.grid_map = np.zeros([100, 100])
        for obs in self._obstacles:
            obs.update()
            self.grid_map += obs.distribution(self._XY).reshape([100, 100])

map = ProbabilityMap()
mean = np.array([2.5,2.5]).reshape([-1,1])
cov = np.array([[0.4, 0.3], 
                [0.1, 1]])

t = np.linspace(0, 2*np.pi, 20)
x = np.sin(t)*2 + 2.5
y = np.ones(x.shape[0])*2
x = x.reshape([1, -1])
y = y.reshape([1, -1])
trj = np.concatenate([x,y], axis=0)
obs = DynObstacle(trj)
map.add_obstacle(obs)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

sim_dt = 0.1
def sim_run(T=5):
    global map
    t = 0

    while t<T:
        map.map_update()
        t += sim_dt
        yield map

sim = sim_run()

fig = plt.figure()
ax = fig.add_axes([0.03, 0.05, 0.94, 0.9])
im = ax.imshow(map.grid_map, origin='lower', extent=(0,5, 0,5), animated=False)
line, = ax.plot([1,2,3], [1,2,3])

def init():
    im = ax.imshow(map.grid_map, origin='lower', extent=(0,5, 0,5), animated=False)
    line.set_data([1,2,3], [1,2,3])
    return [im, line]
def update_plot(*args):
    map = next(sim)
    im.set_data(map.grid_map)
    line.set_data([1,2,3], [1,2,3])
    print('111')
    return [im, line]

ani = animation.FuncAnimation(fig, update_plot, init_func=init, interval=100, blit=True, repeat=False)
# fig.show()
plt.show()
