import numpy as np

class Gates(object):
    def __init__(self):
        gp = [[14.87, -0.79,  -2.74],
              [37.68, -12.27, -1.02],
              [67.42, -12.94, -0.30],
              [94.44, -8.05,   0.02],
              [113.95,-35.70, -0.34],
              [121.77,-67.73, -3.83],
              [121.80,-96.00, -7.39]]
        self._num = len(gp)
        self._g_pos_pre = np.array(gp)
        self._g_pos_pre_cov = np.ones(self._g_pos_pre.shape)*3
        self._g_pos = self._g_pos_pre + np.random.uniform(-0.5, 0.5, size=self._g_pos_pre.shape)
    
    def update(self):
        self._g_pos = self._g_pos_pre + np.random.uniform(-0.5, 0.5, size=self._g_pos_pre.shape)
        pass

class FreeGate(object):
    def __init__(self, p0):
        self._p0 = p0
        self._G = np.array([0,0,9.81])

        self._p = self._p0
        vy = np.random.random()*4+1
        vz = np.random.random()+4
        self._v = np.array([0, vy, vz])

    def reset(self):
        self._p = self._p0
        vy = np.random.random()*4+1
        vz = np.random.random()+4
        self._v = np.array([0, vy, vz])