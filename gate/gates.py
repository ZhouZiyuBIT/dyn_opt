import numpy as np

class Gates(object):
    def __init__(self):
        gp = [[14.8715, -0.7906, -2.6],
              [37.0, -12, -0.5],
              [68, -13, 0],
              [95, -8, 0.5],
              [114.5, -36, 0],
              [121.5, -67.7, -3.5],
              [121.5, -96, -7]]
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