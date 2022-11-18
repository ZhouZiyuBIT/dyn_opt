import numpy as np
import casadi as ca
import yaml

class Quadrotor(object):
    def __init__(self, cfg_f):
        
        self._G = 9.81
        self._v_xy_max = 20
        self._v_z_max = 20
        self._omega_xy_max = 6
        self._omega_z_max = 3
        self._a_z_max = 0
        self._a_z_min = -30

        with open(cfg_f, 'r') as f:
            self._cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        if 'omega_xy_max' in self._cfg:
            self._omega_xy_max = self._cfg['omega_xy_max']
        if 'omega_z_max' in self._cfg:
            self._omega_z_max = self._cfg['omega_z_max']
        if 'G' in self._cfg:
            self._G = self._cfg['G']
        
        self._X_lb = [-ca.inf, -ca.inf, -ca.inf,
                      -self._v_xy_max, -self._v_xy_max, -self._v_z_max,
                      -1,-1,-1,-1]
        self._X_ub = [ca.inf, ca.inf, ca.inf,
                      self._v_xy_max, self._v_xy_max, self._v_z_max,
                      1,1,1,1]
        self._U_lb = [self._a_z_min, -self._omega_xy_max, -self._omega_xy_max, -self._omega_z_max]
        self._U_ub = [self._a_z_max,  self._omega_xy_max,  self._omega_xy_max,  self._omega_z_max]
        
    def dynamics(self):
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), ca.SX.sym('qz')
        
        az_B = ca.SX.sym('az_B')
        wx, wy, wz = ca.SX.sym('wx'), ca.SX.sym('wy'), ca.SX.sym('wz')

        X = ca.vertcat(px, py, pz,
                       vx, vy, vz,
                       qw, qx, qy, qz)
        U = ca.vertcat(az_B, wx, wy, wz)

        X_dot = ca.vertcat(
            vx,
            vy,
            vz,
            2 * (qw * qy + qx * qz) * az_B,
            2 * (qy * qz - qw * qx) * az_B,
            (qw * qw - qx * qx - qy * qy + qz * qz) * az_B + self._G,
            0.5 * (-wx * qx - wy * qy - wz * qz),
            0.5 * (wx * qw + wz * qy - wy * qz),
            0.5 * (wy * qw - wz * qx + wx * qz),
            0.5 * (wz * qw + wy * qx - wx * qy)
        )

        fx = ca.Function('f', [X, U], [X_dot], ['X', 'U'], ['X_dot'])
        return fx


if __name__ == "__main__":
    quad = Quadrotor('./gate/quad.yaml')
    print(quad.dynamics())
