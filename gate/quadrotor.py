import numpy as np
import casadi as ca
import yaml

# Quaternion Multiplication
def quat_mult(q1,q2):
    ans = ca.vertcat(q2[0,:] * q1[0,:] - q2[1,:] * q1[1,:] - q2[2,:] * q1[2,:] - q2[3,:] * q1[3,:],
           q2[0,:] * q1[1,:] + q2[1,:] * q1[0,:] - q2[2,:] * q1[3,:] + q2[3,:] * q1[2,:],
           q2[0,:] * q1[2,:] + q2[2,:] * q1[0,:] + q2[1,:] * q1[3,:] - q2[3,:] * q1[1,:],
           q2[0,:] * q1[3,:] - q2[1,:] * q1[2,:] + q2[2,:] * q1[1,:] + q2[3,:] * q1[0,:])
    return ans

# Quaternion-Vector Rotation
def rotate_quat(q1,v1):
    ans = quat_mult(quat_mult(q1, ca.vertcat(0, v1)), ca.vertcat(q1[0,:],-q1[1,:], -q1[2,:], -q1[3,:]))
    return ca.vertcat(ans[1,:], ans[2,:], ans[3,:]) # to covert to 3x1 vec

class Quadrotor(object):
    def __init__(self, cfg_f):
        
        self._G = 9.81
        self._D = np.diag([0.3, 0.3, 1.0])
        self._v_xy_max = ca.inf
        self._v_z_max = ca.inf
        self._omega_xy_max = 12
        self._omega_z_max = 6
        self._a_z_max = 0
        self._a_z_min = -20

        with open(cfg_f, 'r') as f:
            self._cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        # if 'omega_xy_max' in self._cfg:
        #     self._omega_xy_max = self._cfg['omega_xy_max']
        # if 'omega_z_max' in self._cfg:
        #     self._omega_z_max = self._cfg['omega_z_max']
        # if 'G' in self._cfg:
        #     self._G = self._cfg['G']
        
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

        fdrag =rotate_quat(ca.veccat(qw,qx,qy,qz) ,self._D@rotate_quat(ca.veccat(qw,-qx,-qy,-qz), ca.veccat(vx,vy,vz)))

        X_dot = ca.vertcat(
            vx,
            vy,
            vz,
            2 * (qw * qy + qx * qz) * az_B - fdrag[0],
            2 * (qy * qz - qw * qx) * az_B - fdrag[1],
            (qw * qw - qx * qx - qy * qy + qz * qz) * az_B + self._G - fdrag[2],
            0.5 * (-wx * qx - wy * qy - wz * qz),
            0.5 * (wx * qw + wz * qy - wy * qz),
            0.5 * (wy * qw - wz * qx + wx * qz),
            0.5 * (wz * qw + wy * qx - wx * qy)
        )

        fx = ca.Function('f', [X, U], [X_dot], ['X', 'U'], ['X_dot'])
        return fx

a, b, c = ca.SX.sym('a'), ca.SX.sym('b'), ca.SX.sym('c')        
qw = ca.cos(a)
qx = ca.sin(a)*ca.cos(b)*ca.cos(c)
qy = ca.sin(a)*ca.cos(b)*ca.sin(c)
qz = ca.sin(a)*ca.sin(c)

if __name__ == "__main__":
    quad = Quadrotor('./gate/quad.yaml')
    print(quad.dynamics())
