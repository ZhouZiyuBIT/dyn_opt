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

def RK4(f_c:ca.Function, dt):
    M = 2
    DT = dt/M
    X = ca.SX.sym('X', f_c.size1_in(0))
    U = ca.SX.sym('U', f_c.size1_in(1))
    X0 = X
    for _ in range(M):
        k1 = DT*f_c(X,        U)
        k2 = DT*f_c(X+0.5*k1, U)
        k3 = DT*f_c(X+0.5*k2, U)
        k4 = DT*f_c(X+k3,     U)
        X = X+(k1+2*k2+2*k3+k4)/6
    F = ca.Function('F', [X0, U], [X])
    return F

class Quadrotor(object):
    def __init__(self, cfg_f):
        
        self._G = 9.81
        self._D = np.diag([0.2, 0.2, 0.7])
        self._v_xy_max = ca.inf
        self._v_z_max = ca.inf
        self._omega_xy_max = 12
        self._omega_z_max = 6
        self._a_z_max = 0
        self._a_z_min = -17

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
    
    def ddynamics(self, dt):
        # px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        # vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        # qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), ca.SX.sym('qz')
        
        # az_B = ca.SX.sym('az_B')
        # wx, wy, wz = ca.SX.sym('wx'), ca.SX.sym('wy'), ca.SX.sym('wz')

        # X = ca.vertcat(px, py, pz,
        #                vx, vy, vz,
        #                qw, qx, qy, qz)
        # U = ca.vertcat(az_B, wx, wy, wz)

        # fdrag =rotate_quat(ca.veccat(qw,qx,qy,qz) ,self._D@rotate_quat(ca.veccat(qw,-qx,-qy,-qz), ca.veccat(vx,vy,vz)))

        # X_dot = ca.vertcat(
        #     vx,
        #     vy,
        #     vz,
        #     2 * (qw * qy + qx * qz) * az_B - fdrag[0],
        #     2 * (qy * qz - qw * qx) * az_B - fdrag[1],
        #     (qw * qw - qx * qx - qy * qy + qz * qz) * az_B + self._G - fdrag[2],
        #     0.5 * (-wx * qx - wy * qy - wz * qz),
        #     0.5 * (wx * qw + wz * qy - wy * qz),
        #     0.5 * (wy * qw - wz * qx + wx * qz),
        #     0.5 * (wz * qw + wy * qx - wx * qy)
        # )
        # 
        f = self.dynamics()
        return RK4(f,dt)
    
    def ddynamics2(self):
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), ca.SX.sym('qz')
        dt = ca.SX.sym('dt')
        
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
        X_next = X+X_dot*dt
        X_next[6:10] = X_next[6:10]/ca.sqrt(X_next[6:10].T@X_next[6:10])
        fx = ca.Function('f', [X, U, dt], [X_next], ['X', 'U', 'dt'], ['X_next'])
        return fx

class Quadrotor2(object):
    def __init__(self, cfg_f):
        
        self._m = 1.0         # total mass
        self._arm_l = 0.23    # arm length
        self._c_tau = 0.0133  # torque constant
        
        self._G = 9.81
        self._J = np.diag([0.01, 0.01, 0.02])     # inertia
        self._J_inv = np.linalg.inv(self._J)
        self._D = np.diag([0.6, 0.6, 0.6])
        
        self._v_xy_max = 5
        self._v_z_max = 5
        self._omega_xy_max = 1
        self._omega_z_max = 1
        self._T_max = 4.179
        self._T_min = 0

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
                      -1,-1,-1,-1,
                      -self._omega_xy_max, -self._omega_xy_max, -self._omega_z_max]
        self._X_ub = [ca.inf, ca.inf, ca.inf,
                      self._v_xy_max, self._v_xy_max, self._v_z_max,
                      1,1,1,1,
                      self._omega_xy_max, self._omega_xy_max, self._omega_z_max]

        self._U_lb = [self._T_min, self._T_min, self._T_min, self._T_min]
        self._U_ub = [self._T_max, self._T_max, self._T_max, self._T_max]

#
#   T1    T3
#     \  /
#      \/
#      /\
#     /  \
#   T4    T2
#

    def dynamics(self):
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), ca.SX.sym('qz')
        wx, wy, wz = ca.SX.sym('wx'), ca.SX.sym('wy'), ca.SX.sym('wz')
        
        T1, T2, T3, T4 = ca.SX.sym('T1'), ca.SX.sym('T2'), ca.SX.sym('T3'), ca.SX.sym('T4')

        taux = self._arm_l/np.sqrt(2)*(T1+T4-T2-T3)
        tauy = self._arm_l/np.sqrt(2)*(T1+T3-T2-T4)
        tauz = self._c_tau*(T3+T4-T1-T2)
        thrust = (T1+T2+T3+T4)
        
        tau = ca.veccat(taux, tauy, tauz)
        w = ca.veccat(wx, wy, wz)
        w_dot = self._J_inv@( tau - ca.cross(w,self._J@w) )

        fdrag =rotate_quat(ca.veccat(qw,qx,qy,qz) ,self._D@rotate_quat(ca.veccat(qw,-qx,-qy,-qz), ca.veccat(vx,vy,vz)))

        X_dot = ca.vertcat(
            vx,
            vy,
            vz,
            2 * (qw * qy + qx * qz) * (-thrust/self._m) - fdrag[0],
            2 * (qy * qz - qw * qx) * (-thrust/self._m) - fdrag[1],
            (qw * qw - qx * qx - qy * qy + qz * qz) * (-thrust/self._m) + self._G - fdrag[2],
            0.5 * (-wx * qx - wy * qy - wz * qz),
            0.5 * (wx * qw + wz * qy - wy * qz),
            0.5 * (wy * qw - wz * qx + wx * qz),
            0.5 * (wz * qw + wy * qx - wx * qy),
            w_dot[0],
            w_dot[1],
            w_dot[2]
        )

        X = ca.vertcat(px, py, pz,
                       vx, vy, vz,
                       qw, qx, qy, qz,
                       wx, wy, wz)
        U = ca.vertcat(T1, T2, T3, T4)

        fx = ca.Function('f', [X, U], [X_dot], ['X', 'U'], ['X_dot'])
        return fx
    
    def ddynamics(self, dt):
        f = self.dynamics()
        return RK4(f,dt)
    
    def ddynamics2(self):
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), ca.SX.sym('qz')
        dt = ca.SX.sym('dt')
        
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
        X_next = X+X_dot*dt
        X_next[6:10] = X_next[6:10]/ca.sqrt(X_next[6:10].T@X_next[6:10])
        fx = ca.Function('f', [X, U, dt], [X_next], ['X', 'U', 'dt'], ['X_next'])
        return fx

a, b, c = ca.SX.sym('a'), ca.SX.sym('b'), ca.SX.sym('c')        
qw = ca.cos(a)
qx = ca.sin(a)*ca.cos(b)*ca.cos(c)
qy = ca.sin(a)*ca.cos(b)*ca.sin(c)
qz = ca.sin(a)*ca.sin(c)

if __name__ == "__main__":
    quad = Quadrotor('./gate/quad.yaml')
    print(quad.dynamics())
