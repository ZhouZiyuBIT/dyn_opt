import casadi as ca
import numpy as np
from gates import Gates
from quadrotor import Quadrotor

class TimeOpt(object):
    def __init__(self, quad:Quadrotor, wp_num):

        self._quad = quad

        self._N_per_wp = 50
        self._wp_num = wp_num
        self._dynamics = quad.dynamics()
        self._X_dim = self._dynamics.size1_in(0)
        self._U_dim = self._dynamics.size1_in(1)
        self._Herizon = self._wp_num*self._N_per_wp

        self._DT = ca.SX.sym('DT', self._wp_num)
        self._X_init = ca.SX.sym('X_init', self._X_dim, 1)
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Herizon)
        self._Us = ca.SX.sym('Us', self._U_dim, self._Herizon)
        self._WP = ca.SX.sym('WP', 3, self._wp_num)

        self._co_R = ca.diag([3, 1.0, 1.0, 2.0])
        self._co_WP = ca.diag([1000,1000,1000])
        self._wp_tol = 0.1

        self._init_option = {
            'verbose': False,
            # 'ipopt.tol': 1e-8,
            # 'ipopt.acceptable_tol': 1e-8,
            'ipopt.max_iter': 1000,
            'ipopt.warm_start_init_point': 'yes',
            # 'ipopt.print_level': 0,
        }
        self._topt_option = {
            'verbose': False,
            'ipopt.tol': 1e-2,
            # 'ipopt.acceptable_tol': 1e-8,
            'ipopt.max_iter': 1000,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 0
        }


    def define_opt(self):

        nlp_g_dyn = []
        self._nlp_lbg_dyn = []
        self._nlp_ubg_dyn = []

        nlp_g_wp = []
        self._nlp_lbg_wp = []
        self._nlp_ubg_wp = []

        nlp_g_quat = []
        self._nlp_lbg_quat = []
        self._nlp_ubg_quat = []
        
        nlp_x_x = []
        self._nlp_lbx_x = []
        self._nlp_ubx_x = []
        
        nlp_x_u = []
        self._nlp_lbx_u = []
        self._nlp_ubx_u = []
        
        nlp_x_time = []
        self._nlp_lbx_time = []
        self._nlp_ubx_time = []
        
        nlp_p_xinit = [self._X_init[:,0]]
        nlp_p_time = []
        nlp_p_wp = []

        nlp_obj_minco = 0
        nlp_obj_time = 0
        nlp_obj_wp = 0
        nlp_obj_quat = 0
        for i in range(self._wp_num):
            if i==0:
                nlp_g_dyn += [ self._Xs[:,0] - self._X_init - self._dynamics( self._X_init, self._Us[:,0] )*self._DT[0] ]
            else:
                nlp_g_dyn += [ self._Xs[:,i*self._N_per_wp] - self._Xs[:,i*self._N_per_wp-1] - self._dynamics( self._Xs[:,i*self._N_per_wp-1], self._Us[:,i*self._N_per_wp] )*self._DT[i] ]
            self._nlp_lbg_dyn += [0 for i in range(self._X_dim)]
            self._nlp_ubg_dyn += [0 for i in range(self._X_dim)]

            nlp_obj_minco += (self._Us[:,i*self._N_per_wp] - [-9.8,0,0,0]).T@self._co_R@(self._Us[:,i*self._N_per_wp] - [-9.8,0,0,0])
            nlp_obj_time += self._DT[i]*self._N_per_wp
            nlp_obj_wp += (self._Xs[:3,(i+1)*self._N_per_wp-1]-self._WP[:,i]).T@self._co_WP@(self._Xs[:3,(i+1)*self._N_per_wp-1]-self._WP[:,i])
            nlp_obj_quat += self._Xs[6:,i*self._N_per_wp].T@self._Xs[6:,i*self._N_per_wp] + 1 - 2*ca.norm_2(self._Xs[6:,i*self._N_per_wp])
            nlp_g_wp += [(self._Xs[:3,(i+1)*self._N_per_wp-1]-self._WP[:,i]).T@(self._Xs[:3,(i+1)*self._N_per_wp-1]-self._WP[:,i])]
            self._nlp_lbg_wp += [0]
            self._nlp_ubg_wp += [self._wp_tol*self._wp_tol]
            nlp_g_quat += [((self._Xs[6:,i*self._N_per_wp]).T@self._Xs[6:,i*self._N_per_wp]) - 1]
            self._nlp_lbg_quat += [-0.1]
            self._nlp_ubg_quat += [ 0.1]
            nlp_x_x += [self._Xs[:,i*self._N_per_wp]]
            self._nlp_lbx_x += self._quad._X_lb
            self._nlp_ubx_x += self._quad._X_ub
            nlp_x_u += [self._Us[:,i*self._N_per_wp]]
            self._nlp_lbx_u += self._quad._U_lb
            self._nlp_ubx_u += self._quad._U_ub
            nlp_x_time += [ self._DT[i] ]
            self._nlp_lbx_time += [0]
            self._nlp_ubx_time += [0.2]
            nlp_p_time += [ self._DT[i] ]
            nlp_p_wp += [self._WP[:,i]]
            for j in range(1, self._N_per_wp):
                nlp_obj_minco += (self._Us[:,i*self._N_per_wp+j] - [-9.8,0,0,0]).T@self._co_R@(self._Us[:,i*self._N_per_wp+j] - [-9.8,0,0,0])
                nlp_obj_quat += self._Xs[6:,i*self._N_per_wp+j].T@self._Xs[6:,i*self._N_per_wp+j] + 1 - 2*ca.norm_2(self._Xs[6:,i*self._N_per_wp+j])
                nlp_g_dyn += [ self._Xs[:,i*self._N_per_wp+j] - self._Xs[:,i*self._N_per_wp+j-1] - self._dynamics( self._Xs[:,i*self._N_per_wp+j-1], self._Us[:,i*self._N_per_wp+j] )*self._DT[i] ]
                self._nlp_lbg_dyn += [0 for i in range(self._X_dim)]
                self._nlp_ubg_dyn += [0 for i in range(self._X_dim)]
                nlp_g_quat += [((self._Xs[6:,i*self._N_per_wp+j]).T@self._Xs[6:,i*self._N_per_wp+j]) - 1]
                self._nlp_lbg_quat += [-0.1]
                self._nlp_ubg_quat += [ 0.1]
                nlp_x_x += [self._Xs[:,i*self._N_per_wp+j]]
                self._nlp_lbx_x += self._quad._X_lb
                self._nlp_ubx_x += self._quad._X_ub
                nlp_x_u += [self._Us[:,i*self._N_per_wp+j]]
                self._nlp_lbx_u += self._quad._U_lb
                self._nlp_ubx_u += self._quad._U_ub


        init_nlp_dect = {
            'f': nlp_obj_minco+nlp_obj_wp,
            'x': ca.vertcat(*(nlp_x_x+nlp_x_u)),
            'p': ca.vertcat(*(nlp_p_xinit+nlp_p_wp+nlp_p_time)),
            'g': ca.vertcat(*(nlp_g_dyn+nlp_g_quat)),
            # 'g': ca.vertcat(*(nlp_g_dyn))
        }
        self._init_solver = ca.nlpsol('init_solver', 'ipopt', init_nlp_dect, self._init_option)

        self._x0 = np.zeros((self._X_dim+self._U_dim)*self._Herizon)
        self._dt = np.zeros(self._wp_num)
        for i in range(self._wp_num):
            self._dt[i] = 0.5
            for j in range(self._N_per_wp):
                self._x0[self._X_dim*(i*self._N_per_wp+j)+6] = 1
                self._x0[self._X_dim*self._Herizon + self._U_dim*(i*self._N_per_wp+j)] = -9.8
        self._xt0 = np.concatenate([self._x0, self._dt])

        topt_nlp_dect = {
            'f': nlp_obj_time,
            'x': ca.vertcat(*(nlp_x_x+nlp_x_u+nlp_x_time)),
            'p': ca.vertcat(*(nlp_p_xinit+nlp_p_wp)),
            'g': ca.vertcat(*(nlp_g_dyn+nlp_g_wp+nlp_g_quat))
        }
        self._topt_solver = ca.nlpsol('topt_solver', 'ipopt', topt_nlp_dect, self._topt_option)
        self._lam_x0 = np.zeros(len(nlp_x_x)*self._X_dim+len(nlp_x_u)*self._U_dim+len(nlp_x_time))
        self._lam_g0 = np.zeros(len(nlp_g_dyn)*self._X_dim+len(nlp_g_wp)+len(nlp_g_quat))
        
    def state_initialize(self, gates:Gates):
            
        p = np.zeros((self._X_dim+3*self._wp_num+self._wp_num))
        p[:self._X_dim] = np.array([0,0,0, 0,0,0, 1,0,0,0])
        for i in range(self._wp_num):
            p[self._X_dim+3*i:self._X_dim+3*(i+1)] = gates._g_pos[i,:]
            # self._dt[i] = 0.1
            # if i==0:
            #     self._dt[i] = np.linalg.norm(gates._g_pos[0,:])/6/self._N_per_wp
            # else:
            #     self._dt[i] = np.linalg.norm(gates._g_pos[i,:]-gates._g_pos[i-1,:])/6/self._N_per_wp
            # print(self._dt[i])
            p[self._X_dim+3*self._wp_num+i] = self._dt[i]
        # print(p)

        res = self._init_solver(
            x0=self._x0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u),
            lbg=(self._nlp_lbg_dyn+self._nlp_lbg_quat),
            ubg=(self._nlp_ubg_dyn+self._nlp_ubg_quat),
            # lbg=(self._nlp_lbg_dyn),
            # ubg=(self._nlp_ubg_dyn),
            p=p
        )
        self._x0 = np.array(res['x']).reshape(-1)
        self._xt0 = np.concatenate([self._x0, self._dt])
        return res

    def time_optimization(self, gates):

        p = np.zeros((self._X_dim+3*self._wp_num))
        p[:self._X_dim] = np.array([0,0,0, 0,0,0, 1,0,0,0])
        for i in range(self._wp_num):
            p[self._X_dim+3*i:self._X_dim+3*(i+1)] = gates._g_pos[i,:]
            
        res = self._topt_solver(
            x0=self._xt0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_time),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_time),
            lbg=(self._nlp_lbg_dyn+self._nlp_lbg_wp+self._nlp_lbg_quat),
            ubg=(self._nlp_ubg_dyn+self._nlp_ubg_wp+self._nlp_ubg_quat),
            p=p,
            lam_x0=self._lam_x0,
            lam_g0=self._lam_g0
        )
        self._xt0 = res['x'].full().flatten()
        self._x0 = self._xt0[:(self._X_dim+self._U_dim)*self._Herizon]
        self._dt = self._xt0[(self._X_dim+self._U_dim)*self._Herizon:]
        self._lam_x0 = res['lam_x'].full().flatten()
        self._lam_g0 = res['lam_g'].full().flatten()

        
        return res


class DynGateTimeOpt(object):
    def __init__(self, quad:Quadrotor):
        self._N1 = 20
        self._N2 = 20

        self._quad = quad
        self._dynamics = quad.dynamics()
        self._X_dim = self._dynamics.size1_in(0)
        self._U_dim = self._dynamics.size1_in(1)
        self._Herizon = self._N1+self._N2

        self._g_dyn = self._gate_dynamics()

        self._DT1 = ca.SX.sym('DT1')
        self._DT2 = ca.SX.sym('DT2')
        self._Xs = ca.SX.sym('X', self._X_dim, self._N1+self._N2)
        self._Us = ca.SX.sym('U', self._U_dim, self._N1+self._N2)
        
        self._X_init = ca.SX.sym('X_init', self._X_dim)
        self._X_end = ca.SX.sym('X_end', self._X_dim)
        self._g_p0 = ca.SX.sym('g_p0', 3)
        self._g_v0 = ca.SX.sym('g_v0', 3)
        self._g_a0 = ca.SX.sym('p_a0', 3)

        self._co_R = np.diag([0.1, 0.3,0.3,0.3])
        self._co_G = np.diag([1000, 1000, 1000])
        self._co_E = np.diag([10,10,10])

        self._init_option = {
            'verbose': False,
            'ipopt.tol': 1e-3,
            # 'ipopt.acceptable_tol': 1e-8,
            'ipopt.max_iter': 1000,
            'ipopt.warm_start_init_point': 'yes',
            # 'ipopt.print_level': 0,
        }
        self._topt_option = {
            'verbose': False,
            'ipopt.tol': 1e-2,
            # 'ipopt.acceptable_tol': 1e-8,
            'ipopt.max_iter': 1000,
            'ipopt.warm_start_init_point': 'yes',
            # 'ipopt.print_level': 0
        }

    def setup_opt(self):
        nlp_g_dyn = []
        self._nlp_lbg_dyn = []
        self._nlp_ubg_dyn = []

        nlp_g_gate = []
        self._nlp_lbg_gate = []
        self._nlp_ubg_gate = []

        nlp_g_quat = []
        self._nlp_lbg_quat = []
        self._nlp_ubg_quat = []

        nlp_x_x = []
        self._nlp_lbx_x = []
        self._nlp_ubx_x = []

        nlp_x_u = []
        self._nlp_lbx_u = []
        self._nlp_ubx_u = []

        nlp_x_time = []
        self._nlp_lbx_time = []
        self._nlp_ubx_time = []

        nlp_p_xinit = [self._X_init]
        nlp_p_xend = [self._X_end]
        nlp_p_time = [self._DT1, self._DT2]
        nlp_p_gate = [self._g_p0, self._g_v0, self._g_a0]

        nlp_obj_dyn = 0
        nlp_obj_minco = 0
        nlp_obj_time = self._DT1*self._N1+self._DT2*self._N2
        nlp_obj_gate = 0
        nlp_obj_end = 0

        t1 = self._DT1*self._N1
        gate_x_pos = self._g_p0 + self._g_v0*t1 + 0.5*self._g_a0*t1*t1

        dx = self._Xs[:,0]-self._X_init-self._dynamics(self._X_init, self._Us[:,0])*self._DT1
        nlp_g_dyn += [ dx ]
        self._nlp_lbg_dyn += [0 for _ in range(self._X_dim)]
        self._nlp_ubg_dyn += [0 for _ in range(self._X_dim)]
        nlp_obj_dyn += dx.T@dx

        nlp_g_quat += [ self._Xs[6:,0].T@self._Xs[6:,0] - 1 ]
        self._nlp_lbg_quat += [-0.1]
        self._nlp_ubg_quat += [ 0.1]

        nlp_x_x += [ self._Xs[:,0] ]
        self._nlp_lbx_x += self._quad._X_lb
        self._nlp_ubx_x += self._quad._X_ub

        nlp_x_u += [ self._Us[:,0] ]
        self._nlp_lbx_u += self._quad._U_lb
        self._nlp_ubx_u += self._quad._U_ub

        nlp_obj_minco += (self._Us[:,0]-[-9.81,0,0,0]).T@self._co_R@(self._Us[:,0]-[-9.81,0,0,0])

        for i in range(1,self._N1):
            dx = self._Xs[:,i]-self._Xs[:,i-1]-self._dynamics(self._Xs[:,i-1], self._Us[:,i])*self._DT1
            nlp_g_dyn += [ dx ]
            self._nlp_lbg_dyn += [0 for _ in range(self._X_dim)]
            self._nlp_ubg_dyn += [0 for _ in range(self._X_dim)]
            nlp_obj_dyn += dx.T@dx
            nlp_g_quat += [ self._Xs[6:,i].T@self._Xs[6:,i] - 1 ]
            self._nlp_lbg_quat += [-0.1]
            self._nlp_ubg_quat += [ 0.1]
            nlp_x_x += [ self._Xs[:,i] ]
            self._nlp_lbx_x += self._quad._X_lb
            self._nlp_ubx_x += self._quad._X_ub
            nlp_x_u += [ self._Us[:,i] ]
            self._nlp_lbx_u += self._quad._U_lb
            self._nlp_ubx_u += self._quad._U_ub
            nlp_obj_minco += (self._Us[:,i]-[-9.81,0,0,0]).T@self._co_R@(self._Us[:,i]-[-9.81,0,0,0])
            

        nlp_g_gate += [ (self._Xs[:3,self._N1-1]-gate_x_pos).T@(self._Xs[:3,self._N1-1]-gate_x_pos) ]
        self._nlp_lbg_gate += [0]
        self._nlp_ubg_gate += [0.001]
        nlp_obj_gate += (self._Xs[:3,self._N1-1]-gate_x_pos).T@self._co_G@(self._Xs[:3,self._N1-1]-gate_x_pos)

        for i in range(self._N1, self._N1+self._N2):
            dx = self._Xs[:,i]-self._Xs[:,i-1]-self._dynamics(self._Xs[:,i-1], self._Us[:,i])*self._DT2
            nlp_g_dyn += [ dx ]
            self._nlp_lbg_dyn += [0 for _ in range(self._X_dim)]
            self._nlp_ubg_dyn += [0 for _ in range(self._X_dim)]
            nlp_obj_dyn += dx.T@dx
            nlp_g_quat += [ self._Xs[6:,i].T@self._Xs[6:,i] - 1 ]
            self._nlp_lbg_quat += [-0.1]
            self._nlp_ubg_quat += [ 0.1]
            nlp_x_x += [ self._Xs[:,i] ]
            self._nlp_lbx_x += self._quad._X_lb
            self._nlp_ubx_x += self._quad._X_ub
            nlp_x_u += [ self._Us[:,i] ]
            self._nlp_lbx_u += self._quad._U_lb
            self._nlp_ubx_u += self._quad._U_ub
            nlp_obj_minco += (self._Us[:,i]-[-9.81,0,0,0]).T@self._co_R@(self._Us[:,i]-[-9.81,0,0,0])
        dd = self._Xs[:6,self._Herizon-1]-self._X_end[:6]
        nlp_g_dyn += [ dd ]
        self._nlp_lbg_dyn += [-0.1 for _ in range(6)]
        self._nlp_ubg_dyn += [0.1 for _ in range(6)]
        nlp_obj_dyn += dd.T@dd
        
        nlp_obj_end += (self._Xs[:3,self._Herizon-1]-self._X_end[:3]).T@self._co_E@(self._Xs[:3,self._Herizon-1]-self._X_end[:3])

        nlp_x_time += [self._DT1, self._DT2]
        self._nlp_lbx_time += [0, 0]
        self._nlp_ubx_time += [0.5, 0.5]

        init_nlp_dect = {
            'f': nlp_obj_dyn,
            'x': ca.vertcat(*(nlp_x_x+nlp_x_u)),
            'p': ca.vertcat(*(nlp_p_xinit+nlp_p_xend+nlp_p_gate+nlp_p_time)),
            # 'g': ca.vertcat(*(nlp_g_dyn+nlp_g_quat)),
            # 'g': ca.vertcat(*(nlp_g_dyn+nlp_g_gate))
        }
        self._init_solver = ca.nlpsol('init_solver', 'ipopt', init_nlp_dect, self._init_option)

        topt_nlp_dect = {
            'f': nlp_obj_time,
            'x': ca.vertcat(*(nlp_x_x+nlp_x_u+nlp_x_time)),
            'p': ca.vertcat(*(nlp_p_xinit+nlp_p_xend+nlp_p_gate)),
            'g': ca.vertcat(*(nlp_g_dyn+nlp_g_gate+nlp_g_quat))
        }
        self._topt_solver = ca.nlpsol('topt_solver', 'ipopt', topt_nlp_dect, self._topt_option)
        self._lam_x0 = np.zeros(len(nlp_x_x)*self._X_dim+len(nlp_x_u)*self._U_dim+len(nlp_x_time))
        self._lam_g0 = np.zeros(len(nlp_g_dyn)*self._X_dim+len(nlp_g_gate)+len(nlp_g_quat))

        self._x0 = np.zeros((self._X_dim+self._U_dim)*self._Herizon)
        self._dt0 = np.array([0.1, 0.1])
        for i in range(self._Herizon):
            self._x0[self._X_dim*i+6] = 1
            # self._x0[self._X_dim*self._Herizon + self._U_dim*i] = -9.8
        self._xt0 = np.concatenate([self._x0, self._dt0])

    def state_initialize(self, xinit ,xend, g_p, g_v, g_a):
        p = np.zeros(self._X_dim*2+9+2)
        p[:self._X_dim] = xinit
        p[self._X_dim:self._X_dim*2] = xend
        p[self._X_dim*2:self._X_dim*2+3] = g_p
        p[self._X_dim*2+3:self._X_dim*2+6] = g_v
        p[self._X_dim*2+6:self._X_dim*2+9] = g_a
        p[self._X_dim*2+9] = 0.05
        p[self._X_dim*2+10] = 0.05

        res = self._init_solver(
            x0=self._x0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u),
            # lbg=(self._nlp_lbg_dyn+self._nlp_lbg_quat),
            # ubg=(self._nlp_ubg_dyn+self._nlp_ubg_quat),
            # lbg=(self._nlp_lbg_dyn+self._nlp_lbg_gate),
            # ubg=(self._nlp_ubg_dyn+self._nlp_ubg_gate),
            p=p
        )
        self._x0 = res['x'].full().flatten()
        self._xt0 = np.zeros((self._X_dim+self._U_dim)*self._Herizon+2)
        self._xt0[:(self._X_dim+self._U_dim)*self._Herizon] = self._x0
        self._xt0[-1] = 0.02
        self._xt0[-2] = 0.02
        return res


    def time_optimization(self, xinit ,xend, g_p, g_v, g_a):
        p = np.zeros(self._X_dim*2+9)
        p[:self._X_dim] = xinit
        p[self._X_dim:self._X_dim*2] = xend
        p[self._X_dim*2:self._X_dim*2+3] = g_p
        p[self._X_dim*2+3:self._X_dim*2+6] = g_v
        p[self._X_dim*2+6:self._X_dim*2+9] = g_a

        res = self._topt_solver(
            x0=self._xt0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_time),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_time),
            # lbg=(self._nlp_lbg_dyn+self._nlp_lbg_quat),
            # ubg=(self._nlp_ubg_dyn+self._nlp_ubg_quat),
            lbg=(self._nlp_lbg_dyn+self._nlp_lbg_gate+self._nlp_lbg_quat),
            ubg=(self._nlp_ubg_dyn+self._nlp_ubg_gate+self._nlp_ubg_quat),
            p=p
        )
        self._xt0 = res['x'].full().flatten()

        return res

    def _gate_dynamics(self):
        
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        ax, ay, az = ca.SX.sym('ax'), ca.SX.sym('ay'), ca.SX.sym('az')

        X = ca.vertcat(px,py,pz, vx,vy,vz, ax,ay,az)
        X_dot = ca.vertcat(vx,vy,vz, ax,ay,az, 0,0,0)

        f = ca.Function('f', [X],[X_dot], ['x'],['x_dot'])
        return f


if __name__=="__main__":
    quad = Quadrotor('./gate/quad.yaml')
    gates = Gates()
    opt = TimeOpt(quad,gates._num)
    opt.define_opt()
    print(opt._init_solver)
    res = opt.state_initialize(gates)
    print(np.array(res['x']).reshape(-1))

    

