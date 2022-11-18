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
            'ipopt.max_iter': 100,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 0,
        }
        self._topt_option = {
            'verbose': False,
            # 'ipopt.tol': 1e-12,
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
            nlp_g_wp += [(self._Xs[:3,(i+1)*self._N_per_wp-1]-self._WP[:,i]).T@(self._Xs[:3,(i+1)*self._N_per_wp-1]-self._WP[:,i])]
            self._nlp_lbg_wp += [0]
            self._nlp_ubg_wp += [self._wp_tol*self._wp_tol]
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
                nlp_g_dyn += [ self._Xs[:,i*self._N_per_wp+j] - self._Xs[:,i*self._N_per_wp+j-1] - self._dynamics( self._Xs[:,i*self._N_per_wp+j-1], self._Us[:,i*self._N_per_wp+j] )*self._DT[i] ]
                self._nlp_lbg_dyn += [0 for i in range(self._X_dim)]
                self._nlp_ubg_dyn += [0 for i in range(self._X_dim)]
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
            'g': ca.vertcat(*(nlp_g_dyn))
        }
        self._init_solver = ca.nlpsol('init_solver', 'ipopt', init_nlp_dect, self._init_option)

        self._x0 = np.zeros((self._X_dim+self._U_dim)*self._Herizon)
        self._dt = np.zeros(self._wp_num)
        for i in range(self._wp_num):
            self._dt[i] = 0.1
            for j in range(self._N_per_wp):
                self._x0[self._X_dim*(i*self._N_per_wp+j)+6] = 1
                self._x0[self._X_dim*self._Herizon + self._U_dim*(i*self._N_per_wp+j)] = -9.8
        self._xt0 = np.concatenate([self._x0, self._dt])

        topt_nlp_dect = {
            'f': nlp_obj_time,
            'x': ca.vertcat(*(nlp_x_x+nlp_x_u+nlp_x_time)),
            'p': ca.vertcat(*(nlp_p_xinit+nlp_p_wp)),
            'g': ca.vertcat(*(nlp_g_dyn+nlp_g_wp))
        }
        self._topt_solver = ca.nlpsol('topt_solver', 'ipopt', topt_nlp_dect, self._topt_option)
        
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
            lbg=self._nlp_lbg_dyn,
            ubg=self._nlp_ubg_dyn,
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
            lbg=(self._nlp_lbg_dyn+self._nlp_lbg_wp),
            ubg=(self._nlp_ubg_dyn+self._nlp_ubg_wp),
            p=p
        )
        self._xt0 = np.array(res['x']).reshape(-1)
        self._x0 = self._xt0[:(self._X_dim+self._U_dim)*self._Herizon]
        self._dt = self._xt0[(self._X_dim+self._U_dim)*self._Herizon:]

        
        return res


if __name__=="__main__":
    quad = Quadrotor('./gate/quad.yaml')
    gates = Gates()
    opt = TimeOpt(quad,gates._num)
    opt.define_opt()
    print(opt._init_solver)
    res = opt.state_initialize(gates)
    print(np.array(res['x']).reshape(-1))

    

