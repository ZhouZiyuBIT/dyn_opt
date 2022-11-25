import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from quadrotor import Quadrotor
from t_opt import DynGateTimeOpt



fig = plt.figure(figsize=(10,8))
grid = gridspec.GridSpec(nrows=3, ncols=10)

ax_3d = fig.add_subplot(grid[:,4:],projection="3d")
ax_3d.view_init(elev=200, azim=-15)
ax_vel = fig.add_subplot(grid[0,:4])
ax_acc = fig.add_subplot(grid[1,:4])
ax_rot = fig.add_subplot(grid[2,:4])
# ax_quat = fig.add_subplot(grid[3,:4])

quad = Quadrotor('./gate/quad.yaml')
opt = DynGateTimeOpt(quad)
opt.setup_opt()
print(opt._init_solver)

####
g_p0 = np.array([1, -1, -1])
g_v0 = np.array([0, 1, -5])
g_a0 = np.array([0, 0, 9.81])

g_t = np.linspace(0, 1.2, 100)
g_px = g_p0[0] + g_v0[0]*g_t + 0.5*g_a0[0]*g_t*g_t
g_py = g_p0[1] + g_v0[1]*g_t + 0.5*g_a0[1]*g_t*g_t
g_pz = g_p0[2] + g_v0[2]*g_t + 0.5*g_a0[2]*g_t*g_t
ax_3d.plot(g_px, g_py, g_pz)

quad_xinit = np.array([0,0,0, 0,0,0, 1,0,0,0])
quad_xend = np.array([3,0,-1, 0,0,0, 1,0,0,0])

####

# ##########################################################################################
res = opt.state_initialize(quad_xinit, quad_xend, g_p0, g_v0, g_a0)
for i in range(10):
    res = opt.time_optimization(quad_xinit, quad_xend, g_p0, g_v0, g_a0)
print(res['f'])
Xs = res['x'].full().flatten()
pos = np.zeros((3, opt._Herizon+1))
vel = np.zeros([3, opt._Herizon])
az_B = np.zeros(opt._Herizon)
rot_w = np.zeros((3,opt._Herizon))
for i in range(opt._Herizon):
    pos[:,i+1] = Xs[opt._X_dim*i:opt._X_dim*i+3]
    vel[:,i] = Xs[opt._X_dim*i+3:opt._X_dim*i+6]
    az_B[i] = Xs[opt._X_dim*opt._Herizon+opt._U_dim*i]
    rot_w[:,i] = Xs[opt._X_dim*opt._Herizon+opt._U_dim*i+1:opt._X_dim*opt._Herizon+opt._U_dim*i+4]

import matplotlib.pyplot as plt
import matplotlib.animation as animation

ax_3d.plot(pos[0,:], pos[1,:], pos[2,:])

ax_3d.plot(pos[0,opt._N1],pos[1,opt._N1-1],pos[2,opt._N1-1], '*')
ax_3d.plot(pos[0,-1],pos[1,-1],pos[2,-1], '*')

ax_vel.plot(vel[0,:])
ax_vel.plot(vel[1,:])
ax_vel.plot(vel[2,:])

ax_acc.plot(az_B)
ax_rot.plot(rot_w[0,:])
ax_rot.plot(rot_w[1,:])
ax_rot.plot(rot_w[2,:])

dd1, = ax_3d.plot([],[],[], "*")
dd2, = ax_3d.plot([],[],[], "*")


def sim_run(steps=100):
    step = 0    
    while step<steps:
        step+=1
        yield step

sim = sim_run()

def update_plot(*arg):
    i = next(sim)
    k = 3.4
    dd1.set_data_3d(pos[0,i], pos[1,i], pos[2,i])
    dd2.set_data_3d(g_px[int(i*k)], g_py[int(i*k)], g_pz[int(i*k)])
    return [dd1, dd2]
# ani = animation.FuncAnimation(fig, update_plot, interval=100, save_count=30, blit=True, repeat=False)
# ani.save('./gif/111.gif', writer='imagemagick')
# ##########################################################################################
# for _ in range(10):
#     res = opt.time_optimization(gates)
#     gates.update()
#     print(np.sum(res['x'].full().flatten()[-opt._wp_num:])*50)
#     # opt.state_initialize(gates)
# # print(res['lam_p'])
# # print(res['x'].full().flatten()[10*350:])
# # print(np.sum(res['x'].full().flatten()[-opt._wp_num:])*50)

# Xs = np.array(res['x']).reshape(-1)
# pos = np.zeros((3, opt._Herizon+1))
# time = np.zeros(opt._Herizon)
# vel = np.zeros((3,opt._Herizon))
# q_len = np.zeros(opt._Herizon)
# az_B = np.zeros(opt._Herizon)
# rot_w = np.zeros((3,opt._Herizon))
# t = 0
# for i in range(opt._wp_num):
#     dt = opt._dt[i]
#     for j in range(opt._N_per_wp):
#         t += dt
#         time[i*opt._N_per_wp+j] = t
#         pos[:,i*opt._N_per_wp+j+1] = Xs[opt._X_dim*(i*opt._N_per_wp+j):opt._X_dim*(i*opt._N_per_wp+j)+3]
#         vel[:,i*opt._N_per_wp+j] = Xs[opt._X_dim*(i*opt._N_per_wp+j)+3:opt._X_dim*(i*opt._N_per_wp+j)+6]
#         q_len[i*opt._N_per_wp+j] = np.linalg.norm(Xs[opt._X_dim*(i*opt._N_per_wp+j)+6:opt._X_dim*(i*opt._N_per_wp+j)+10])
#         az_B[i*opt._N_per_wp+j] = Xs[opt._X_dim*opt._Herizon+opt._U_dim*(i*opt._N_per_wp+j)]
#         rot_w[:,i*opt._N_per_wp+j] = Xs[opt._X_dim*opt._Herizon+opt._U_dim*(i*opt._N_per_wp+j)+1:opt._X_dim*opt._Herizon+opt._U_dim*(i*opt._N_per_wp+j)+4]

# ax_vel.plot(time, vel[0,:])
# ax_vel.plot(time, vel[1,:])
# ax_vel.plot(time, vel[2,:])
# ax_acc.plot(time, az_B)
# ax_rot.plot(time, rot_w[0,:])
# ax_rot.plot(time, rot_w[1,:])
# ax_rot.plot(time, rot_w[2,:])
# ax_quat.plot(q_len)

# ax_3d.plot(pos[0,:], pos[1,:], pos[2,:], "-")
# ax_3d.plot(gates._g_pos[:,0], gates._g_pos[:,1], gates._g_pos[:,2], '*')
plt.show()
