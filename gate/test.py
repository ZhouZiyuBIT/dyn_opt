import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from gates import Gates
from quadrotor import Quadrotor
from t_opt import TimeOpt



fig = plt.figure(figsize=(10,8))
grid = gridspec.GridSpec(nrows=4, ncols=10)

ax_3d = fig.add_subplot(grid[:,4:],projection="3d")
ax_vel = fig.add_subplot(grid[0,:4])
ax_acc = fig.add_subplot(grid[1,:4])
ax_rot = fig.add_subplot(grid[2,:4])

quad = Quadrotor('./gate/quad.yaml')
gates = Gates()
opt = TimeOpt(quad,gates._num)
opt.define_opt()
# print(opt._init_solver)

##########################################################################################
res = opt.state_initialize(gates)

Xs = np.array(res['x']).reshape(-1)
pos = np.zeros((3, opt._Herizon+1))
for i in range(opt._Herizon):
    pos[:,i+1] = Xs[opt._X_dim*i:opt._X_dim*i+3]
ax_3d.plot(pos[0,:], pos[1,:], pos[2,:])

##########################################################################################
for _ in range(1):
    res = opt.time_optimization(gates)
# print(res)

Xs = np.array(res['x']).reshape(-1)
pos = np.zeros((3, opt._Herizon+1))
time = np.zeros(opt._Herizon)
vel = np.zeros((3,opt._Herizon))
q_len = np.zeros(opt._Herizon)
t = 0
for i in range(opt._wp_num):
    dt = opt._dt[i]
    for j in range(opt._N_per_wp):
        t += dt
        time[i*opt._N_per_wp+j] = t
        pos[:,i*opt._N_per_wp+j+1] = Xs[opt._X_dim*(i*opt._N_per_wp+j):opt._X_dim*(i*opt._N_per_wp+j)+3]
        vel[:,i*opt._N_per_wp+j] = Xs[opt._X_dim*(i*opt._N_per_wp+j)+3:opt._X_dim*(i*opt._N_per_wp+j)+6]
        q_len[i*opt._N_per_wp+j] = np.linalg.norm(Xs[opt._X_dim*(i*opt._N_per_wp+j)+6:opt._X_dim*(i*opt._N_per_wp+j)+10])

ax_vel.plot(time, vel[0,:])
ax_vel.plot(time, vel[1,:])
ax_vel.plot(time, vel[2,:])

ax_rot.plot(q_len)

ax_3d.plot(pos[0,:], pos[1,:], pos[2,:], "-")
ax_3d.plot(gates._g_pos[:,0], gates._g_pos[:,1], gates._g_pos[:,2], '*')
plt.show()
