"""
https://github.com/borlum/casadi_intro/blob/master/robot_kinematics.ipynb

Note how the result consists of only two 
operations (one multiplication and one addition)
using MX symbolics, whereas the SX equivalent has eight 
(two for each element of the resulting matrix). 
As a consequence, MX can be more economical when 
working with operations
that are naturally vector or matrix valued with many elements

"""

import casadi as ca
import casadi.tools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.close('all')

# State vector
x = ca.tools.struct_symMX(
    [ca.tools.entry("p", shape=2), "𝜃"])

# Input vector
u = ca.tools.struct_symMX(["v", "𝜔"])

# State equations
dxdt = ca.tools.struct_MX(x)

dp_x = u["v"] * ca.cos(x["𝜃"])
dp_y = u["v"] * ca.sin(x["𝜃"])
d𝜃 = u["𝜔"]

dxdt["p"] = ca.vertcat(dp_x, dp_y)
dxdt["𝜃"] = d𝜃
                               
# ODE Right-hand side
rhs = dxdt
f = ca.Function('f', 
                [x, u], 
                [rhs], 
                ['x', 'u'], 
                ['dx/dt'])

#%% Integration 
#dt = 1 # [s], 10 Hz sampling
dt = ca.MX.sym("dt")

# RK4
k1 = f(x, u)
k2 = f(x + dt / 2.0 * k1, u)
k3 = f(x + dt / 2.0 * k2, u)
k4 = f(x + dt * k3, u)
xf = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

# Single step time propagation
F_RK4 = ca.Function("F_RK4", 
                    [x, u, dt], [xf], 
                    ['x[k]', 'u[k]', "dt"], 
                    ['x[k+1]'])

# simulator = F_RK4.mapaccum(100)
# res = simulator([0, 0, 0], [1, 1.57], 0.1) # dt = 1

# sol_traj = pd.DataFrame(res.toarray().T, columns=['x', 'y', '𝜃'])

# plt.plot(sol_traj["x"], sol_traj["y"])

#%% Optimal Control Formulations
opti = ca.Opti()

# Optimization horizon
N = 50
dt_val = 0.1
# Decision variables for states and inputs
X = opti.variable(x.size, N+1)

p_x = X[0,:]
p_y = X[1,:]
𝜃 = X[2,:]

U = opti.variable(u.size, N)

v = U[0,:]
𝜔 = U[1,:]

# Initial state is a parameter
x0 = opti.parameter(x.size)

# Gap-closing shooting constraints
for k in range(N):
   opti.subject_to(
       X[:,k+1] == F_RK4(X[:,k], U[:,k], dt_val)
       )
   
   
# Path constraints
MAX_VEL = 25
MAX_ANG = 2.5
opti.subject_to(opti.bounded(-MAX_VEL, v, MAX_VEL))
opti.subject_to(opti.bounded(-MAX_ANG, 𝜔, MAX_ANG))


goal_x = 2.0
goal_y = 1.0
e_x = (goal_x - p_x)
e_y = (goal_y - p_y)

s = opti.variable(N)

d = ca.sqrt((0.5 - p_x)**2 + (0.15 - p_y)**2)

#opti.subject_to(d[0:-1].T >= s + 0.2)

#opti.minimize(3*ca.sumsqr(e_x) + 3*ca.sumsqr(e_y) + 0.01*ca.sumsqr(U) + 1000*ca.sumsqr(s))

opti.minimize(ca.sumsqr(e_x) + ca.sumsqr(e_y) + ca.sumsqr(U))

# Initial and terminal constraints
opti.subject_to(X[:,0] == x0)
# opti.subject_to(X[:,-1] == [goal_x,goal_y, 0])


opti.solver('ipopt')#, {'ipopt': {'print_level': 0}})

opti.set_value(x0, [0, 0, 0])
sol = opti.solve()

x_traj = sol.value(X).T[:-1]
u_traj = sol.value(U).T
t_simulate = np.arange(0, N*dt_val, dt_val)

sol_traj = pd.DataFrame(np.hstack((x_traj, u_traj)), columns=['x', 'y', 'theta', 'v', 'omega'])


fig, ax = plt.subplots(figsize=(8, 8))

#c = plt.Circle((0.5, 0.15), radius=0.2, alpha=0.5)

ax.plot(goal_x, goal_y, 'x', markersize=20)
ax.plot(sol_traj["x"], sol_traj["y"])
#ax.add_patch(c)

ax.set_xlim([-1, 3])
ax.set_ylim([-1, 3])
ax.grid()

fig2, ax2 = plt.subplots(figsize=(8,8))

ax2.plot(t_simulate, u_traj[:,0], label='velocity')
ax2.plot(t_simulate, u_traj[:,1], label='angular_velocity')
ax2.legend(loc="upper left")

ax2.grid()