import casadi as ca 
import casadi.tools 

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

"""
State of nonholonomic vehicle:
x = [
    x_pos,
    y_pos, 
    psi,
    body_velocity 
]

inputs = [
    v 
    omega
]

"""


################################################################
# Global Variables
################################################################ 


################################################################
# States, Inputs, Equations
################################################################ 
"""States """
states = ['x_pos', 
          'y_pos', 
          'psi']

x = ca.tools.struct_symMX(
    states
)


"""Inputs"""
inputs = ['velocity', 
          'angular_velocity']

u = ca.tools.struct_symMX(
    inputs
)

"""Equation setup"""
dx = ca.tools.struct_MX(x) #make sure this is a matrix 
 
dx_pos = u["velocity"] * ca.cos(x["psi"])
dy_pos = u["velocity"] * ca.sin(x["psi"])
dpsi = u['angular_velocity'] 

dx["x_pos"] = dx_pos
dx["y_pos"] = dy_pos
dx["psi"] = dpsi

# Ode Right Hand Side ax+bu = x_dot 
rhs = dx
f = ca.Function('f',
                [x,u],
                [rhs],
                ['x', 'u'],
                ['dx']
)

################################################################
# Initial Integration
################################################################

dt = ca.MX.sym("dt")
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


################################################################
# Optimization setup
################################################################

opti = ca.Opti()

#optimization grid points 
N = 35
dt_val = 0.1 

# define decision variables 
X = opti.variable(x.size, N+1) # decision variables of states x grid_points + 1 
x_pos = X[0,:]
y_pos = X[1,:]
psi = X[2,:]

U = opti.variable(u.size, N) # decision variables of inputs x grid_points + 1
velocity = U[0,:]
angular_velocity = U[1,:]

#Initial State parameter 
x0 = opti.parameter(x.size) # number of states 

"""
Add dynamic constraints, g utilizing Runge Kutta, can use collocation methoimport casadi
ds
"""  

for k in range(N):
   opti.subject_to(
       X[:,k+1] == F_RK4(X[:,k], U[:,k], dt_val)
       )

# Control constraints defined as path constraints
MAX_VELOCITY = 5 #m/s
MAX_ANG_VELOCITY = np.deg2rad(60) #rad/s

opti.subject_to(opti.bounded(-MAX_VELOCITY, velocity, MAX_VELOCITY))
opti.subject_to(opti.bounded(-MAX_ANG_VELOCITY, angular_velocity, MAX_ANG_VELOCITY))

# Initial and terminal constraints
opti.subject_to(X[:,0] == x0)
# opti.subject_to(X[:,-1] == [2,0,0])

#cost function
goal_x = 2.0
goal_y = 2.0
goal_psi = np.deg2rad(45)

e_x = (goal_x - x_pos)
e_y = (goal_y - y_pos)
e_psi = (goal_psi - psi)

weights = [1, 1 , 100]

cost_value = ca.sumsqr(e_x) + ca.sumsqr(e_y) + \
    (weights[-1]* ca.sumsqr(e_psi)) + ca.sumsqr(U)

opti.minimize(cost_value)
#opti.minimize(ca.sumsqr(U))

#solve problem
#opti.solver('ipopt', {'ipopt': {'print_level': 0}})

opti.solver('ipopt')#, {'ipopt': {'print_level': 0}})
#set initial guesses 
opti.set_value(x0, [0, 0, 0])
sol = opti.solve()

x_traj = sol.value(X).T[:-1]
u_traj = sol.value(U).T
t_simulate = np.arange(0, N*dt_val, dt_val)


sol_traj = pd.DataFrame(np.hstack((x_traj, u_traj)), 
                        columns=['x', 'y', 'psi', 'velocity', 'angular_velocity'])


#%% 
################################################################
# Visualize Results
################################################################
plt.close('all')

fig, ax = plt.subplots(figsize=(8, 8))

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

fig3, ax3 = plt.subplots(figsize=(8,8))

ax3.plot(t_simulate, np.rad2deg(sol_traj['psi']), label='psi')
ax3.legend(loc="upper left")


ax2.grid()
