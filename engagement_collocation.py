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
def stateFunc(x, u):

    psi = x[2]
    velocity = u[0]
    ang_velocity = u[1]
    
    dx_pos = velocity*ca.cos(psi)
    dy_pos = velocity*ca.sin(psi)
    dpsi = ang_velocity

    return [dx_pos, dy_pos, dpsi]


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
dx["psi"] = dpsi #+ x["psi"]

# Ode Right Hand Side ax+bu = x_dot 
rhs = dx
f = ca.Function('f',
                [x,u],
                [rhs],
                ['x', 'u'],
                ['dx']
)


################################################################
# Optimization setup
################################################################

opti = ca.Opti()

#optimization grid points 
N = 20
dt_val = 0.1

# define decision variables 
X = opti.variable(x.size, N+1) # decision variables of states x grid_points + 1 
x_pos = X[0,:]
y_pos = X[1,:]
psi = X[2,:]

U = opti.variable(u.size, N) # decision variables of inputs x grid_points + 1
velocity = U[0,:]
angular_velocity = U[1,:]
T = opti.variable() # final time


#Initial State parameter 
x0 = opti.parameter(x.size) # number of states 

"""
Add dynamic constraints, g utilizing Runge Kutta, can use collocation methods
"""  

################################################################
# Initial Integration
################################################################

#doing this through matrix operations
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


#dt_val = T/N
for k in range(N):
    opti.subject_to(
        X[:,k+1] == F_RK4(X[:,k], U[:,k], dt_val)
        )


#hermite simpson qudrature
# dt = ca.MX.sym("dt")
# z_low = X[:,:-1] #all states to just before the last time grid point
# z_upp = X[:,1:] #all states but skip the first grid point

# u_low = U[:,:-1] 
# u_upp = U[:,1:] 

# f_low = stateFunc(z_low,u_low)
# f_upp = stateFunc(z_upp, u_upp)

# ## compute dynamics at intermediate point, centroid
# z_bar = 0.5*(z_low + z_upp) + (dt/8)*(f_low-f_upp);
# u_bar = u_low + (u_low + u_upp)/2 #trapezoid 
# f_bar = stateFunc(z_bar, u_bar)

# ## compute the defects
# defects = z_upp - z_low - (dt/6)*(f_low + 4*f_bar + f_upp);


# Control constraints defined as path constraints
MAX_VELOCITY = 45 #m/s
MIN_VELOCITY = 15 #m/s
MAX_ANG_VELOCITY = np.deg2rad(60) #rad/s

opti.subject_to(opti.bounded(MIN_VELOCITY, velocity, MAX_VELOCITY))
opti.subject_to(opti.bounded(-MAX_ANG_VELOCITY, angular_velocity, MAX_ANG_VELOCITY))


goal_x = 25
goal_y = 25
goal_psi = np.deg2rad(45)

# Initial and terminal constraints
opti.subject_to(X[:,0] == x0)
# opti.subject_to(T>=0) #time must be grater than 0

opti.subject_to(X[:,-1] == [goal_x,goal_y,goal_psi])

#cost function
e_x = (goal_x - x_pos)
e_y = (goal_y - y_pos)
e_psi = (goal_psi - psi)

weights = [1, 1, 10]

# cost_value = (weights[0]*ca.sumsqr(e_x)) + (weights[1]*ca.sumsqr(e_y)) + \
#     (weights[-1]* ca.sumsqr(e_psi)) + ca.sumsqr(U)

cost_value = (weights[0]*ca.sumsqr(e_x)) + (weights[1]*ca.sumsqr(e_y)) + \
    (weights[-1]* ca.sumsqr(e_psi)) 

#opti.minimize(cost_value)
opti.minimize(cost_value)

#solve problem
opti.solver('ipopt')#, {'ipopt': {'print_level': 0}})
#set initial guesses 
opti.set_value(x0, [0, 0, 0])
sol = opti.solve()

x_traj = sol.value(X).T[:-1]
u_traj = sol.value(U).T


sol_traj = pd.DataFrame(np.hstack((x_traj, u_traj)), 
                        columns=['x', 'y', 'psi', 'velocity', 'angular_velocity'])


#%% 
################################################################
# Visualize Results
################################################################
plt.close('all')
t_simulate = np.arange(0, N*dt_val, dt_val)
# t_simulate = np.arange(0,sol.value(T), sol.value(dt_val))

fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(goal_x, goal_y, 'x', markersize=20)
ax.plot(sol_traj["x"], sol_traj["y"])
#ax.add_patch(c)

ax.set_xlim([-1, goal_x+10])
ax.set_ylim([-1, goal_y+10])
ax.grid()

fig2, ax2 = plt.subplots(figsize=(8,8))

ax2.plot(t_simulate, np.rad2deg(u_traj[:,1]), label='angular_velocity')
ax2.plot(t_simulate, np.rad2deg(sol_traj['psi']), label='psi')

ax2.legend(loc="upper left")


ax2.grid()

fig3, ax3 = plt.subplots(figsize=(8,8))
ax3.plot(t_simulate, u_traj[:,0], label='velocity')

ax2.grid()
