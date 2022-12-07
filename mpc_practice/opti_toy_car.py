"""
Easier formulation of MPC without the crazy stuff
"""
from time import time 
import casadi as ca
import numpy as np
import pandas as pd


class ToyCar():
    """
    Toy Car Example 
    
    3 States: 
    [x, y, psi]
     
     2 Inputs:
     [v, psi_rate]
    
    """
    def __init__(self):
        self.define_states()
        self.define_controls()
        
    def define_states(self):
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.psi = ca.SX.sym('psi')
        
        self.states = ca.vertcat(
            self.x,
            self.y,
            self.psi
        )
        #column vector of 3 x 1
        self.n_states = self.states.size()[0] #is a column vector 
        
    def define_controls(self):
        self.v_cmd = ca.SX.sym('v_cmd')
        self.psi_cmd = ca.SX.sym('psi_cmd')
        
        self.controls = ca.vertcat(
            self.v_cmd,
            self.psi_cmd
        )
        #column vector of 2 x 1
        self.n_controls = self.controls.size()[0] 
        
    def set_state_space(self):
        #this is where I do the dynamics for state space
        self.x_dot = self.v_cmd * ca.cos(self.psi)
        self.y_dot = self.v_cmd * ca.sin(self.psi)
        self.psi_dot = self.psi_cmd
        
        self.z_dot = ca.vertcat(
            self.x_dot, self.y_dot, self.psi_dot    
        )
        
        #ODE right hand side function
        self.function = ca.Function('f', 
                        [self.states, self.controls],
                        [self.z_dot]
                        ) 
        
        return self.function
    

if __name__ == '__main__':
    
    #limits 
    v_max = 5
    v_min = -5
    
    psi_rate_max = np.deg2rad(45)
    psi_rate_min = -np.deg2rad(45)
    
    #init locations
    init_x = 0.0
    init_y = 0.0
    init_psi = 0.0
    
    #goal locations
    goal_x = 3.0
    goal_y = 1.0
    goal_psi = np.deg2rad(20)
    
    #state model
    toy_car = ToyCar()
    f = toy_car.set_state_space()
    
    #%% Optimizatoin 
    #horizon time and dt value, can make basis on how fast I'm localizing
    N = 50
    dt_val = 0.1
    opti = ca.Opti()
    dt_val = 0.1 
    t_init = 0.0
    
    #decision variables to send to NLP
    X = opti.variable(toy_car.n_states, N+1) #allow one more iteration for final
    x_pos = X[0,:]
    y_pos = X[1,:]
    psi = X[2,:]
    
    U = opti.variable(toy_car.n_controls, N) # U decision variables
    u_vel = U[0,:]
    u_psi_rate = U[1,:]
    
    ## initial parameters 
    #Initiate State parameter
    x0 = opti.parameter(toy_car.n_states)    
    xF = opti.parameter(toy_car.n_states)

    #set initial value
    opti.set_value(x0, [init_x, init_y, init_psi])
    opti.set_value(xF, [goal_x, goal_y, goal_psi])
        
    #intial and terminal constraints 
    opti.subject_to(X[:,0] == x0)
    # opti.subject_to(X[:,-1] == xF)

    #boundary constraints
    # opti.subject_to(opti.bounded(-0.5, x_pos, 5.5))
    # opti.subject_to(opti.bounded(-0.5, y_pos, 5.5))
    
    opti.subject_to(opti.bounded(psi_rate_min, u_psi_rate, psi_rate_max))
    opti.subject_to(opti.bounded(v_min, u_vel, v_max))
    
    #g constraints dynamic constraints
    for k in range(N):
        states = X[:, k]
        controls = U[:, k]
        state_next =X[:, k+1]
        k1 = f(states, controls)
        k2 = f(states + dt_val/2*k1, controls)
        k3 = f(states + dt_val/2*k2, controls)
        k4 = f(states + dt_val * k3, controls)
        state_next_RK4 = states + (dt_val / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        opti.subject_to(X[:, k+1] == state_next_RK4)
        
    # cost function
    e_x = (goal_x - x_pos)
    e_y = (goal_y - y_pos)
    e_z = (goal_psi - psi)
    
    weights = [100, 100, 1]
    
    cost_value = 1*ca.sumsqr(e_x) + 1*ca.sumsqr(e_y) + \
        1*ca.sumsqr(e_z) + ca.sumsqr(U)


    opti.minimize(cost_value)
    opts = {
        'ipopt': {
            'max_iter': 2000,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6
        },
    }
    opti.solver('ipopt', opts)#, {'ipopt': {'print_level': 0}})
    
    sol = opti.solve()

    #unpack solution
    t_simulate = np.arange(t_init, N*dt_val, dt_val)
    x_traj = sol.value(X).T[:-1]
    u_traj = sol.value(U).T
    sol_traj = pd.DataFrame(np.hstack((x_traj, u_traj)), 
                            columns=['x', 'y', 'psi', 
                                      'vx', 'psi_rate'])
    
#%% 
import matplotlib.pyplot as plt

plt.close('all')
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d


buffer_zone = 1

fig2, ax2 = plt.subplots(figsize=(8,8))

ax2.plot(goal_x, goal_y, 'x', markersize=20, label='goal position')
ax2.plot(sol_traj["x"], sol_traj["y"])
