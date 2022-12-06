import casadi as ca
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

"""

Lateral Dynamics:
V velocity in lateral direction  
p = roll rate dynamics 
r = yaw rate 
phi = yaw heading 


"""

aileron_min = -np.deg2rad(25)
aileron_max = np.deg2rad(25)

rudder_min = -np.deg2rad(25)
rudder_max = np.deg2rad(25)

p_min = -np.deg2rad(120)
p_max = np.deg2rad(120)

r_min = -np.deg2rad(45)
r_max = np.deg2rad(45)

phi_min = -np.deg2rad(45)
phi_max = np.deg2rad(45)

v_min = -10 #m/s
v_max = 10.0 #m/s

v_index = 0 
p_index = 1
r_index = 2
phi_index = 3

da_index = 0
dr_index = 1

class LateralPlane():
    """lateral airplane"""
    def __init__(self) -> None:
        self.define_states()
        self.define_controls()
    
    
    def define_states(self):
        """define the states of your system"""
        #body velocities
        self.v = ca.SX.sym('v')
        self.p = ca.SX.sym('p')
        self.r = ca.SX.sym('r')
        self.phi = ca.SX.sym('phi')
        
        self.states = ca.vertcat(
            self.v,
            self.p, 
            self.r, 
            self.phi
        )
            
        self.n_states = self.states.size()[0] #is a column vector 

    def define_controls(self):
        """controls for system"""
        self.b1 = ca.horzcat(
            0,   -0.3526  
        )
        
        self.b2 = ca.horzcat(
            31.0036,   -1.4273
        )
        
        self.b3 = ca.horzcat(
            0,    2.4580
        )
        
        self.b4 = ca.horzcat(
            0,         0
        )
        
        self.B = ca.vertcat(
            self.b1, 
            self.b2, 
            self.b3,
            self.b4
        )
        
        self.da = ca.SX.sym('da')
        self.dr = ca.SX.sym('dr')
        self.controls = ca.vertcat(
            self.da,
            self.dr 
        )
            
        #column vector of 2 x 1
        self.n_controls = self.controls.size()[0] 
    
    def set_state_space(self):
        """set state space """
        self.a1 = ca.horzcat(
          -0.3728,         0,   -1.0000,    9.8100
        )
        
        self.a2 = ca.horzcat(
          -0.6802,   -1.1195,    0.0600,         0
        )   
        
        self.a3 = ca.horzcat(
            0.6320,   -0.0087,   -0.0473,         0
        ) 
        
        self.a4 = ca.horzcat(
            0,    1.0000,         0,         0
        )
        
        self.A = ca.vertcat(
            self.a1,
            self.a2,
            self.a3,
            self.a4, 
        )
        
        self.z_dot = ca.mtimes(airplane.A ,airplane.states) + \
            ca.mtimes(airplane.B ,airplane.controls)
            
        #ODE right hand side function
        self.function = ca.Function('f', 
                        [self.states, self.controls],
                        [self.z_dot]
                        ) 
        
        return self.function
    
class Optimization():
    
    def __init__(self, model, dt_val, N):
        self.model = model
        self.f = model.function        
        
        self.n_states = model.n_states
        self.n_controls = model.n_controls
        
        self.dt_val = dt_val 
        self.N = N
        
        """this needs to be changed, let user define this"""
        self.Q = ca.diagcat(1.0, #v
                            1.0, #p
                            1.0, #r
                            1.0) #phi
        
        self.R = ca.diagcat(1.0, #da
                            1.0) #dr
        
        #initialize cost function as 0
        self.cost_fn = 0        

    def init_decision_variables(self):
        """intialize decision variables for state space models"""
        self.X = ca.SX.sym('X', self.n_states, self.N + 1)
        self.U = ca.SX.sym('U', self.n_controls, self.N)
        
        #column vector for storing initial and target locations
        self.P = ca.SX.sym('P', self.n_states + self.n_states)
        
        #dynamic constraints 
        self.g = self.X[:,0] - self.P[:self.n_states]


    def define_bound_constraints(self):
        """define bound constraints of system"""
        self.variables_list = [self.X, self.U]
        self.variables_name = ['X', 'U']
        
        #function to turn decision variables into one long row vector
        self.pack_variables_fn = ca.Function('pack_variables_fn', 
                                             self.variables_list, 
                                             [self.OPT_variables], 
                                             self.variables_name, 
                                             ['flat'])
        
        #function to turn decision variables into respective matrices
        self.unpack_variables_fn = ca.Function('unpack_variables_fn', 
                                               [self.OPT_variables], 
                                               self.variables_list, 
                                               ['flat'], 
                                               self.variables_name)

        ##helper functions to flatten and organize constraints
        self.lbx = self.unpack_variables_fn(flat=-ca.inf)
        self.ubx = self.unpack_variables_fn(flat=ca.inf)

        """
        REFACTOR THIS
        """
        #right now still coupled with state space system, 0 means velcoity 1 is psi rate        
        self.lbx['X'][v_index, :] = v_min
        self.ubx['X'][v_index, :] = v_max
        
        self.lbx['X'][p_index, :] = p_min
        self.ubx['X'][p_index, :] = p_max
        
        self.lbx['X'][r_index, :] = r_min
        self.ubx['X'][r_index, :] = r_max
        
        self.lbx['X'][phi_index, :] = phi_min
        self.ubx['X'][phi_index, :] = phi_max
        
        
        self.lbx['U'][da_index,:] = aileron_min
        self.ubx['U'][da_index,:] = aileron_max

        self.lbx['U'][dr_index,:] = rudder_min
        self.ubx['U'][dr_index,:] = rudder_max

    def compute_cost(self):
        """this is where we do integration methods to find cost"""
        #tired of writing self
        P = self.P
        Q = self.Q
        R = self.R
        n_states = self.n_states
        
        for k in range(N):
            states = self.X[:, k]
            controls = self.U[:, k]
            state_next = self.X[:, k+1]
            
            #penalize states and controls for now, can add other stuff too
            self.cost_fn = self.cost_fn \
                + (states - P[n_states:]).T @ Q @ (states - P[n_states:]) \
                + controls.T @ R @ controls 
            
            ##Runge Kutta
            k1 = self.f(states, controls)
            k2 = self.f(states + self.dt_val/2*k1, controls)
            k3 = self.f(states + self.dt_val/2*k2, controls)
            k4 = self.f(states + self.dt_val * k3, controls)
            state_next_RK4 = states + (self.dt_val / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.g = ca.vertcat(self.g, state_next - state_next_RK4) #dynamic constraints

    def init_solver(self):
        """init the NLP solver utilizing IPOPT this is where
        you can set up your options for the solver"""
        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            self.U.reshape((-1, 1))
        )
        
        nlp_prob = {
            'f': self.cost_fn,
            'x': self.OPT_variables,
            'g': self.g,
            'p': self.P
        }

        opts = {
            'ipopt': {
                'max_iter': 5000,
                'print_level': 5,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            # 'jit':True,
            'print_time': 5
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def solve_trajectory(self,start, goal, t0):
        """solve just the traectory"""
        #Packing the solution into a single row vector
        main_loop = time()  # return time in sec
        n_states = self.n_states
        n_controls = self.n_controls
        solution_list = []
        

        self.state_init = ca.DM(start)        # initial state
        self.state_target = ca.DM(goal)  # target state
        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(self.state_init, 1, self.N+1)         # initial state full
        times = np.array([[0]]) 
        
        self.t0 = t0
        
        time_history = [self.t0]

        t1 = time()

        ## the arguments are what I care about
        args = {
            'lbg': ca.DM.zeros((self.n_states*(self.N+1), 1)),  # constraints lower bound
            'ubg': ca.DM.zeros((self.n_states*(self.N+1), 1)),  # constraints upper bound
            'lbx': self.pack_variables_fn(**self.lbx)['flat'],
            'ubx': self.pack_variables_fn(**self.ubx)['flat'],
        }

        #this is where you can update the target location
        args['p'] = ca.vertcat(
            self.state_init,    # current state
            self.state_target   # target state
        )
        
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(self.X0, n_states*(self.N+1), 1),
            ca.reshape(self.u0, n_controls*self.N, 1)
        )

        #this is where we solve
        sol = self.solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        print("solution found")

        #unpack as a matrix
        self.u = ca.reshape(sol['x'][self.n_states * (self.N + 1):], 
                            self.n_controls, self.N)
        
        self.X0 = ca.reshape(sol['x'][: n_states * (self.N+1)], 
                                self.n_states, self.N+1)

        main_loop_time = time()
        ss_error = ca.norm_2(self.state_init - self.state_target)
        
        solution_list.append((self.u, self.X0))

        print('\n\n')
        print('Total time: ', main_loop_time - main_loop)
        print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
        print('final error: ', ss_error)
         
        return time_history, solution_list

    
if __name__ == '__main__':
    airplane = LateralPlane()
    f = airplane.set_state_space()

    #CHECKING SIZE OF matrix
    ax = ca.mtimes(airplane.A ,airplane.states)
    bu = ca.mtimes(airplane.B,  airplane.controls)
    print("size of Ax", ax.size())
    print("Size of Bu", bu.size())

    #initial states 
    init_states = [0, 0, 0, 0]
    
    desired_states = [0, 0, 0, np.rad2deg(45)]


    #### Optimization Process ######### 
    t0 = 0
    step_horizon = 0.01
    N = 100
    optimizer = Optimization(airplane, step_horizon, N)
    optimizer.init_decision_variables()
    optimizer.compute_cost()
    optimizer.init_solver()
    optimizer.define_bound_constraints()

    times, solution_list = optimizer.solve_trajectory(init_states, 
                                               desired_states,
                                               t0)

    #%% Visualize results 
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import animation

    plt.close('all')
    
    t_sim = np.arange(t0, N*step_horizon+step_horizon, step_horizon)
    ref_cmd = 45 * np.ones(len(t_sim))
    #for control_info each index is length [control x N]
    #for state_info each index is length [state x N+1]
    control_info = [control[0] for control in solution_list]
    state_info = [state[1] for state in solution_list]
    
    actual_v = []
    actual_p = []
    actual_r = []
    actual_phi = []
 
     
    for controls in control_info:
        controls = np.asarray(controls)
    
    for state in state_info:
        state = np.asarray(state)
        actual_v.append(state[0,0])
        actual_p.append(state[1,0])
        actual_r.append(state[2,0])
        actual_phi.append(state[3,0])
       
        
    fig2, ax2 = plt.subplots(figsize=(8,8))
    ax2.plot(t_sim, np.rad2deg(state[3,:]), label='Phi (Degrees)')
    ax2.plot(t_sim, ref_cmd)
    ax2.set_xlabel('Time (secs)')
    ax2.set_ylabel('Deg')
    fig2.legend()

       
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t_sim[:-1], np.rad2deg(controls[0,:]), label='aileron')
    axs[1].plot(t_sim[:-1], np.rad2deg(controls[1,:]), label='rudder', color='orange')
    fig.legend()
    axs[0].set_ylabel('Deg/s')
    axs[1].set_ylabel('Deg/s')

 