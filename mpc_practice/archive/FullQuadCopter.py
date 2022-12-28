import casadi as ca
import numpy as np 
import math
from time import time 


m = 1          # Mass of the quadrotor [kg]
g = 9.8         # Gravity [m/s^2]
Ix = 4e-3       # Moment of inertia about Bx axis [kg.m^2]
Iy = 4e-3       # Moment of inertia about By axis [kg.m^2]
Iz = 8.4e-3     # Moment of inertia about Bz axis [kg.m^2]
L = 0.2        # Quadrotor arm length [m]
b = 29e-6       # Thrust coefficient [N.s^2]
d = 1.1e-6      # Drag coefficient [N.m.s^2]

U1 = g 

step_horizon = 0.1 #time between steps in seconds
N = 25   #horizon length

thrust_max = 5# Newtons
thrust_min = -5
 
class FullQuad():
    """
    Full 6 DOF Quadcopter
    https://robotics.stackexchange.com/questions/11509/how-to-find-state-space-representation-of-quadcopter
    """
    
    def __init__(self):
        self.define_states()
        self.set_helper_variables()
        self.define_controls()
        
    def define_states(self):
        """12 state space system of quadcopter"""
        
        self.x = ca.SX.sym('x')
        self.x_dot = ca.SX.sym('x_dot')
        
        self.y = ca.SX.sym('y')
        self.y_dot = ca.SX.sym('y_dot')
        
        self.z = ca.SX.sym('z')
        self.z_dot = ca.SX.sym('z_dot')
        
        self.phi = ca.SX.sym('phi')
        self.phi_dot = ca.SX.sym('phi_dot')
        
        self.theta = ca.SX.sym('theta')
        self.theta_dot = ca.SX.sym('theta_dot')
        
        self.psi = ca.SX.sym('psi')
        self.psi_dot = ca.SX.sym('psi_dot')
        
        self.states = ca.vertcat(
            self.x,
            self.x_dot, 
            self.y, 
            self.y_dot,
            self.z, 
            self.z_dot,
            self.phi,
            self.phi_dot,
            self.theta, 
            self.theta_dot,
            self.psi,
            self.psi_dot
            )
        
        self.n_states = self.states.size()[0] #is a column vector 
 
        
    def define_controls(self):
        """4 motors"""
        self.u_1 = ca.SX.sym('u_1')
        self.u_2 = ca.SX.sym('u_2')
        self.u_3 = ca.SX.sym('u_3')
        self.u_4 = ca.SX.sym('u_4')
        
        self.controls = ca.vertcat(
            self.u_1,
            self.u_2,
            self.u_3,
            self.u_4
        )
        
        self.n_controls = self.controls.size()[0] 

    def set_helper_variables(self):
        """set formulation of z_dot = ax + bu"""
        self.a1 = g/self.x
        self.a2 = self.psi_dot * (Iy - Iz)/Ix
        self.a3 = self.psi_dot * (Iz - Ix)/Iy
        self.a4 = self.psi_dot * (Ix - Iy)/Iz
        
        self.b1 = -((ca.cos(self.psi) * ca.sin(self.theta) * ca.cos(self.psi)) + \
            (ca.sin(self.psi) * ca.sin(self.psi))) / m
        
        self.b2 = -((ca.cos(self.psi) * ca.sin(self.theta) * ca.sin(self.psi)) + \
            (ca.sin(self.psi) * ca.cos(self.psi))) / m    
        
        self.b3 = -(ca.cos(self.psi) * ca.sin(self.theta))/m
        
        self.b4 = L/Ix
        self.b5 = L/Iy 
        self.b6 = 1/Iz
        
    def set_state_space(self):
        """z_dot = ax + bu"""
        self.z_0 = self.states[0] #x_dot
        #self.z_1 = self.b1 * self.u_1 #x_ddot
        self.z_1 = ca.sin(self.u_1) * U1 / m
        
        self.z_2 = self.states[3] #y_dot
        #self.z_3 = self.b2 * self.u_1 #y_ddot
        self.z_3 = ca.sin(self.u_1) * U1 / m
        
        self.z_4 = self.states[5] #z_dot
        #self.z_5 = (self.states[0] * self.a1) + (self.u_1 * self.b3) #z_ddot
        self.z_5 = (g - self.u_1)/ m#z_ddot
        
        self.z_6 = self.states[7] #phi_dot
        #self.z_7 = (self.states[9] * self.a2) + (self.u_2 * self.b4)
        self.z_7 = (self.states[9] * self.states[11] * (Iy- Iz)) + (self.u_2 * self.b4) #phi_ddot
        
        self.z_8 = self.states[9] #theta_dot
        #self.z_9 = (self.states[7] * self.a3) + (self.u_3 * self.b5) #theta_ddot
        self.z_9 = (self.states[7] * self.states[11] * (Iz-Ix)) + (self.u_3 * self.b5) #theta_ddot
        
        self.z_10 = self.states[11] #psi_dot 
        #self.z_11 = (self.states[9]*self.a4) + (self.u_4 * self.b6)
        self.z_11 = (self.states[9]*self.states[7] * (Ix-Iy)) + (self.u_4 * self.b6)
        
        self.z_dot = ca.vertcat(
            self.z_0,
            self.z_1,
            self.z_2,
            self.z_3,
            self.z_4,
            self.z_5, 
            self.z_6,
            self.z_7,
            self.z_8,
            self.z_9,
            self.z_10,
            self.z_11
        )
        
        self.function = ca.Function('f',
                                    [self.states, self.controls],
                                    [self.z_dot]
                                    )
        

class Optimization():
    
    def __init__(self, model, dt_val, N):
        self.model = model
        self.f = model.function        
        
        self.n_states = model.n_states
        self.n_controls = model.n_controls
        
        self.dt_val = dt_val 
        self.N = N
        
        """this needs to be changed, let user define this"""
        self.Q = ca.diagcat(20.0, 
                            20.0, 
                            20.0,
                            20.0,
                            20.0,
                            20.0,
                            20.0,
                            20.0,
                            20.0,
                            20.0,
                            20.0,
                            20.0) # weights for states
        
        self.R = ca.diagcat(1.0, 
                            1.0,
                            1.0,
                            1.0) # weights for controls
        
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
        self.lbx['U'][0,:] = thrust_min
        self.ubx['U'][0,:] = thrust_max

        self.lbx['U'][1,:] = thrust_min
        self.ubx['U'][1,:] = thrust_max

        self.lbx['U'][2,:] = thrust_min
        self.ubx['U'][2,:] = thrust_max

        self.lbx['U'][3,:] = thrust_min
        self.ubx['U'][3,:] = thrust_max

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
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            # 'jit':True,
            'print_time': 0
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)


    def solve_mpc(self,start, goal, t0, mpc_iter, sim_time):
        """main loop to solve for MPC"""
        #Packing the solution into a single row vector
        main_loop = time()  # return time in sec
        n_states = self.n_states
        n_controls = self.n_controls
        solution_list = []
        

        self.state_init = ca.DM(start)        # initial state
        self.state_target = ca.DM(goal)  # target state
        self.t0 = t0
        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(self.state_init, 1, self.N+1)         # initial state full
        self.mpc_iter = mpc_iter
        times = np.array([[0]]) 
        
        time_history = [self.t0]


        while (ca.norm_2(self.state_init - self.state_target) > 1e-1) \
            and (self.t0 < sim_time):
            print("t0", self.t0)
            #initial time reference
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

            #unpack as a matrix
            self.u = ca.reshape(sol['x'][self.n_states * (self.N + 1):], 
                                self.n_controls, self.N)
            
            self.X0 = ca.reshape(sol['x'][: n_states * (self.N+1)], 
                                 self.n_states, self.N+1)

            #this is where we shift the time step
            self.t0, self.state_init, self.u0 = shift_timestep(
                self.dt_val, self.t0, self.state_init, self.u, self.f)

            t2 = time()

            #shift forward the X0 vector
            self.X0 = ca.horzcat(
                self.X0[:, 1:],
                ca.reshape(self.X0[:, -1], -1, 1)
            )
            
            #won't need this for real time system
            solution_list.append((self.u, self.X0))
            
            # xx ...
            print(mpc_iter)
            print(t2-t1)
            times = np.vstack((
                times,
                t2-t1
            ))
            time_history.append(self.t0)

            mpc_iter = mpc_iter + 1

        main_loop_time = time()
        ss_error = ca.norm_2(self.state_init - self.state_target)

        print('\n\n')
        print('Total time: ', main_loop_time - main_loop)
        print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
        print('final error: ', ss_error)
         
        return time_history, solution_list

def shift_timestep(step_horizon, t_init, state_init, u, f):
    """
    we shift the time horizon over one unit of the step horizon
    reinitialize the time, new state (which is next state), and 
    the control parameters
    """
    f_value = f(state_init, u[:, 0]) #calls out the runge kutta
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    #shift time and controls
    next_t = t_init + step_horizon
    next_control = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return next_t, next_state, next_control

def DM2Arr(dm): #convert sparse matrix to full 
    return np.array(dm.full())


if __name__ == '__main__':
    fullquad = FullQuad()
    f = fullquad.set_state_space()
   
    # #intial times
    t0 = 0
    mpc_iter = 0
    sim_time = 10
    
    x_init = 0
    y_init = 0 
    z_init = 0
    
    x_end = 2
    y_end = 2
    z_end = 2
    
    start = [x_init,
             0,
             y_init, 
             0, 
             z_init,
             0,
             0,
             0,
             0,
             0,
             0,
             0]
    
    end = [x_end,
             0,
             y_end, 
             0, 
             z_end,
             0,
             0,
             0,
             0,
             0,
             0,
             0]
    
    n_states = fullquad.n_states
    n_controls = fullquad.n_controls

    #### Optimization Process ######### 
    optimizer = Optimization(fullquad, step_horizon, N)
    optimizer.init_decision_variables()
    optimizer.compute_cost()
    optimizer.init_solver()
    optimizer.define_bound_constraints()


    #### Find soluton time
    times, solution_list = optimizer.solve_mpc(start, end, t0, mpc_iter, sim_time)

    #%% Visualize results 
    import matplotlib.pyplot as plt
    from matplotlib import animation

    plt.close('all')
    
    #for control_info each index is length [control x N]
    #for state_info each index is length [state x N+1]
    control_info = [control[0] for control in solution_list]
    state_info = [state[1] for state in solution_list]
    
    actual_x = []
    actual_y = []
    actual_psi = []
    
    horizon_x = []
    horizon_y = []
    horizon_psi = []
    
    for state in state_info:
        state = np.asarray(state)
        
        actual_x.append(state[0,0])
        actual_y.append(state[2,0])
        actual_psi.append(state[4,0 ])
        
        horizon_x.append(state[0,1:])
        horizon_y.append(state[1,1:])
        horizon_psi.append(state[2,1:])
    
    #actual_vel = [np.array(control[0,0]) for control in control_info]
    #actual_psi = [np.array(control[1,0]) for control in control_info]
    
    horizon_psi_rate = [np.array(control[1,1:]) for control in control_info]
    
    fig1, ax1 = plt.subplots(figsize=(8,8))
    ax1.plot(actual_x, actual_y)
    ax1.plot(x_end, y_end, 'x')
    
    
    fig2, ax2 = plt.subplots(figsize=(8,8))
    ax2.plot(times[1:], np.rad2deg(actual_psi))
    
    actual_pos_array = np.transpose(
        np.array((actual_x, actual_y, actual_psi), dtype=float))
    
    horizon_pos_array = np.transpose(
        np.array((horizon_x, horizon_y, horizon_psi), dtype=float))
    