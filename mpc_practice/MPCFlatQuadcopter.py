import casadi as ca
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

"""
Omni does not stack - VICON 

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8442967

Differential Flat Quadcopter, 
system x_dot is flat if there exists flat outputs 

That is the dimension of y and equal to the dimension of u

For quadcopter:
    If I know the x,y,z, yaw position 
    then I can back out and find the states and actions
    for the quadcopter to get there, thrusters included 

For a plane with a carrying load (lots of DF):
    If I know the trajectory of the load
    Then I can back out and find the states and actions 
    of the plane's states and inputs to get the load to move       

The Optimization HAS a model it wants to optimize -> Composition

"""
thrust_max = 1
thrust_min = -1

vel_min = -5
vel_max = 5

psi_min = -np.deg2rad(45)
psi_max = np.deg2rad(45)

class FlatQuadcopter():
    def __init__(self):
        
        #model constants for dj100 from paper
        self.k_x = 1 
        self.k_y = 1 
        self.k_z = 1 
        self.k_psi = np.pi/180
        
        #tau constaints
        self.tau_x = 0.8355
        self.tau_y = 0.7701
        self.tau_z = 0.5013
        self.tau_psi = 0.5142 
        
        self.define_states()
        self.define_controls()
        
    def define_states(self) -> None:
        """
        define the 8 flat states of the quadcopter
        
        [x, y, z, psi, vx, vy, vz, psi_dot]
        
        """
        
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.z = ca.SX.sym('z')
        self.psi = ca.SX.sym('psi')
        
        self.vx = ca.SX.sym('vx')
        self.vy = ca.SX.sym('vy')
        self.vz = ca.SX.sym('vz')
        self.psi_dot = ca.SX.sym('psi')
        
        self.states = ca.vertcat(
            self.x, 
            self.y,
            self.z,
            self.psi,
            self.vx,
            self.vy,
            self.vz, 
            self.psi_dot
        )
        
        #column vector of 3 x 1
        self.n_states = self.states.size()[0] #is a column vector 
        
        
    def define_controls(self) -> None:
        """4 motors"""
        self.u_0 = ca.SX.sym('u_0')
        self.u_1 = ca.SX.sym('u_1')
        self.u_2 = ca.SX.sym('u_2')
        self.u_3 = ca.SX.sym('u_3')
        
        self.controls = ca.vertcat(
            self.u_0,
            self.u_1,
            self.u_2,
            self.u_3
        )
        
        #column vector of 2 x 1
        self.n_controls = self.controls.size()[0] 
        
    def set_state_space(self) -> None:
        #this is where I do the dynamics for state space
        self.z_0 = self.vx * ca.cos(self.psi) - self.vy * ca.sin(self.psi)
        self.z_1 = self.vx * ca.sin(self.psi) + self.vy * ca.cos(self.psi)
        self.z_2 = self.vz
        self.z_3 = self.psi_dot
        
        self.x_ddot = -(self.vx + (self.k_x * self.u_0))
        self.y_ddot = -(self.vy + (self.k_y * self.u_1))
        self.z_ddot = -(self.vz + (self.k_z * self.u_2))
        self.psi_ddot = -(self.psi_dot + (self.k_psi * self.u_3))

        #renamed it as z because I have an x variable, avoid confusion    
        self.z_dot = ca.vertcat(
            self.z_0, 
            self.z_1, 
            self.z_2, 
            self.z_3,
            self.x_ddot, 
            self.y_ddot, 
            self.z_ddot, 
            self.psi_ddot
        )
        
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
        self.Q = ca.diagcat(100.0, 
                            100.0, 
                            100.0,
                            100.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0)
        
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
        self.lbx['X'][4,:] = vel_min
        self.ubx['X'][4,:] = vel_max
        
        self.lbx['X'][5,:] = vel_min 
        self.ubx['X'][5,:] = vel_max
        
        self.lbx['X'][6,:] = vel_min 
        self.ubx['X'][6,:] = vel_max
        
        
        self.lbx['X'][7,:] = psi_min
        self.ubx['X'][7,:] = psi_max
        
        
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

    def solve_trajectory(self,start, goal):
        """solve just the traectory"""
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

        main_loop_time = time()
        ss_error = ca.norm_2(self.state_init - self.state_target)
        
        solution_list.append((self.u, self.X0))

        print('\n\n')
        print('Total time: ', main_loop_time - main_loop)
        print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
        print('final error: ', ss_error)
         
        return time_history, sol


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

# def DM2Arr(dm): #convert sparse matrix to full 
#     return np.array(dm.full())

        
if __name__=='__main__':
    
    flat_quad = FlatQuadcopter()
    f = flat_quad.set_state_space()
    step_horizon = 0.1
    N = 25

    # #intial times
    t0 = 0
    mpc_iter = 0
    sim_time = 50

    x_init = 0
    y_init = 0
    z_init = 1
    psi_init = 0

    x_target = 3
    y_target = 3
    z_target = 2
    psi_target = np.deg2rad(45)

    start = [x_init, y_init, z_init, psi_init,
             0, 0, 0, 0]
    
    end = [x_target, y_target, z_target, psi_target,
           0, 0, 0, 0 ]

    n_states = flat_quad.n_states
    n_controls = flat_quad.n_controls

    #### Optimization Process ######### 
    optimizer = Optimization(flat_quad, step_horizon, N)
    optimizer.init_decision_variables()
    optimizer.compute_cost()
    optimizer.init_solver()
    optimizer.define_bound_constraints()
    
    #%% 
    #### Find soluton time
    times, solution_list = optimizer.solve_mpc(start, end, t0, mpc_iter, sim_time)


    #%% Visualize results 
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import animation

    plt.close('all')
    
    #for control_info each index is length [control x N]
    #for state_info each index is length [state x N+1]
    control_info = [control[0] for control in solution_list]
    state_info = [state[1] for state in solution_list]
    
    actual_x = []
    actual_y = []
    actual_z = []
    actual_psi = []
    
    horizon_x = []
    horizon_y = []
    horizon_z = []
    
    for state in state_info:
        state = np.asarray(state)
        
        actual_x.append(state[0,0])
        actual_y.append(state[1,0])
        actual_z.append(state[2,0])
        
        
        horizon_x.append(state[0,1:])
        horizon_y.append(state[1,1:])
        horizon_z.append(state[2,1:])
    
    
    overall_horizon_x = []
    overall_horizon_y = []
    overall_horizon_z = []
    
    for x,y,z in zip(horizon_x, horizon_y, horizon_z):
        overall_horizon_x.extend(x)
        overall_horizon_y.extend(y)        
        overall_horizon_z.extend(z)
        

    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='3d'))
    ax.plot(x_target, y_target, z_target, 'x', markersize=20, label='goal position')
    ax.plot(actual_x, actual_y, actual_z)
    
    
    ## Animation
    """format position"""
    actual_pos_array = np.array([actual_x, actual_y, actual_z])
    horizon_pos_array = np.array([overall_horizon_x, overall_horizon_y, 
                                  overall_horizon_z])
    

    buffer = 2
    min_x = min(actual_x)
    max_x = max(actual_x)
    
    min_y = min(actual_y)
    max_y = max(actual_y)
    
    min_z = min(actual_z)
    max_z = max(actual_z)

    TIME_SPAN = 5
    position_data = [actual_pos_array, horizon_pos_array]
    labels = ['Actual Position', 'Horizon']
    
    fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='3d'))
    
    ax2.set_xlim([min_x-buffer, max_x+buffer])
    ax2.set_ylim([min_y-buffer, max_y+buffer])
    ax2.set_zlim([min_z-buffer, max_z+buffer])
    
    ax2.plot(x_init, y_init, z_init, 'x', markersize=10, label='start')
    ax2.plot(x_target, y_target, z_target, 'x', markersize=10,label='end')
    
    lines = [ax2.plot([], [], [])[0] for _ in range(len(position_data))] 
    # line2, = ax2.plot(horizon_pos_array[0,0], 
    #                   horizon_pos_array[1,0],
    #                   horizon_pos_array[2,0])
    
    color_list = sns.color_palette("hls", len(position_data))
    # ax.scatter(5,5,1, s=100, c='g')
    for i, line in enumerate(lines):
        line._color = color_list[i]
        #line._color = color_map[uav_list[i]]
        line._linewidth = 5.0
        line.set_label(labels[i])
        
    patches = lines

    def init():
        lines = [ax2.plot(uav[0, 0:1], uav[1, 0:1], uav[2, 0:1])[0] for uav in position_data]
        
        #scatters = [ax.scatter3D(uav[0, 0:1], uav[1, 0:1], uav[2, 0:1])[0] for uav in data]

        return patches
    
    def update_lines(num, dataLines, lines):
        count = 0 
        for i, (line, data) in enumerate(zip(lines, dataLines)):
            #NOTE: there is no .set_data() for 3 dim data...
            time_span = TIME_SPAN
            if num < time_span:
                interval = 0
            else:
                interval = num - time_span
                
            if i == 1:
                line.set_data(data[:2, N*num:N*num+N])
                line.set_3d_properties(data[2, N*num:N*num+N])
            else:
                
                line.set_data(data[:2, interval:num])
                line.set_3d_properties(data[2, interval:num])
            
            count +=1
        
        return patches
    
    # color_list = sns.color_palette("hls", num_columns)
    # patches = set_lines(lines, color_list, column_names)

    ax2.legend()
    line_ani = animation.FuncAnimation(fig2, update_lines, fargs=(position_data, patches), init_func=init,
                                        interval=0.2, blit=True, repeat=True, frames=position_data[0].shape[1]+1)
    # line_ani.save('quad_mpc.gif', writer='imagemagick', fps=60)

    
    
   