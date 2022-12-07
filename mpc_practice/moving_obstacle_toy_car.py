from time import time 
import casadi as ca
import numpy as np

step_horizon = 0.05 #time between steps in seconds 
N = 50 #number of look ahead steps

#intial parameters
X_BOUND = 20
Y_BOUND = 20  
x_init = 0 
y_init = 0 
psi_init = 0


## GlOBAL OBSTACLE
OBS_X = 1.5
OBS_Y = 1.5

rob_diam = 0.25
obs_diam = 1.0


#target parameters
x_target = 5
y_target = 5
psi_target = np.deg2rad(45)

v_max = 5
v_min = -5

psi_rate_max = np.deg2rad(45)
psi_rate_min = - psi_rate_max

sim_time = 15      # simulation time seconds


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
    

class Optimization():
    """
    Tightly coupled with the quadcopter 
    
    """
    def __init__(self, model, dt_val, N):
        self.model = model
        self.f = model.function        
        
        self.n_states = model.n_states
        self.n_controls = model.n_controls
        
        self.dt_val = dt_val 
        self.N = N

        """this needs to be changed, let user define this"""
        self.Q = ca.diagcat(1.0, 1.0,1.0) # weights for states
        self.R = ca.diagcat(1.0, 1.0) # weights for controls
        
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
        self.pack_variables_fn = ca.Function('pack_variables_fn', self.variables_list, 
                                             [self.OPT_variables], self.variables_name, 
                                             ['flat'])
        
        #function to turn decision variables into respective matrices
        self.unpack_variables_fn = ca.Function('unpack_variables_fn', [self.OPT_variables], 
                                               self.variables_list, ['flat'], self.variables_name)

        ##helper functions to flatten and organize constraints
        self.lbx = self.unpack_variables_fn(flat=-ca.inf)
        self.ubx = self.unpack_variables_fn(flat=ca.inf)

        """
        REFACTOR THIS
        """

        self.lbx['U'][0,:] = v_min
        self.ubx['U'][0,:] = v_max

        self.lbx['U'][1,:] = psi_rate_min
        self.ubx['U'][1,:] = psi_rate_max

    def compute_cost(self):
        """this is where we do integration methods to find cost"""
        #tired of writing self
        P = self.P
        Q = self.Q
        R = self.R
        n_states = self.n_states
        
        """
        going to add a stationary object and see what happens
        I need to abstract this someway to insert this into the MPC 
        """

        for k in range(N):
            states = self.X[:, k]
            controls = self.U[:, k]
            state_next = self.X[:, k+1]

            #penalize states and controls for now, can add other stuff too
            self.cost_fn = self.cost_fn \
                + (states - P[n_states:]).T @ Q @ (states - P[n_states:]) \
                + controls.T @ R @ controls 

            # self.cost_fn =             
            ##Runge Kutta
            k1 = self.f(states, controls)
            k2 = self.f(states + self.dt_val/2*k1, controls)
            k3 = self.f(states + self.dt_val/2*k2, controls)
            k4 = self.f(states + self.dt_val * k3, controls)
            state_next_RK4 = states + (self.dt_val / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.g = ca.vertcat(self.g, state_next - state_next_RK4) #dynamic constraints

        for k in range(N):
            #penalize obtacle distance
            x_pos = self.X[0,k]
            y_pos = self.X[1,k]
            obst_constraint = -ca.sqrt((x_pos - OBS_X)**2 + \
                                       (y_pos - OBS_Y)**2 + \
                                        (rob_diam) + (obs_diam/2))

            self.g = ca.vertcat(self.g, obst_constraint)

            
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
                'print_level': 1,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            # 'jit':True,
            'print_time': 1
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        
    
    def solve_mpc(self,start, goal, t0, mpc_iter, solve_once=False):
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
        print("solving")
        while (ca.norm_2(self.state_init - self.state_target) > 1e-1) \
            and (self.t0 < sim_time):
            print("t0", self.t0)
            #initial time reference
            t1 = time()
            
            ## the arguments are what I care about
            """NEEED TO ADD OBSTACLES IN THE LBG AND UBG"""
            lbg =  ca.DM.zeros((self.n_states*(self.N+1)+self.N, 1))  # constraints lower bound
            lbg[self.n_states*N+n_states:] = -ca.inf # -infinity to minimum marign value for obs avoidance
            
            ubg  =  ca.DM.zeros((self.n_states*(self.N+1)+self.N, 1))  # constraints upper bound
            ubg[self.n_states*N+n_states:] = -rob_diam - obs_diam #adding inequality constraints at the end


            args = {
                'lbg': lbg,  # constraints lower bound
                'ubg': ubg,  # constraints upper bound
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

            #this is if we just one do to compute one trajectory
            if solve_once == True and mpc_iter == 10:
                return time_history, solution_list


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


#%%

if __name__ == '__main__':
    
    toy_car = ToyCar()
    f = toy_car.set_state_space()

    # #intial times
    t0 = 0
    mpc_iter = 0

    start = [x_init, y_init, psi_init]
    end = [x_target, y_target, psi_target]

    n_states = toy_car.n_states
    n_controls = toy_car.n_controls

    #### Optimization Process ######### 
    optimizer = Optimization(toy_car, step_horizon, N)
    optimizer.init_decision_variables()
    optimizer.compute_cost()
    optimizer.init_solver()
    optimizer.define_bound_constraints()


    #### Find soluton time
    times, solution_list = optimizer.solve_mpc(start, end, t0, mpc_iter, False)
    # times, solution_list = optimizer.solve_trajectory(start,end, t0)

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
        actual_y.append(state[1,0])
        actual_psi.append(state[2,0])
        
        horizon_x.append(state[0,1:])
        horizon_y.append(state[1,1:])
        horizon_psi.append(state[2,1:])

    overall_horizon_x = []
    overall_horizon_y = []
    overall_horizon_z = []
    
    for x,y,z in zip(horizon_x, horizon_y, horizon_psi):
        overall_horizon_x.extend(x)
        overall_horizon_y.extend(y)        
        overall_horizon_z.extend(z)

    #actual_vel = [np.array(control[0,0]) for control in control_info]
    #actual_psi = [np.array(control[1,0]) for control in control_info]
    
    horizon_psi_rate = [np.array(control[1,1:]) for control in control_info]
    
    fig1, ax1 = plt.subplots(figsize=(8,8))
    ax1.plot(actual_x, actual_y)
    ax1.plot(x_target, y_target, 'x')
    
    obstacle = plt.Circle( (OBS_X, OBS_Y ),
                                        obs_diam/2 ,
                                        fill = True )

    ax1.add_patch(obstacle)
    
    # fig2, ax2 = plt.subplots(figsize=(8,8))
    # ax2.plot(times[1:], np.rad2deg(actual_psi))
    
    # actual_pos_array = np.transpose(
    #     np.array((actual_x, actual_y, actual_psi), dtype=float))
    
    # horizon_pos_array = np.transpose(
    #     np.array((horizon_x, horizon_y, horizon_psi), dtype=float))
    
    #%% Animate 
 
    
