import casadi as ca
import numpy as np
from src import MPC, Config, data_utils
from time import time
"""

Path-following control for small fixed-wing unmanned aerial vehicles under wind disturbances

"""

class AirplaneSimpleModel():
    def __init__(self) -> None:
        self.define_states()
        self.define_controls()

    def define_states(self):
        """define the states of your system"""
        #positions ofrom world
        self.x_f = ca.SX.sym('x_f')
        self.y_f = ca.SX.sym('y_f')
        self.z_f = ca.SX.sym('z_f')

        #attitude
        self.theta_f = ca.SX.sym('theta_f')
        self.psi_f = ca.SX.sym('psi_f')

        self.states = ca.vertcat(
            self.x_f,
            self.y_f,
            self.z_f,
            self.theta_f,
            self.psi_f
        )

        self.n_states = self.states.size()[0] #is a column vector 

    def define_controls(self):
        """controls for your system"""
        self.u_psi = ca.SX.sym('u_psi')
        self.u_theta = ca.SX.sym('u_theta')
        self.v_cmd = ca.SX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_psi,
            self.u_theta,
            self.v_cmd
        )
        self.n_controls = self.controls.size()[0] 

    def set_state_space(self):
        """define the state space of your system"""
        self.x_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.cos(self.psi_f)
        self.y_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.sin(self.psi_f)
        self.z_fdot = -self.v_cmd * ca.sin(self.theta_f)
        self.theta_fdot = self.u_theta
        self.psi_fdot = self.u_psi

        self.z_dot = ca.vertcat(
            self.x_fdot,
            self.y_fdot,
            self.z_fdot,
            self.theta_fdot,
            self.psi_fdot
        )

        #ODE function
        self.function = ca.Function('f', 
            [self.states, self.controls], 
            [self.z_dot])
        
class AirplaneSimpleModelMPC(MPC.MPC):
    def __init__(self, model, dt_val:float, N:int, 
        Q:ca.diagcat, R:ca.diagcat, airplane_params):
        super().__init__(model, dt_val, N, Q, R)

        self.airplane_params = airplane_params

    def add_additional_constraints(self):
        """add additional constraints to the MPC problem"""
        #add control constraints
        self.lbx['U'][0,:] = self.airplane_params['u_psi_min']
        self.ubx['U'][0,:] = self.airplane_params['u_psi_max']

        self.lbx['U'][1,:] = self.airplane_params['u_theta_min']
        self.ubx['U'][1,:] = self.airplane_params['u_theta_max']

        self.lbx['U'][2,:] = self.airplane_params['v_cmd_min']
        self.ubx['U'][2,:] = self.airplane_params['v_cmd_max']

    def solve_mpc(self, start, goal, t0=0, sim_time = 10):
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
        times = np.array([[0]]) 
        time_history = [self.t0]
        mpc_iter = 0
        obstacle_history = []

        if Config.OBSTACLE_AVOID:
            obs_x = Config.OBSTACLE_X
            obs_y = Config.OBSTACLE_Y

        while (ca.norm_2(self.state_init - self.state_target) > 1e-1) \
            and (self.t0 < sim_time):
            #initial time reference
            t1 = time()

            if Config.MOVING_OBSTACLE:
                obs_x, obs_y = self.move_obstacle(obs_x, obs_y)
                obstacle_history.append((obs_x, obs_y))        

            self.init_solver()
            self.compute_cost()

            if Config.OBSTACLE_AVOID:
                """NEEED TO ADD OBSTACLES IN THE LBG AND UBG"""
                # constraints lower bound added 
                lbg =  ca.DM.zeros((self.n_states*(self.N+1)+self.N, 1))
                # -infinity to minimum marign value for obs avoidance  
                lbg[self.n_states*self.N+n_states:] = -ca.inf 
                
                # constraints upper bound
                ubg  =  ca.DM.zeros((self.n_states*(self.N+1)+self.N, 1))
                #rob_diam/2 + obs_diam/2 #adding inequality constraints at the end 
                ubg[self.n_states*self.N+n_states:] = 0 

            elif Config.MULTIPLE_OBSTACLE_AVOID:
                """NEEED TO ADD OBSTACLES IN THE LBG AND UBG"""
                # constraints lower bound added 
                num_constraints = Config.N_OBSTACLES * self.N
                lbg =  ca.DM.zeros((self.n_states*(self.N+1)+num_constraints, 1))
                # -infinity to minimum marign value for obs avoidance  
                lbg[self.n_states*self.N+n_states:] = -ca.inf 
                
                # constraints upper bound
                ubg  =  ca.DM.zeros((self.n_states*(self.N+1)+num_constraints, 1))
                #rob_diam/2 + obs_diam/2 #adding inequality constraints at the end 
                ubg[self.n_states*self.N+n_states:] = 0

            else:
                lbg = ca.DM.zeros((self.n_states*(self.N+1), 1))
                ubg  =  ca.DM.zeros((self.n_states*(self.N+1), 1))

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
            
            t2 = time()

            #unpack as a matrix
            self.u = ca.reshape(sol['x'][self.n_states * (self.N + 1):], 
                                self.n_controls, self.N)
            
            self.X0 = ca.reshape(sol['x'][: n_states * (self.N+1)], 
                                 self.n_states, self.N+1)


            #this is where we shift the time step
            self.t0, self.state_init, self.u0 = self.shift_timestep(
                self.dt_val, self.t0, self.state_init, self.u, self.f)
            
            # self.target = [goal[0], goal[1], self.state_init[2][0]]

            #shift forward the X0 vector
            self.X0 = ca.horzcat(
                self.X0[:, 1:],
                ca.reshape(self.X0[:, -1], -1, 1)
            )
            
            #won't need this for real time system
            solution_list.append((self.u, self.X0))
            
            # xx ...
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
         
        return time_history, solution_list, obstacle_history                

## start main code
if __name__ == "__main__":

    airplane = AirplaneSimpleModel()
    airplane.set_state_space()

    airplane_params = {
        'u_psi_min': np.deg2rad(-45),
        'u_psi_max': np.deg2rad(45),
        'u_theta_min': np.deg2rad(-30),
        'u_theta_max': np.deg2rad(30),
        'v_cmd_min': 1,
        'v_cmd_max': 5,
    }  

    Q = ca.diag([1.0, 1.0, 1.0, 0.0, 0.0])
    R = ca.diag([1.0, 1.0, 1.0])

    mpc_airplane = AirplaneSimpleModelMPC(
        model=airplane,
        N=10,
        dt_val=0.1,
        Q=Q,
        R=R,
        airplane_params=airplane_params
    )

    start = [0, 0, 0, 0, 0]
    goal = [Config.GOAL_X, Config.GOAL_Y, 10, 0, 0]

    mpc_airplane.init_decision_variables()
    mpc_airplane.reinit_start_goal(start, goal)
    mpc_airplane.compute_cost()
    mpc_airplane.init_solver()
    mpc_airplane.define_bound_constraints()
    mpc_airplane.add_additional_constraints()

    times, solution_list, obstacle_history = mpc_airplane.solve_mpc(start, goal, 0, 10)

    #%% Data 
    control_info, state_info = data_utils.get_state_control_info(solution_list)
    state_history = data_utils.get_info_history(state_info, mpc_airplane.n_states)
    
    #%% Visualize
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import seaborn as sns
    import matplotlib.patches as patches

    plt.close('all')
    
    x_target = goal[0]
    y_target = goal[1]
    
    #%% Plot 2d trajectory
    fig1, ax1 = plt.subplots(figsize=(8,8))
    #set plot to equal aspect ratio
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax1.plot(state_history[0], state_history[1], 'o-')
    ax1.plot(x_target, y_target, 'x')

    if Config.MULTIPLE_OBSTACLE_AVOID:
        for obstacle in Config.OBSTACLES:
            circle = patches.Circle((obstacle[0], obstacle[1]), obstacle[2]/2, 
                edgecolor='r', facecolor='none')
            ax1.add_patch(circle)


    #%% plot 3d trajectory
    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.plot(state_history[0], state_history[1], state_history[2], 'o-')
    ax2.plot(x_target, y_target, goal[2], 'x')
    #set plot to equal aspect ratio
    ax2.set_aspect('equal')
    
