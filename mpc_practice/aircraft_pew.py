import casadi as ca
import numpy as np
from src import MPC, Config, data_utils,Effector
from time import time
"""

Path-following control for small fixed-wing unmanned aerial vehicles under wind disturbances

Included Zurich's equation for psi command 

"""

class AirplaneSimpleModel():
    def __init__(self):
        self.define_states()
        self.define_controls()

    def define_states(self):
        """define the states of your system"""
        #positions ofrom world
        self.x_f = ca.SX.sym('x_f')
        self.y_f = ca.SX.sym('y_f')
        self.z_f = ca.SX.sym('z_f')

        #attitude
        self.phi_f = ca.SX.sym('phi_f')
        self.theta_f = ca.SX.sym('theta_f')
        self.psi_f = ca.SX.sym('psi_f')

        self.states = ca.vertcat(
            self.x_f,
            self.y_f,
            self.z_f,
            self.phi_f,
            self.theta_f,
            self.psi_f
        )

        self.n_states = self.states.size()[0] #is a column vector 

    def define_controls(self):
        """controls for your system"""
        self.u_phi = ca.SX.sym('u_phi')
        self.u_theta = ca.SX.sym('u_theta')
        self.u_psi = ca.SX.sym('u_psi')
        self.v_cmd = ca.SX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_phi,
            self.u_theta,
            self.u_psi,
            self.v_cmd
        )
        self.n_controls = self.controls.size()[0] 

    def set_state_space(self):
        """define the state space of your system"""
        self.g = 9.81 #m/s^2
        #body to inertia frame
        self.x_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.cos(self.psi_f) 
        self.y_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.sin(self.psi_f)
        self.z_fdot = -self.v_cmd * ca.sin(self.theta_f)
        
        self.phi_fdot = self.u_phi
        self.theta_fdot = self.u_theta
        self.psi_fdot = self.u_psi + (self.g * (ca.tan(self.phi_f) / self.v_cmd))

        self.z_dot = ca.vertcat(
            self.x_fdot,
            self.y_fdot,
            self.z_fdot,
            self.phi_fdot,
            self.theta_fdot,
            self.psi_fdot
        )

        #ODE function
        self.function = ca.Function('f', 
            [self.states, self.controls], 
            [self.z_dot])
        
class AircraftEffectorMPC(MPC.MPC):
    def __init__(self, model, dt_val:float, N:int, 
        Q:ca.diagcat, R:ca.diagcat, airplane_params, effector_config=None):
        super().__init__(model, dt_val, N, Q, R)

        self.airplane_params = airplane_params

        if effector_config:
            self.S = 3 #weight for effector
            self.effector_config = effector_config
            self.effector = Effector.Effector(effector_config,True)
            self.effector_history = []


    def add_additional_constraints(self):
        """add additional constraints to the MPC problem"""
        #add control constraints
        self.lbx['U'][0,:] = self.airplane_params['u_phi_min']
        self.ubx['U'][0,:] = self.airplane_params['u_phi_max']

        self.lbx['U'][1,:] = self.airplane_params['u_theta_min']
        self.ubx['U'][1,:] = self.airplane_params['u_theta_max']

        self.lbx['U'][2,:] = self.airplane_params['u_psi_min']
        self.ubx['U'][2,:] = self.airplane_params['u_psi_max']

        self.lbx['U'][3,:] = self.airplane_params['v_cmd_min']
        self.ubx['U'][3,:] = self.airplane_params['v_cmd_max']

        self.lbx['X'][2,:] = self.airplane_params['z_min']
        self.ubx['X'][2,:] = self.airplane_params['z_max']

        self.lbx['X'][3,:] = self.airplane_params['phi_min']
        self.ubx['X'][3,:] = self.airplane_params['phi_max']

        self.lbx['X'][4,:] = self.airplane_params['theta_min']
        self.ubx['X'][4,:] = self.airplane_params['theta_max']

    def compute_cost(self):
        #tired of writing self
        #dynamic constraints 
        self.g = []
        self.g = self.X[:,0] - self.P[:self.n_states]
        
        P = self.P
        Q = self.Q
        R = self.R
        n_states = self.n_states
        
        for k in range(self.N):
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
            
            #add effector cost
            if self.effector_config:
                x_pos = self.X[0,k]
                y_pos = self.X[1,k]
                theta = self.X[4,k]
                psi = self.X[5,k]

                #compute line of sight angle between current position and target
                dx = Config.GOAL_X - x_pos
                dy = Config.GOAL_Y - y_pos
                dz = Config.GOAL_Z - self.X[2,k]

                dtarget = ca.sqrt(dx**2 + dy**2 + dz**2)
                error_dist_factor = 1 - (dtarget/self.effector.effector_range)

                los_psi = ca.atan2(dy,dx)
                error_psi = ca.fabs(los_psi - psi)    
                error_psi_factor = 1 - (error_psi/self.effector.effector_angle)

                los_theta = ca.atan2(dz, dx)
                error_theta = ca.fabs(los_theta - theta)
                error_theta_factor = 1 - (error_theta/self.effector.effector_angle) 

                #compute norm distance between current position and target
                target_distance = ca.sqrt((dx)**2 + (dy)**2 + (dz)**2)
                
                effector_dmg = self.effector.compute_power_density(
                    target_distance, error_psi_factor*error_dist_factor*error_theta_factor, use_casadi=True)
                
                #self.cost_fn = self.cost_fn - error_psi_factor #(self.S * effector_dmg)#minus because we want to maximize
                self.cost_fn = self.cost_fn - (self.S * effector_dmg)#minus because we want to maximize

                #add effector history
                self.effector_history.append([effector_dmg])


        if Config.OBSTACLE_AVOID:
            for k in range(self.N):
                #penalize obtacle distance
                x_pos = self.X[0,k]
                y_pos = self.X[1,k]                
                obs_distance = ca.sqrt((x_pos - Config.OBSTACLE_X)**2 + \
                                        (y_pos - Config.OBSTACLE_Y)**2)
                

                obs_constraint = -obs_distance + (Config.ROBOT_DIAMETER/2) + \
                    (Config.OBSTACLE_DIAMETER/2)

                self.g = ca.vertcat(self.g, obs_constraint) 

        if Config.MULTIPLE_OBSTACLE_AVOID:
            for obstacle in Config.OBSTACLES:
                obs_x = obstacle[0]
                obs_y = obstacle[1]
                obs_diameter = obstacle[2]

                for k in range(self.N):
                    #penalize obtacle distance
                    x_pos = self.X[0,k]
                    y_pos = self.X[1,k]                
                    obs_distance = ca.sqrt((x_pos - obs_x)**2 + \
                                            (y_pos - obs_y)**2)
                    

                    obs_constraint = -obs_distance + (Config.ROBOT_DIAMETER/2) + \
                        (obs_diameter/2)

                    self.g = ca.vertcat(self.g, obs_constraint)


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

        error_tol = 0.5
        while (ca.norm_2(self.state_init - self.state_target) > error_tol) \
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
        'u_psi_min': np.deg2rad(-10),
        'u_psi_max': np.deg2rad(10),
        'u_phi_min': np.deg2rad(-30),
        'u_phi_max': np.deg2rad(30),
        'u_theta_min': np.deg2rad(-20),
        'u_theta_max': np.deg2rad(20),
        'z_min': 0.0,
        'z_max': 100.0,
        'v_cmd_min': 10,
        'v_cmd_max': 30,
        'theta_min': np.deg2rad(-25),
        'theta_max': np.deg2rad(25),
        'phi_min': np.deg2rad(-45),
        'phi_max': np.deg2rad(45),
    }  

    Q = ca.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    R = ca.diag([1.0, 1.0, 1.0, 1.0])

    effector_config = {
            'effector_range': 20, 
            'effector_power': 1, 
            'effector_type': 'directional_3d', 
            'effector_angle': np.deg2rad(60) #double the angle of the cone, this will be divided to two
            }

    mpc_airplane = AircraftEffectorMPC(
        model=airplane,
        N=10,
        dt_val=0.1,
        Q=Q,
        R=R,
        airplane_params=airplane_params,
        effector_config=effector_config
    )

    start = [0, 0, 5, 0, 0, Config.START_PSI]
    goal = [Config.GOAL_X, Config.GOAL_Y, Config.GOAL_Z, 0, 0, 0]

    mpc_airplane.init_decision_variables()
    mpc_airplane.reinit_start_goal(start, goal)
    mpc_airplane.compute_cost()
    mpc_airplane.init_solver()
    mpc_airplane.define_bound_constraints()
    mpc_airplane.add_additional_constraints()

    times, solution_list, obstacle_history = mpc_airplane.solve_mpc(start, goal, 0, 30)

    #%% Data 
    control_info, state_info = data_utils.get_state_control_info(solution_list)
    state_history = data_utils.get_info_history(state_info, mpc_airplane.n_states)
    control_history = data_utils.get_info_history(control_info, mpc_airplane.n_controls)

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

    z_min = min(state_history[2]) - 1
    z_upp = max(state_history[2]) + 1

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.plot(state_history[0], state_history[1], state_history[2], 'o-')
    ax2.plot(x_target, y_target, goal[2], 'x')
    

    #plot obstacles as cylinders
    if Config.MULTIPLE_OBSTACLE_AVOID:
        for obstacle in Config.OBSTACLES:
            x = obstacle[0]
            y = obstacle[1]
            z = 0
            radius = Config.OBSTACLE_DIAMETER/2
            height = z_upp - z_min
            n_space = 10

            ax2.plot_surface(
                x + radius * np.outer(np.cos(np.linspace(0, 2 * np.pi, n_space)), np.ones(n_space)),
                y + radius * np.outer(np.sin(np.linspace(0, 2 * np.pi, n_space)), np.ones(n_space)),
                z + height * np.outer(np.ones(n_space), np.linspace(0, 1, n_space)),
                color='g',
                alpha=0.2
            )

    ax2.set_zlim(z_min, z_upp)
    #set plot to equal aspect ratio
    ax2.set_aspect('equal')
    
    #%% plot psi and theta as subplots with time
    #get time 
    time = times[:-1]
    fig3, ax3 = plt.subplots(4, 1, figsize=(8,8))
    ax3[0].plot(time, np.rad2deg(state_history[4]), '-', label='theta')
    ax3[0].plot(time, np.rad2deg(control_history[1]), '-', label='theta rate command')
    ax3[0].set_xlabel('time [s]')
    ax3[0].set_ylabel('theta')
    ax3[0].legend()


    #plot phi command in fourth subplot
    ax3[1].plot(time, np.rad2deg(state_history[3]), '-', label='phi')
    ax3[1].plot(time, np.rad2deg(control_history[0]), '-', label='phi command')
    ax3[1].set_xlabel('time [s]')
    ax3[1].set_ylabel('phi [deg]')
    ax3[1].legend()
    

    ax3[2].plot(time, np.rad2deg(state_history[5]), '-', label='psi')
    ax3[2].plot(time, np.rad2deg(control_history[2]), '-', label='psi rate command')
    #draw a line at 360 deg
    #ax3[2].plot([time[0], time[-1]], [360, 360], '--', color='k')
    ax3[2].set_xlabel('time [s]')
    ax3[2].set_ylabel('psi')
    ax3[2].legend()

    #plot velocity command in third subplot
    ax3[3].plot(time, control_history[3], '-', label='airspeed command')
    ax3[3].set_xlabel('time [s]')
    ax3[3].set_ylabel('Airspeed [m/s]')
    ax3[3].legend()

    #%% Effector information 
    mpc_airplane.effector.use_casadi = False
    effector_history = []
    num_hits = []

    for i in range(len(times)-1):
        ref_point = [state_history[0][i], state_history[1][i], state_history[2][i]]
        ref_roll = state_history[3][i]
        ref_pitch = state_history[4][i]
        ref_yaw = state_history[5][i]

        mpc_airplane.effector.set_effector_location3d(ref_point, ref_roll, ref_pitch, ref_yaw)

                
        effector_array = np.array([[state_history[0][i], 
                                    state_history[1][i], 
                                    state_history[2][i]] ])
        
        effector_location = np.transpose(mpc_airplane.effector.effector_location)

        effector_array = np.hstack((np.transpose(effector_array), effector_location)) 

        effector_history.append(effector_array)

        #compute line of sight angle between current effector and target
        dx = Config.GOAL_X - state_history[0][i]
        dy = Config.GOAL_Y - state_history[1][i]
        dz = Config.GOAL_Z - state_history[2][i]

        dtarget = np.sqrt(dx**2 + dy**2 + dz**2)

        #check if outside range of effector
        if dtarget >= effector_config['effector_range']:
            num_hits.append(0)
            continue
        
        #compute line of sight angle
        los_psi = np.arctan2(dy, dx)

        #wrap ref yaw to [-2pi, 2pi]
        ref_yaw = np.mod(ref_yaw, 2*np.pi)
        error_psi = los_psi - ref_yaw


        #check if line of sight angle is within effector
        if abs(error_psi) >= effector_config['effector_angle']:
            num_hits.append(0)
            continue

        los_theta = np.arctan2(dz, dx)
        error_theta = los_theta - ref_pitch

        #check if line of sight angle is within effector
        if abs(error_theta) >= effector_config['effector_angle']:
            num_hits.append(0)
            continue

        num_hits.append(1)
    
    print('Total hits: ', np.sum(num_hits))

    #get indices of hits
    hit_indices = np.where(np.array(num_hits) == 1)[0]

    #get indices of misses 
    miss_indices = np.where(np.array(num_hits) == 0)[0]
    

    #%% plot effector 3d
    fig4 = plt.figure(figsize=(8,8))
    ax4 = fig4.add_subplot(111, projection='3d')

    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    ax4.plot(state_history[0], state_history[1], state_history[2], 'o-')
    ax4.plot(x_target, y_target, goal[2], 'x')

    #plot the effector location
    for i in range(len(times)-1):
        if i in hit_indices:
            ax4.plot(effector_history[i][0], effector_history[i][1], effector_history[i][2], color='g')

    fig15, ax15 = plt.subplots(figsize=(8,8))
    ax15.plot(times[:-1], np.rad2deg(state_history[2]), 'o-')
    
    if Config.MULTIPLE_OBSTACLE_AVOID:
        for obstacle in Config.OBSTACLES:
            circle = patches.Circle((obstacle[0], obstacle[1]), obstacle[2]/2, 
                edgecolor='r', facecolor='none')
            ax1.add_patch(circle)


    #set plot to equal aspect ratio
    ax4.set_aspect('equal')
    #ax4.set_zlim(z_min, z_upp)


