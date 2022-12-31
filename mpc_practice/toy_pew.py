import numpy as np
import math as m
import casadi as ca

from src import rotation_utils as rot
from src.StateModels import ToyCar
from src import MPC,Config,Effector

from time import time

class CarMPCEffector(MPC.MPC):
    """
    Inherit from MPC and add additional constraints
    """
    def __init__(self, model, dt_val:float, N:int, 
        Q:ca.diagcat, R:ca.diagcat, toy_car_params, effector_config=None):

        super().__init__(model, dt_val, N, Q, R)

        self.toy_car_params = toy_car_params
        if effector_config:
            self.S = 3 #weight for effector
            self.effector_config = effector_config
            self.effector = Effector.Effector(effector_config,True)
            self.effector_history = []

    def add_additional_constraints(self):
        """reset boundary constraints"""

        self.lbx['U'][0,:] = self.toy_car_params['v_min']
        self.ubx['U'][0,:1] = self.toy_car_params['v_max']

        self.lbx['U'][1,:] = self.toy_car_params['psi_rate_min']
        self.ubx['U'][1,:] = self.toy_car_params['psi_rate_max']

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
                psi = self.X[2,k]

                #compute line of sight angle between current position and target
                dx = Config.GOAL_X - x_pos
                dy = Config.GOAL_Y - y_pos
                dtarget = ca.sqrt(dx**2 + dy**2)
                error_dist_factor = 1 - (dtarget/self.effector.effector_range)

                los_angle = ca.atan2(dy,dx)
                error_psi = ca.fabs(los_angle - psi)    
                error_psi_factor = 1 - (error_psi/self.effector.effector_angle)
                
                #compute norm distance between current position and target
                target_distance = ca.sqrt((x_pos - Config.GOAL_X)**2 + \
                                            (y_pos - Config.GOAL_Y)**2)
                
                effector_dmg = self.effector.compute_power_density(
                    target_distance, error_psi_factor*error_dist_factor, use_casadi=True)
                
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

    
def get_state_control_info(solution):
    """get actual position and control info"""
    control_info = [np.asarray(control[0]) for control in solution]

    state_info = [np.asarray(state[1]) for state in solution]

    return control_info, state_info 

def get_info_history(info: list, n_info: int) -> list:
    """get actual position history"""
    info_history = []

    for i in range(n_info):
        states = []
        for state in info:
            states.append(state[i,0])
        info_history.append(states)

    return info_history

def get_info_horizon(info:list, n_info:int) -> list:
    """get actual position history"""
    info_horizon = []

    for i in range(n_info):
        states = []
        for state in info:
            states.append(state[i, 1:])
        info_horizon.append(states)

    return info_horizon

#%% Implementation
if __name__ == '__main__':
    car = ToyCar.ToyCar()
    car.set_state_space()
        
    start = [Config.START_X, Config.START_Y, Config.START_PSI]
    end = [Config.GOAL_X, Config.GOAL_Y, Config.GOAL_PSI]
    
    x_init = Config.START_X
    y_init = Config.START_Y
    psi_init = Config.START_PSI
    
    N = 10
    dt_val = 0.1

    #set diagonal matrices of Q and R   
    Q = ca.diag([1.0, 1.0, 0.0])
    R = ca.diag([1.0, 1.0])

    toy_car_params = {
        'v_min': 1,
        'v_max': 5,
        'psi_rate_min': np.deg2rad(-45),
        'psi_rate_max': np.deg2rad(45)
    }

    effector_config = {
            'effector_range': 3, 
            'effector_power': 1, 
            'effector_type': 'directional', 
            'effector_angle': np.deg2rad(60) #double the angle of the cone
            }

    mpc_car = CarMPCEffector(dt_val=dt_val,model=car, N=N, Q=Q, R=R,
        toy_car_params=toy_car_params, effector_config=effector_config)

    mpc_car.init_decision_variables()
    mpc_car.reinit_start_goal(start, end)
    mpc_car.compute_cost()
    mpc_car.init_solver()
    mpc_car.define_bound_constraints()
    mpc_car.add_additional_constraints()

    times, solution_list, obs_history = mpc_car.solve_mpc(start, end, 0, 20)
    
    #%% Testing out function parser 
    control_info, state_info = get_state_control_info(solution_list)
    state_history = get_info_history(state_info, mpc_car.n_states)
    control_history = get_info_history(control_info, mpc_car.n_controls)
    
    #horizon info
    state_horizon = get_info_horizon(state_info, mpc_car.n_states)
    control_horizon = get_info_horizon(control_info, mpc_car.n_controls)
    
    #combine state history to numpy array
    state_history = np.asarray(state_history)
    
    #compute effector location history
    effector_history = []
    num_hits = []
    mpc_car.effector.use_casadi = False
    for i in range(len(times)-1):
        ref_point = [state_history[0,i], state_history[1,i]]
        ref_angle = state_history[2,i]
        mpc_car.effector.set_effector_location(ref_point, ref_angle)
        effector_location = np.transpose(mpc_car.effector.effector_location)
        effector_history.append(effector_location)
        
        #check if target is in effector
        target_in_effector = mpc_car.effector.is_inside_effector(end)
        
        if target_in_effector:
            num_hits.append(1)
        else:
            num_hits.append(0)

    #show total hits
    print('Total hits: ', np.sum(num_hits))


    #%% Visualize results 
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import seaborn as sns
    import matplotlib.patches as patches

    plt.close('all')
    
    x_target = end[0]
    y_target = end[1]
    horizon_psi_rate = [np.array(control[1,1:]) for control in control_info]
    
    fig1, ax1 = plt.subplots(figsize=(8,8))
    #set plot to equal aspect ratio
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax1.plot(state_history[0], state_history[1], 'o-')
    ax1.plot(x_target, y_target, 'x')
    
    #plot the effector location
    for i in range(len(times)-1):
        ax1.plot(effector_history[i][0], effector_history[i][1], color='g')

    fig15, ax15 = plt.subplots(figsize=(8,8))
    ax15.plot(times[:-1], np.rad2deg(state_history[2]), 'o-')
    
    if Config.MULTIPLE_OBSTACLE_AVOID:
        for obstacle in Config.OBSTACLES:
            circle = patches.Circle((obstacle[0], obstacle[1]), obstacle[2]/2, 
                edgecolor='r', facecolor='none')
            ax1.add_patch(circle)

    #%% 
    """format position"""
    actual_x = state_history[0]
    actual_y = state_history[1]

    overall_horizon_x = state_horizon[0]
    overall_horizon_y = state_horizon[1]

    actual_pos_array = np.array([actual_x, actual_y])
    horizon_pos_array = np.array([overall_horizon_x, overall_horizon_y])
    
    buffer = 2
    min_x = min(actual_x)
    max_x = max(actual_x)
    
    min_y = min(actual_y)
    max_y = max(actual_y)
    
    TIME_SPAN = 5
    position_data = [actual_pos_array, horizon_pos_array]
    labels = ['Actual Position', 'Horizon']


    fig2, ax2 = plt.subplots(figsize=(6, 6))
    
    #ax2.add_patch(obstacle)
    
    #ax2.plot(OBS_X, OBS_Y, 'o', markersize=35*obs_diam, label='obstacle')
    #ax2.plot(actual_x, actual_y)
    
    ax2.set_xlim([min_x-buffer, max_x+buffer])
    ax2.set_ylim([min_y-buffer, max_y+buffer])
    
    ax2.plot(x_init, y_init, 'x', markersize=10, label='start')
    ax2.plot(x_target, y_target, 'x', markersize=10,label='end')
    
    lines = [ax2.plot([], [], [])[0] for _ in range(len(position_data))] 
    # line2, = ax2.plot(horizon_pos_array[0,0], 
    #                   horizon_pos_array[1,0],
    #                   horizon_pos_array[2,0])
    
    color_list = sns.color_palette("hls", len(position_data))
    # ax.scatter(5,5,1, s=100, c='g')
    for i, line in enumerate(lines):
        line._color = color_list[i]
        #line._color = color_map[uav_list[i]]
        line._linewidth = 2.0
        line.set_label(labels[i])
        
    patches = lines

    def init():
        #lines = [ax2.plot(uav[0, 0:1], uav[1, 0:1], uav[2, 0:1])[0] for uav in position_data]
        lines = [ax2.plot(uav[0, 0:1], uav[1, 0:1])[0] for uav in position_data]

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
                # line.set_3d_properties(data[2, N*num:N*num+N])
            else:
                
                line.set_data(data[:2, interval:num])
                # line.set_3d_properties(data[2, interval:num])
            
            count +=1
        
        return patches
    
    # color_list = sns.color_palette("hls", num_columns)
    # patches = set_lines(lines, color_list, column_names)

    ax2.legend()
    line_ani = animation.FuncAnimation(fig2, update_lines, fargs=(position_data, patches), init_func=init,
                                        interval=5.0, blit=True, repeat=True, frames=position_data[0].shape[1]+1)
    




            


