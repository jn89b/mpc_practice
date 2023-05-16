import casadi as ca
import numpy as np
from src import MPC, Config
from time import time

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

class CarMPC(MPC.MPC):
    """
    Inherit from MPC and add additional constraints
    """
    def __init__(self, model, dt_val:float, N:int, 
        Q:ca.diagcat, R:ca.diagcat, toy_car_params):

        super().__init__(model, dt_val, N, Q, R)

        self.toy_car_params = toy_car_params

    def add_additional_constraints(self):
        """reset boundary constraints"""

        self.lbx['U'][0,:] = self.toy_car_params['v_min']
        self.ubx['U'][0,:] = self.toy_car_params['v_max']

        self.lbx['U'][1,:] = self.toy_car_params['psi_rate_min']
        self.ubx['U'][1,:] = self.toy_car_params['psi_rate_max']


if __name__ == '__main__':
     
    #check Config.py and see of OBSTACLE_AVOID and MULTIPLE_OBSTACLE_AVOID are true
    #if they are both true then raise an error because they are mutually exclusive
    if Config.OBSTACLE_AVOID and Config.MULTIPLE_OBSTACLE_AVOID == True:
        raise ValueError("OBSTACLE_AVOID and MULTIPLE_OBSTACLE_AVOID are mutually exclusive")

    """
    ALL PARAMETERS ARE IN CONFIG.PY
    """
    toy_car = ToyCar()
    toy_car.set_state_space()

    x_init = 0
    y_init = 0
    psi_init = 0
    start = [Config.START_X, Config.START_Y, Config.START_PSI]
    end = [Config.GOAL_X, Config.GOAL_Y, Config.GOAL_PSI]

    N = 15
    dt_val = 0.1

    #set diagonal matrices of Q and R   
    Q = ca.diag([1, 1, 0.0])
    R = ca.diag([1, 1])

    toy_car_params = {
        'v_min': 5,
        'v_max': 15,
        'psi_rate_min': np.deg2rad(-45),
        'psi_rate_max': np.deg2rad(45)
    }

    mpc_car = CarMPC(dt_val=dt_val,model=toy_car, N=N, Q=Q, R=R, 
        toy_car_params=toy_car_params)
    
    mpc_car.init_decision_variables()
    mpc_car.reinit_start_goal(start, end)
    mpc_car.compute_cost()
    mpc_car.init_solver()
    mpc_car.define_bound_constraints()
    mpc_car.add_additional_constraints()

    times, solution_list, obs_history = mpc_car.solve_mpc(start, end, 0, 15)
    print(solution_list)

    #%% Visualize results 
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import seaborn as sns
    import matplotlib.patches as patches

    plt.close('all')
    
    x_target = end[0]
    y_target = end[1]

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


    #actual_vel = [np.array(control[0,0]) for control in control_info]
    #actual_psi = [np.array(control[1,0]) for control in control_info]
    overall_horizon_x = []
    overall_horizon_y = []
    overall_horizon_z = []
    
    for x,y,z in zip(horizon_x, horizon_y, horizon_psi):
        overall_horizon_x.extend(x)
        overall_horizon_y.extend(y)        
    
    horizon_psi_rate = [np.array(control[1,1:]) for control in control_info]
    
    fig1, ax1 = plt.subplots(figsize=(8,8))
    ax1.plot(actual_x, actual_y)
    ax1.plot(x_target, y_target, 'x')
    
    fig15, ax15 = plt.subplots(figsize=(8,8))
    ax15.plot(times[:-1], np.rad2deg(actual_psi))
    
    
    # obstacle = plt.Circle( (Config.OBSTACLE_X, Config.OBSTACLE_Y ),
    #                                     Config.OBSTACLE_DIAMETER/2 ,
    #                                     fill = True )
    
    if Config.MULTIPLE_OBSTACLE_AVOID:
        for obstacle in Config.OBSTACLES:
            circle = patches.Circle((obstacle[0], obstacle[1]), obstacle[2]/2, 
                edgecolor='r', facecolor='none')
            ax1.add_patch(circle)

    # fig2, ax2 = plt.subplots(figsize=(8,8))
    # ax2.plot(times[1:], np.rad2deg(actual_psi))
    
    # actual_pos_array = np.transpose(
    #     np.array((actual_x, actual_y, actual_psi), dtype=float))
    
    # horizon_pos_array = np.transpose(
    #     np.array((horizon_x, horizon_y, horizon_psi), dtype=float))
    

    #%% 
    """format position"""
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
    
    # ax2.add_patch(obstacle)
    
    #ax2.plot(OBS_X, OBS_Y, 'o', markersize=35*obs_diam, label='obstacle')
    # ax2.plot(actual_x, actual_y)
    
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
