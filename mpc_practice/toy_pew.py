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
            self.effector_config = effector_config
            self.effector = Effector.Effector(effector_config,True)

    def add_additional_constraints(self):
        """reset boundary constraints"""

        self.lbx['U'][0,:] = self.toy_car_params['v_min']
        self.ubx['U'][0,:] = self.toy_car_params['v_max']

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

                #convert to ca arrau
                ref_point = ca.horzcat(self.X[:-1,k])
                
                psi = self.X[2,k]
                self.effector.set_effector_location(ref_point, psi)
                #check if target location in beam profile
                # if self.effector.is_inside_effector(
                #     np.array([Config.GOAL_X, Config.GOAL_Y])):

                #     print("yes")

                #self.cost_fn = self.cost_fn + self.effector.compute_cost(states, controls)
         

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


if __name__ == '__main__':
    car = ToyCar.ToyCar()
    car.set_state_space()

    x_init = 0
    y_init = 0
    psi_init = 0
    start = [Config.START_X, Config.START_Y, Config.START_PSI]
    end = [Config.GOAL_X, Config.GOAL_Y, Config.GOAL_PSI]

    N = 15
    dt_val = 0.1

    #set diagonal matrices of Q and R   
    Q = ca.diag([1, 1, 1.0])
    R = ca.diag([1, 1])

    toy_car_params = {
        'v_min': 1,
        'v_max': 3,
        'psi_rate_min': np.deg2rad(-45),
        'psi_rate_max': np.deg2rad(45)
    }

    effector_config = {
            'effector_range': 1, 
            'effector_power': 1, 
            'effector_type': 'directional', 
            'effector_angle': np.deg2rad(30) #double the angle of the cone
            }

    mpc_car = CarMPCEffector(dt_val=dt_val,model=car, N=N, Q=Q, R=R,
        toy_car_params=toy_car_params, effector_config=effector_config)

    mpc_car.init_decision_variables()
    mpc_car.reinit_start_goal(start, end)
    mpc_car.compute_cost()
    mpc_car.init_solver()
    mpc_car.define_bound_constraints()
    mpc_car.add_additional_constraints()

    times, solution_list, obs_history = mpc_car.solve_mpc(start, end, 0, 15)





