import casadi as ca
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
"""

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

"""

MAX_MOTOR = 1
MAX_VEL = 5

# #horizon time and dt value, can make basis on how fast I'm localizing
N = 25
dt_val = 0.1

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
        self.states = ca.MX.sym("X", 8) #X matrix
        
        self.x = self.states[0]
        self.y = self.states[1] 
        self.z = self.states[2]
        self.psi = self.states[3]
        
        self.vx = self.states[4]
        self.vy = self.states[5]
        self.vz = self.states[6]
        self.psi_dot = self.states[7]
        
    def define_controls(self) -> None:
        """4 motors"""
        self.controls = ca.MX.sym("U", 4)
        self.u_0 = self.controls[0]
        self.u_1 = self.controls[1]
        self.u_2 = self.controls[2]
        self.u_3 = self.controls[3]

    def set_state_space(self):
        #this is where I do the dynamics for state space
        x_dot = self.vx * ca.cos(self.psi) - self.vy * ca.sin(self.psi)
        y_dot = self.vx * ca.sin(self.psi) + self.vy * ca.cos(self.psi)
        z_dot = self.vz
        psi_dot = self.psi_dot
        
        x_ddot = (-self.vx + (self.k_x * self.u_0))/self.tau_x
        y_ddot = (-self.vy + (self.k_y * self.u_1))/self.tau_y
        z_ddot = (-self.vz + (self.k_z * self.u_2))/self.tau_z
        psi_ddot = (-self.psi_dot + (self.k_psi * self.u_3))/self.tau_psi

        #renamed it as z because I have an x variable, avoid confusion    
        self.z_dot = ca.vertcat(
            x_dot, y_dot, z_dot, psi_dot,
            x_ddot, y_ddot, z_ddot, psi_ddot
        )
        
        #ODE right hand side function
        function = ca.Function('f', 
                        [self.states, self.controls],
                        [self.z_dot]
                        ) 
        
        return function
    
    
    def set_runge_kutta(self, f):
        """set runge kutta equations"""
        dt = ca.MX.sym("dt")
        k1 = f(self.states, self.controls)
        k2 = f(self.states + dt / 2.0 * k1, self.controls)
        k3 = f(self.states + dt / 2.0 * k2, self.controls)
        k4 = f(self.states + dt * k3, self.controls)
        xf = (self.states + dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Single step time propagation
        f_rk4 = ca.Function("F_RK4", 
                            [self.states, self.controls, dt], [xf], 
                            ['x[k]', 'u[k]', "dt"], 
                            ['x[k+1]'])
        
        return f_rk4

class Optimization(ca.Opti):
    """
    Tightly coupled with the quadcopter 
    
    """
    def __init__(self, dt_val, N):
        super().__init__()
        self.dt_val = dt_val
        self.N = N
        self.radius_obstacle = 0.5
        self.safe_margin = 0.5

    def set_init_x(self, model,x_init):
        #Initiate State parameter
        self.x0 = self.parameter(model.states.size()[0])
        self.set_value(self.x0, x_init)
        
    def set_goal_position(self, goal_position):
        self.goal_x = goal_position[0]
        self.goal_y = goal_position[1]
        self.goal_z = goal_position[2]
        self.goal_psi = goal_position[3]
        
        
    def set_obstacles(self,obstacle):
        """need to update this section right here for multiple obstacles"""
        self.obstacle_x = obstacle[0]
        self.obstacle_y = obstacle[1]
        self.obstacle_z = obstacle[2]
        

    def optimize_problem(self, f_rk4, model):    
        #decision variables to send to NLP
        self.X = self.variable(model.states.size()[0], self.N+1) #allow one more iteration for final
        
        #Going to pull out the solutions from the decision variables 
        x_pos = self.X[0,:]
        y_pos = self.X[1,:]
        z_pos = self.X[2,:]
        psi = self.X[3,:]
        vx = self.X[4,:]
        vy = self.X[5,:]
        vz = self.X[6,:]
        r = self.X[7,:] #yaw rate
        
        self.U = self.variable(model.controls.size()[0], self.N)
        u0 = self.U[0]
        u1 = self.U[1]
        u2 = self.U[2]
        u3 = self.U[3]
                
        #dynamic constraint fro runge kutta
        for k in range(N):
            self.subject_to(
                self.X[:,k+1] == f_rk4(self.X[:,k], self.U[:,k], self.dt_val)
                )

        #Control Segments for motor
        self.subject_to(self.bounded(-MAX_MOTOR, u0, MAX_MOTOR))
        self.subject_to(self.bounded(-MAX_MOTOR, u1, MAX_MOTOR))
        self.subject_to(self.bounded(-MAX_MOTOR, u2, MAX_MOTOR))
        self.subject_to(self.bounded(-MAX_MOTOR, u3, MAX_MOTOR))
        
        self.subject_to(self.bounded(-MAX_VEL, vx, MAX_VEL))
        self.subject_to(self.bounded(-MAX_VEL, vy, MAX_VEL))
        self.subject_to(self.bounded(-MAX_VEL, vz, MAX_VEL))
        #self.subject_to(opti.bounded(-MAX_VEL, vx, MAX_VEL))
        
        #intial and terminal constraints 
        self.subject_to(self.X[:,0] == self.x0)
                
        #margin cost function for obstacle avoidance
        safe_cost = self.variable(self.N)
        
        #might do an inflation
        obstacle_distance = ca.sqrt((self.obstacle_x - x_pos)**2 + \
            (self.obstacle_y - y_pos)**2 + (self.obstacle_z - z_pos)**2)
        
        #inequality constraint for obstacle avoidance
        self.subject_to(obstacle_distance[0:-1].T \
            >= safe_cost + self.radius_obstacle+self.safe_margin)
        
        #goal error 
        e_x = (self.goal_x - x_pos)
        e_y = (self.goal_y - y_pos)
        e_z = (self.goal_z - z_pos)
        e_psi = (self.goal_psi - psi)

        weights = [1, 1 ,1]

        #cost function to minimize position from goal and obstacle avoidance
        cost_value = ca.sumsqr(e_x) + ca.sumsqr(e_y) + \
            ca.sumsqr(e_z) + (weights[-1]* ca.sumsqr(e_psi)) + \
                5000*ca.sumsqr(safe_cost) +  ca.sumsqr(self.U)

        self.minimize(cost_value)
        self.solver('ipopt')#, {'ipopt': {'print_level': 0}})
        
    
    def solve_problem(self):
        """solve the problem"""        
        sol = self.solve()
        
        return sol


#t0, current_state, u0, next_states = shift(T, t0, current_state, u_res, x_m, f_np)

def shift_timestep(step_horizon, t0, state_init, u, f):
    
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    #shift time and controls
    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0

    
if __name__=='__main__':
    
    flat_quad = FlatQuadcopter()
    f = flat_quad.set_state_space() #ode equation
    F_RK4 = flat_quad.set_runge_kutta(f) #runge kutta
    
    x_init_pos = [0, 0, 0, 0, 0, 0, 0, 0]
    
    goal_position = [4,4,4,0] #x,y,z,yaw
    
    obstacle_x = 1.0
    obstacle_y = 1.0
    obstacle_z = 2.0
    radius_obstacle = 1.0
    safe_margin = 0.5        

    """
    Optimization Problem Formulation
    """
    
    optimizer = Optimization(dt_val, N)
    optimizer.set_init_x(flat_quad, x_init_pos)
    optimizer.set_goal_position(goal_position)
    optimizer.set_obstacles([obstacle_x, obstacle_y, obstacle_z])

    optimizer.optimize_problem(F_RK4, flat_quad)
    
    """
    Initialize the stuff
    
    While current position and goal position error greater than tolerance 
    or over the time horizon:
        - Get finite trajectory 
        - Update the intial states since we are moving
    
    """
    
    t0 = 0 
    
    init_position = np.array([0,0,0])
    goal_position = np.array([2,2,2])
    
    main_loop = time()

    # for i in range(0,5):
    #being while loop here
    test = np.linalg.norm(init_position - goal_position)
    
    sol = optimizer.solve_problem()
    
    x_traj = sol.value(optimizer.X).T
    u_traj = sol.value(optimizer.U).T
    # t_simulate = np.arange(0, N*dt_val, dt_val)
    
    current_state = np.transpose(x_traj[0,:])
    current_input = np.transpose(u_traj[0,:])

    #shift time forward need to put this in a function
    f_value = f(current_state, current_input)
    next_state = current_state + (dt_val * f_value)

    #swap between current state and next state
    t0 = t0 + dt_val
    current_state = next_state
    
    #renew the intial x0 
    optimizer.set_init_x(flat_quad, current_state)
    



#%% 
################################################################
# Visualize Results
################################################################
    plt.close('all')

    x_traj = sol.value(optimizer.X).T[:-1]
    u_traj = sol.value(optimizer.U).T
    t_simulate = np.arange(0, N*dt_val, dt_val)
        
    sol_traj = pd.DataFrame(np.hstack((x_traj, u_traj)), 
                            columns=['x', 'y', 'z', 'psi', 
                                      'vx', 'vy', 'vz', 'psi_rate',
                                      'u0', 'u1', 'u2', 'u3'])
    
    from matplotlib.patches import Circle
    import mpl_toolkits.mplot3d.art3d as art3d

    goal_x = goal_position[0]
    goal_y = goal_position[1]
    goal_z = goal_position[2]

    buffer_zone = 1

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='3d'))
    
    ax.plot(goal_x, goal_y, goal_z, 'x', markersize=20, label='goal position')
    ax.plot(sol_traj["x"], sol_traj["y"], sol_traj["z"])

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = radius_obstacle*np.cos(u)*np.sin(v)
    y = radius_obstacle*np.sin(u)*np.sin(v)
    z = radius_obstacle*np.cos(v)
    # alpha controls opacity
    ax.plot_surface(x+obstacle_x, y+obstacle_y, z+obstacle_z, color="g", alpha=0.3)
    
    ax.legend(loc="upper left")
    ax.set_xlim([-1, goal_x + buffer_zone])
    ax.set_ylim([-1, goal_y + buffer_zone])
    ax.set_zlim([-1, goal_z + buffer_zone])

    ax.grid()

    fig2, ax2 = plt.subplots(figsize=(8,8))
    ax2.plot(t_simulate, sol_traj["vx"], label='vx')
    ax2.plot(t_simulate, sol_traj["vy"], label='vy')
    ax2.plot(t_simulate, sol_traj["vz"], label='vz')
    ax2.legend(loc="upper left")
    ax2.grid()

    fig3, ax3 = plt.subplots(figsize=(8,8))
    ax3.plot(t_simulate, np.rad2deg(sol_traj['psi']), label='psi')
    ax3.legend(loc="upper left")
    ax3.grid()
    
    fig4, ax4 = plt.subplots(figsize=(8,8))
    ax4.plot(sol_traj['x'], sol_traj['y'], label='position')
    ax4.legend(loc="upper left")
    c = plt.Circle((obstacle_x, obstacle_y), radius=0.2, alpha=0.5)
    ax4.add_patch(c)
    ax4.grid()