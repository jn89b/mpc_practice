import casadi as ca
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        
if __name__=='__main__':
    
    flat_quad = FlatQuadcopter()
    
    """
    testing state space formulation
    - I need to simulate the force output of the propellers 
    for my system base on my motors spinning
    
    For now making the code readible
    """
    
    #this is where I do the dynamics for state space
    x_dot = flat_quad.vx * ca.cos(flat_quad.psi) - flat_quad.vy * ca.sin(flat_quad.psi)
    y_dot = flat_quad.vx * ca.sin(flat_quad.psi) + flat_quad.vy * ca.cos(flat_quad.psi)
    z_dot = flat_quad.vz
    psi_dot = flat_quad.psi_dot
    
    x_ddot = -(flat_quad.vx + (flat_quad.k_x * flat_quad.u_0))
    y_ddot = -(flat_quad.vy + (flat_quad.k_y * flat_quad.u_1))
    z_ddot = -(flat_quad.vz + (flat_quad.k_z * flat_quad.u_2))
    psi_ddot = -(flat_quad.psi_dot + (flat_quad.k_psi * flat_quad.u_3))

    #renamed it as z because I have an x variable, avoid confusion    
    z_dot = ca.vertcat(
        x_dot, y_dot, z_dot, psi_dot,
        x_ddot, y_ddot, z_ddot, psi_ddot
    )
    
    #ODE right hand side function
    f = ca.Function('f', 
                    [flat_quad.states, flat_quad.controls],
                    [z_dot]
                    ) 
    
    
    #Multiple Shooting with Runge Kuta, good old ODE45
    dt = ca.MX.sym("dt")
    k1 = f(flat_quad.states, flat_quad.controls)
    k2 = f(flat_quad.states + dt / 2.0 * k1, flat_quad.controls)
    k3 = f(flat_quad.states + dt / 2.0 * k2, flat_quad.controls)
    k4 = f(flat_quad.states + dt * k3, flat_quad.controls)
    xf = flat_quad.states + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Single step time propagation
    F_RK4 = ca.Function("F_RK4", 
                        [flat_quad.states, flat_quad.controls, dt], [xf], 
                        ['x[k]', 'u[k]', "dt"], 
                        ['x[k+1]'])
    
    
    """
    
    Optimization Problem Formulation
    
    """
    #horizon time and dt value, can make basis on how fast I'm localizing
    N = 15
    dt_val = 0.1
    
    opti = ca.Opti()
    
    #decision variables to send to NLP
    X = opti.variable(flat_quad.states.size()[0], N+1) #allow one more iteration for final
    
    #Going to pull out the solutions from the decision variables 
    x_pos = X[0,:]
    y_pos = X[1,:]
    z_pos = X[2,:]
    psi = X[3,:]
    vx = X[4,:]
    vy = X[5,:]
    vz = X[6,:]
    r = X[7,:] #yaw rate
    
    U = opti.variable(flat_quad.controls.size()[0], N)
    u0 = U[0,:]
    u1 = U[1,:]
    u2 = U[2,:]
    u3 = U[3,:]
    
    #Initiate State parameter
    x0 = opti.parameter(flat_quad.states.size()[0])
    
    #dynamic constraint fro runge kutta
    for k in range(N):
        opti.subject_to(
            X[:,k+1] == F_RK4(X[:,k], U[:,k], dt_val)
            )

    #Control Segments for motor
    MAX_MOTOR = 15
    MAX_VEL = 15
    opti.subject_to(opti.bounded(-MAX_MOTOR, u0, MAX_MOTOR))
    opti.subject_to(opti.bounded(-MAX_MOTOR, u1, MAX_MOTOR))
    opti.subject_to(opti.bounded(-MAX_MOTOR, u2, MAX_MOTOR))
    opti.subject_to(opti.bounded(-MAX_MOTOR, u3, MAX_MOTOR))
    
    opti.subject_to(opti.bounded(0, vx, MAX_VEL))
    opti.subject_to(opti.bounded(0, vy, MAX_VEL))
    opti.subject_to(opti.bounded(0, vz, MAX_VEL))
    #opti.subject_to(opti.bounded(-MAX_VEL, vx, MAX_VEL))
    
    #intial and terminal constraints 
    opti.subject_to(X[:,0] == x0)
    
    #cost function
    goal_x = 25.0
    goal_y = 0.0
    goal_z = 2.0
    goal_psi = np.deg2rad(60)

    #obstacle constraints
    obstacle_x = 1.0
    obstacle_y = 1.0
    obstacle_z = 2.0
    radius_obstacle = 0.5
    safe_margin = 0.1
    
    #margin cost function for obstacle avoidance
    safe_cost = opti.variable(N)
    
    #might do an inflation
    obstacle_distance = ca.sqrt((obstacle_x - x_pos)**2 + \
        (obstacle_y - y_pos)**2 + (obstacle_z - z_pos)**2)
    
    #inequality constraint for obstacle avoidance
    opti.subject_to(obstacle_distance[0:-1].T \
        >= safe_cost + radius_obstacle+safe_margin)
    
    #goal error 
    e_x = (goal_x - x_pos)
    e_y = (goal_y - y_pos)
    e_z = (goal_z - z_pos)
    e_psi = (goal_psi - psi)

    weights = [1, 1 ,1]

    #cost function to minimize position from goal and obstacle avoidance
    cost_value = ca.sumsqr(e_x) + 100*ca.sumsqr(e_y) + \
        ca.sumsqr(e_z) + (weights[-1]* ca.sumsqr(e_psi)) + \
            5000*ca.sumsqr(safe_cost) +  ca.sumsqr(U)

    # cost_value = ca.sumsqr(e_x) + ca.sumsqr(e_y) + \
    #     ca.sumsqr(e_z) + (weights[-1]* ca.sumsqr(e_psi)) + ca.sumsqr(U)

    opti.minimize(cost_value)
    
    opti.solver('ipopt')#, {'ipopt': {'print_level': 0}})
    
    #set initial guesses 
    init_x = 0
    init_y = 0
    init_z = 0.0
    
    opti.set_value(x0, [init_x, init_y, init_z, 0 , 0, 0, 0, 0])
    sol = opti.solve()

    x_traj = sol.value(X).T[:-1]
    u_traj = sol.value(U).T
    t_simulate = np.arange(0, N*dt_val, dt_val)


    sol_traj = pd.DataFrame(np.hstack((x_traj, u_traj)), 
                            columns=['x', 'y', 'z', 'psi', 
                                      'vx', 'vy', 'vz', 'psi_rate',
                                      'u0', 'u1', 'u2', 'u3'])

#%% 
################################################################
# Visualize Results
################################################################
    plt.close('all')
    from matplotlib.patches import Circle
    import mpl_toolkits.mplot3d.art3d as art3d


    buffer_zone = 1

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='3d'))
    
    ax.plot(goal_x, goal_y, goal_z, 'x', markersize=20, label='goal position')
    ax.plot(sol_traj["x"], sol_traj["y"], sol_traj["z"])

    # draw sphere
    # u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    # x = radius_obstacle*np.cos(u)*np.sin(v)
    # y = radius_obstacle*np.sin(u)*np.sin(v)
    # z = radius_obstacle*np.cos(v)
    # # alpha controls opacity
    # ax.plot_surface(x+obstacle_x, y+obstacle_y, z+obstacle_z, color="g", alpha=0.3)
    
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