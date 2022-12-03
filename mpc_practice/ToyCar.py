from time import time 
import casadi as ca
import numpy as np

step_horizon = 0.1 #time between steps in seconds 
N = 10 #number of look ahead steps

#intial parameters 
x_init = 0 
y_init = 0 
psi_init = 0

#target parameters
x_target = 15
y_target = 10
psi_target = np.deg2rad(45)

v_max = 5
v_min = -v_max

psi_rate_max = np.deg2rad(30)
psi_rate_min = - psi_rate_max

sim_time = 100      # simulation time


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

def DM2Arr(dm): #convert sparse matrix to full 
    return np.array(dm.full())


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
        self.Q = ca.diagcat(20.0, 20.0, 5.0) # weights for states
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

                
#%% Problem formulation
toy_car = ToyCar()
f = toy_car.set_state_space()

n_states = toy_car.n_states
n_controls = toy_car.n_controls

optimizer = Optimization(toy_car, step_horizon, N)
# optimizer.set_state_space()
optimizer.init_decision_variables()

f = optimizer.f

X = optimizer.X
U = optimizer.U

P = optimizer.P
Q = optimizer.Q
R = optimizer.R

g = optimizer.g
cost_fn = optimizer.cost_fn

# cost_fn = 0  # cost function

for k in range(N):
    states = X[:, k]
    controls = U[:, k]
    state_next = X[:, k+1]
    
    #penalize states and controls for now, can add other stuff too
    cost_fn = cost_fn \
        + (states - P[n_states:]).T @ Q @ (states - P[n_states:]) \
        + controls.T @ R @ controls
    
    ##Runge Kutta
    k1 = f(states, controls)
    k2 = f(states + step_horizon/2*k1, controls)
    k3 = f(states + step_horizon/2*k2, controls)
    k4 = f(states + step_horizon * k3, controls)
    state_next_RK4 = states + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, state_next - state_next_RK4)


OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)

nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
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


solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

## Create mapping between structured and flattened variables
variables_list = [X, U]
variables_name = ['X', 'U']
pack_variables_fn = ca.Function('pack_variables_fn', variables_list, [OPT_variables], variables_name, ['flat'])
unpack_variables_fn = ca.Function('unpack_variables_fn', [OPT_variables], variables_list, ['flat'], variables_name)

##helper functions to flatten and organize constraints
lbx = unpack_variables_fn(flat=-ca.inf)
ubx = unpack_variables_fn(flat=ca.inf)

lbx['U'][0,:] = v_min
ubx['U'][0,:] = v_max

lbx['U'][1,:] = psi_rate_min
ubx['U'][1,:] = psi_rate_max

args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints lower bound
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints upper bound
    'lbx': pack_variables_fn(**lbx)['flat'],
    'ubx': pack_variables_fn(**ubx)['flat'],
}

t0 = 0
state_init = ca.DM([x_init, y_init, psi_init])        # initial state
state_target = ca.DM([x_target, y_target, psi_target])  # target state

# xx = DM(state_init)
t = ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(state_init, 1, N+1)         # initial state full

mpc_iter = 0
cat_states = DM2Arr(X0) # size is number states x number of N x number of mpc_iterations 
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]]) 

solution_list = []

#%%
if __name__ == '__main__':
    main_loop = time()  # return time in sec
    
    #check for goal 
    while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * step_horizon < sim_time):
        
        #get time reference
        t1 = time()
        
        #this is where you can update the target location
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_target   # target state
        )
        
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )

        #this is where we solve
        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))

        #this is where we shift the time step
        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)


        # storing data
        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        # xx ...
        t2 = time()
        print(mpc_iter)
        print(t2-t1)
        times = np.vstack((
            times,
            t2-t1
        ))

        mpc_iter = mpc_iter + 1
        
        solution_list.append(sol)
        
    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_target)

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)
