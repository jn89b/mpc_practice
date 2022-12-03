import numpy as np 
import casadi 
from casadi import SX,DM

#state vector indices 
index_position_x = 0
index_velocity_x = 1

#control vector indices 
index_throttle = 0

n_states = 2 
n_inputs = 1 
n_timesteps = 40

def differential_equation(x,u):
    """state space differential equation"""
    vx = x[:,index_velocity_x]
    throttle = u[:,index_throttle]
    
    dxdt = 0*x 
    dxdt[:,index_position_x] = vx
    dxdt[:,index_velocity_x] = throttle
    
    return dxdt

##Optimization Variables
delta_t = SX.sym('delta_t')
states = SX.sym('state', n_timesteps, n_states)
inputs = SX.sym('input', n_timesteps-1, n_inputs)

##Mapping between structued and flatted variables
variables_list = [delta_t, states, inputs]
variables_name = ['delta_t', 'states', 'inputs']
variables_flat = casadi.vertcat(*[casadi.reshape(e,-1,1) \
    for e in variables_list])

pack_variables_fn = casadi.Function(
    'pack_variables_fn', 
    variables_list, 
    [variables_flat], 
    variables_name, ['flat'])

unpack_variables_fn = casadi.Function(
    'unpack_variables_fn', 
    [variables_flat], 
    variables_list, 
    ['flat'], 
    variables_name)

##Boundary constraints 
lower_bounds = unpack_variables_fn(flat=-float('inf'))
upper_bounds = unpack_variables_fn(flat=float('inf'))

## Initial state constraints 
lower_bounds['states'][0,index_position_x]= 0.0
upper_bounds['states'][0,index_position_x] = 0.0

lower_bounds['states'][0,index_velocity_x] = 0.0
upper_bounds['states'][0,index_velocity_x] = 0.0

## Final state constraints
lower_bounds['states'][-1, index_position_x] = 1.0
upper_bounds['states'][-1, index_position_x]= 1.0 #want the block to be at final 


lower_bounds['states'][-1,index_velocity_x] = 0.0 #want velocity to be at 0
upper_bounds['states'][-1,index_velocity_x] = 0.0 #want velocity to be at 0 

## Box constraints 
lower_bounds['delta_t'][:,:] = 0.25 #Time must be greater than 0

lower_bounds['states'][:,index_position_x] = -float('inf') #
lower_bounds['states'][:,index_velocity_x] = -float('inf') #

upper_bounds['states'][:,index_position_x] = float('inf')
upper_bounds['states'][:,index_velocity_x] = float('inf')

## DAE differential algebraic constraints
X0 = states[0:(n_timesteps-1),:]
X1 = states[1:n_timesteps,:]

#Using Heun's Method can use RK4 or direct collocation
K1 = differential_equation(X0, inputs)
K2 = differential_equation(X0 + delta_t * K1, inputs)
defect = X0 + delta_t*(K1+K2)/2.0 - X1
defect = casadi.reshape(defect, -1, 1)


## Optimization objective
# Minimize control
#objective = -states[-1, index_mass]
# Also minimize input changes to get a smoother trajectory.
#objective += 1e-2 * casadi.sum1(casadi.sum2((inputs[0:-2,:] - inputs[1:-1,:])**2))

objective = casadi.sumsqr(inputs)

## Run optimization
# This uses a naive initialization, it starts every variable at 1.0.
# Some problems require a cleverer initialization, but that is a story for another time.
solver = casadi.nlpsol(
    'solver', 'ipopt', {'x':variables_flat, 'f':objective, 'g':defect})

result = solver(x0=0.0, lbg=0.0, ubg=0.0,
                lbx=pack_variables_fn(**lower_bounds)['flat'],
                ubx=pack_variables_fn(**upper_bounds)['flat'])

results = unpack_variables_fn(flat=result['x'])
x_position = results['states']
