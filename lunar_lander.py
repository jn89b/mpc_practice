import numpy as np
import casadi
from casadi import SX, DM
from math import cos, sin
# import cv2

# State Vector Indices
index_mass = 0
index_position_x = 1
index_position_y = 2
index_velocity_x = 3
index_velocity_y = 4
index_zenith_angle = 5
index_angular_velocity = 6

# Control Vector Indices
index_throttle = 0
index_gimbal = 1


def differential_equation(x, u):
    m = x[:,index_mass]
    #px = x[:,index_position_x]
    #py = x[:,index_position_y]
    vx = x[:,index_velocity_x]
    vy = x[:,index_velocity_y]
    angle = x[:,index_zenith_angle]
    omega = x[:,index_angular_velocity]

    throttle = u[:, index_throttle]
    gimbal = u[:, index_gimbal]

    # Model parameters
    effective_exhaust_velocity = 2000.0
    g = 10.0
    max_mass_flow = 0.01

    # See https://en.wikipedia.org/wiki/Specific_impulse#Specific_impulse_as_effective_exhaust_velocity
    max_thrust = effective_exhaust_velocity * max_mass_flow
    thrust = max_thrust * throttle
    mdot = -max_mass_flow * throttle

    # Newton's law: Thrust force and gravity
    ax = casadi.sin(angle + gimbal) * thrust / m
    ay = casadi.cos(angle + gimbal) * thrust / m - g

    # Angular acceleration from gimbaled thrust
    alpha = -2.0 * gimbal * throttle

    dxdt = 0*x
    dxdt[:,index_mass]              = mdot
    dxdt[:,index_position_x]        = vx
    dxdt[:,index_position_y]        = vy
    dxdt[:,index_velocity_x]        = ax
    dxdt[:,index_velocity_y]        = ay
    dxdt[:,index_zenith_angle]      = omega
    dxdt[:,index_angular_velocity]  = alpha
    return dxdt

# def main():
## Problem size
n_states = 7
n_inputs = 2
n_timesteps = 20

## Create Optimization Variables
delta_t = SX.sym('delta_t')
states = SX.sym('state', n_timesteps, n_states)
inputs = SX.sym('input', n_timesteps-1, n_inputs)

## Create mapping between structured and flattened variables
variables_list = [delta_t, states, inputs]
variables_name = ['delta_t', 'states', 'inputs']
variables_flat = casadi.vertcat(*[casadi.reshape(e,-1,1) for e in variables_list])
pack_variables_fn = casadi.Function('pack_variables_fn', variables_list, [variables_flat], variables_name, ['flat'])
unpack_variables_fn = casadi.Function('unpack_variables_fn', [variables_flat], variables_list, ['flat'], variables_name)

## Box constraints
lower_bounds = unpack_variables_fn(flat=-float('inf'))
upper_bounds = unpack_variables_fn(flat=float('inf'))

lower_bounds['delta_t'][:,:] = 1.0e-6 # Time must run forwards
lower_bounds['states'][:,index_mass] = 1.0e-6 # Mass must stay positive

lower_bounds['states'][:,index_zenith_angle] = -2.0 # Limit the maximum rotation from the vertical
upper_bounds['states'][:,index_zenith_angle] =  2.0

lower_bounds['states'][:,index_position_y] = 0.0 # Must stay above the ground

lower_bounds['inputs'][:,index_throttle] = 0.0 # Minimum engine throttle
upper_bounds['inputs'][:,index_throttle] = 1.0 # Maximum engine throttle

lower_bounds['inputs'][:,index_gimbal] = -0.2 # Minimum gimbal angle
upper_bounds['inputs'][:,index_gimbal] =  0.2 # Maximum gimbal angle

## Initial state
lower_bounds['states'][0,index_mass] = 1.0
upper_bounds['states'][0,index_mass] = 1.0

lower_bounds['states'][0,index_position_x] = 100.0
upper_bounds['states'][0,index_position_x] = 100.0

lower_bounds['states'][0,index_position_y] = 100.0
upper_bounds['states'][0,index_position_y] = 100.0

#initial velocity in x must stsay at -40
lower_bounds['states'][0,index_velocity_x] = -40.0
upper_bounds['states'][0,index_velocity_x] = -40.0

lower_bounds['states'][0,index_velocity_y] = 10.0
upper_bounds['states'][0,index_velocity_y] = 10.0

lower_bounds['states'][0,index_zenith_angle] = 0.0
upper_bounds['states'][0,index_zenith_angle] = 0.0

lower_bounds['states'][0,index_angular_velocity] = 0.0
upper_bounds['states'][0,index_angular_velocity] = 0.0

## Final state
lower_bounds['states'][-1,index_position_x] = -1.0
upper_bounds['states'][-1,index_position_x] =  1.0

lower_bounds['states'][-1,index_position_y] =  0.0
upper_bounds['states'][-1,index_position_y] =  0.0

lower_bounds['states'][-1,index_velocity_x] =  0.0
upper_bounds['states'][-1,index_velocity_x] =  0.0

lower_bounds['states'][-1,index_velocity_y] =  0.0
upper_bounds['states'][-1,index_velocity_y] =  0.0

lower_bounds['states'][-1,index_zenith_angle] =  0.0
upper_bounds['states'][-1,index_zenith_angle] =  0.0

lower_bounds['states'][-1,index_angular_velocity] =  0.0
upper_bounds['states'][-1,index_angular_velocity] =  0.0

## Differential equation constraints
# There is no loop here, because it is vectorized.
X0 = states[0:(n_timesteps-1),:]
X1 = states[1:n_timesteps,:]

# Heun's method (some other method, like RK4 could also be used here)
K1 = differential_equation(X0, inputs)
K2 = differential_equation(X0 + delta_t * K1, inputs)
defect = X0 + delta_t*(K1+K2)/2.0 - X1
defect = casadi.reshape(defect, -1, 1)

## Optimization objective
# Maximize final mass
objective = -states[-1, index_mass]
# Also minimize input changes to get a smoother trajectory.
objective += 1e-2 * casadi.sum1(casadi.sum2((inputs[0:-2,:] - inputs[1:-1,:])**2))

## Run optimization
# This uses a naive initialization, it starts every variable at 1.0.
# Some problems require a cleverer initialization, but that is a story for another time.
solver = casadi.nlpsol('solver', 'ipopt', {'x':variables_flat, 'f':objective, 'g':defect})
result = solver(x0=1.0, lbg=0.0, ubg=0.0,
                lbx=pack_variables_fn(**lower_bounds)['flat'],
                ubx=pack_variables_fn(**upper_bounds)['flat'])

results = unpack_variables_fn(flat=result['x'])

#     ## Render video
#     image_height = 1080
#     image_width = 1920
#     flight_path = np.array(results['states'][:, index_position_x:(index_position_y+1)]).T
#     video = cv2.VideoWriter('out.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), 60, (image_width,image_height))
#     for i in range(n_timesteps-1):
#         for interp in np.arange(0.1,1.0,0.2):
#             image = 255 * np.ones((image_height,image_width,3), np.uint8)
#             x_value = results['states'][i,:] + interp * (results['states'][i+1,:] - results['states'][i,:])
#             u_value = results['inputs'][i,:]
#             cz = cos(x_value[0,index_zenith_angle])
#             sz = sin(x_value[0,index_zenith_angle])

#             transform_lander = lambda x: np.array([[cz,sz],[-sz,cz]]) @ x + x_value[0,index_position_x:(index_position_y+1)].T
#             transform = lambda x: [np.array((np.array([[7.0,0],[0,-7.0]]) @ x + np.array([[image_width/2.0],[image_height-30]])).T, np.int32)]

#             # Draw flight path
#             image = cv2.polylines(image, transform(flight_path), False, (255,100,0), 1, cv2.LINE_AA)

#             # Draw lander
#             lander_polygon = transform_lander(np.array([[-1,-3,-2,-1,-1,1,1,2,3,1],[-1,-2,1,1,3,3,1,1,-2,-1]]))
#             image = cv2.polylines(image, transform(lander_polygon), True, (0,0,0), 2, cv2.LINE_AA)

#             # Draw ground
#             image = cv2.polylines(image, transform(np.array([[-50,50],[-2.5,-2.5]])), False, (0,0,0), 2, cv2.LINE_AA)

#             # Draw exhaust
#             exhaust_polygon = np.array([[-0.5,-5*u_value[0,index_throttle]*u_value[0,index_gimbal],0.5],[-1,-1-(5*u_value[0,index_throttle]),-1]])
#             image = cv2.polylines(image, transform(transform_lander(exhaust_polygon)), False, (0,0,255), 2, cv2.LINE_AA)

#             video.write(image)

#     video.release()


# if __name__ == "__main__":
#     main()
