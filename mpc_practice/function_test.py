import casadi as ca
import numpy as np
from src import rotation_utils as rot
from src import Effector
from src.StateModels import ToyCar

#using function objects from casadi 

rotation_matrx = ca.Function('rotation_matrx',
         [ca.SX.sym('psi')], 
         [rot.rot2d_casadi(ca.SX.sym('psi'))])



x = ca.SX.sym('x')
y = ca.SX.sym('y')
add_stuff = ca.Function('add_stuff', \
            [x,y], #inputs for function
            [x, x+y], #what the function will output or the evaluation
            ['x', 'y'], #name of inputs
            ['x', 'output'] #name of outputs
        )

#using function objects from casadi
#add stuff

x, output = add_stuff(3, 3)
print(output)

current_psi = np.deg2rad(359)
effector_angle = np.deg2rad(30)
effector_range = 1
ref_point = np.array([0.0, 0.0])

effector_config = {
        'effector_range': 1, 
        'effector_power': 1, 
        'effector_type': 'directional', 
        'effector_angle': effector_angle
        }


toy_car = ToyCar.ToyCar()


effector = Effector.Effector(effector_config, False)
effector.set_effector_location(ref_point, current_psi)

print("effector shape", effector.effector_location.shape)

"""
#Check if target is within effector range 
"""


# Create scalar/matrix symbols
x = ca.MX.sym('x',5)

# Compose into expressions
y = ca.norm_2(x)

# Sensitivity of expression -> new expression
grad_y = ca.gradient(y,x);

# Create a Function to evaluate expression
f = ca.Function('f',[x],[grad_y])

# Evaluate numerically
grad_y_num = f([1,2,3,4,5]);

print(grad_y_num)