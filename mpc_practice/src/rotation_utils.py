import numpy as np
import casadi as ca

def rot2d(psi):
    return np.array([[np.cos(psi), -np.sin(psi)],
                     [np.sin(psi), np.cos(psi)]])


def rot2d_casadi(psi):
    return ca.vertcat(
        ca.horzcat(ca.cos(psi), -ca.sin(psi)),
        ca.horzcat(ca.sin(psi), ca.cos(psi))
    )



# #example of rot2d_casadi 
# ref_angle = ca.SX.sym('psi')
# #test = rot2d_casadi(psi)

# ref_point = ca.vertcat(0, 0)
 
# effector_profile = np.array([[0, 0],
#                                 [1, 0],
#                                 [0, 1]])

# test = (rot2d_casadi(ref_angle) @ effector_profile)
# print(test)