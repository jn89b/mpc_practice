"""
https://python.plainenglish.io/reference-frame-transformations-in-python-with-numpy-and-matplotlib-6adeb901e0b0

roll,pitch,yaw rotation is inertial frame to body frame

body frame to inertial frame is yaw,pitch,roll rotation

"""

import numpy as np
import math as m
import casadi as ca
from src import rotation_utils as rot
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Effector():
    """
    Currently 2D only 
    """
    def __init__(self, effector_config:dict, use_casadi=False):
        

        self.effector_config = effector_config
        self.effector_range = effector_config['effector_range'] #meters
        self.effector_power = effector_config['effector_power'] #watts
    
        if effector_config['effector_type'] == 'directional':
            self.effector_type = 'triangle'
            self.effector_angle = effector_config['effector_angle']
            self.effector_profile = create_triangle(self.effector_angle, self.effector_range)

        elif effector_config['effector_type'] == 'omnidirectional':
            self.effector_type = 'circle'
            self.effector_angle = 2*np.pi
        else:
            raise Exception("Effector type not recognized")


        if use_casadi == True:
            self.effector_profile = ca.DM(self.effector_profile)
            self.effector_power = ca.DM(self.effector_power)
            self.effector_range = ca.DM(self.effector_range)
            self.effector_angle = ca.DM(self.effector_angle)
            self.use_casadi = True
        else:
            self.use_casadi = False
        
        self.effector_location = None


    def set_effector_location(self, ref_point, ref_angle):
        """
        Set the location of the effector relative to a reference point
        and a reference angle
        """
        if self.use_casadi == True:
            self.effector_location = (rot.rot2d_casadi(ref_angle) @ self.effector_profile) + ref_point
            self.effector_location = ca.horzcat(self.effector_location, ref_point)
        else:
            self.effector_location = (rot.rot2d(ref_angle) @ self.effector_profile.T).T + ref_point
            self.effector_location = np.append(self.effector_location, [ref_point], axis=0)


    def compute_power_density(self, target_distance:float):
        """
        Compute the power density at a given distance from the effector
        """
        if self.effector_type == 'triangle':
            return self.effector_power / (target_distance**2 * 4*np.pi)

        elif self.effector_type == 'circle':
            return self.effector_power / (target_distance**2 * 4*np.pi)

        else:
            raise Exception("Effector type not recognized")
    
    def is_inside_effector(self, target:np.array):
        """check if target is inside the effector"""
        #catch if effector location has not been set
        if self.effector_location is None:
            raise Exception("Effector location has not been set")
            
        point = Point(target)
        polygon = Polygon(self.effector_location)

        if polygon.contains(point):
            return True
        else:
            return False

#create 2d triangle with a given angle and distance
def create_triangle(angle, distance):

    p_x = distance * np.cos(angle)
    p_y = distance * np.sin(angle)
    p_1 = np.array([p_x, p_y])

    p_y = -distance * np.sin(angle)
    p_2 = np.array([p_x, p_y]) #the other point on the the triangle

    #compute the midpoint between p_1 and p_2
    # p_3 = (p_1 + p_2) / 2

    #return as an array of points
    return np.array([p_1, p_2])

def create_semicircle(mid_point, current_psi, diameter, num_points=100):
    r = diameter / 2
    print("mid point: ", mid_point)
    #create a half circle
    t = np.linspace(current_psi - np.pi/2, current_psi + np.pi/2, num_points)
    x = mid_point[0] + r * np.cos(t)
    y = mid_point[1] + r * np.sin(t)
    #return as a list of points
    return np.array([x, y])

# current_psi = np.deg2rad(359)
# effector_angle = np.deg2rad(30)
# effector_range = 1
# ref_point = np.array([0.0, 0.0])

# effector_config = {
#         'effector_range': 1, 
#         'effector_power': 1, 
#         'effector_type': 'directional', 
#         'effector_angle': effector_angle
#         }

# effector = Effector(effector_config)
# effector.set_effector_location(ref_point, current_psi)

# #show effector points
# print("effector points: ", effector.effector_location)

# #check inside polygon 
# target = np.array([0.5, 0.5])
# print("is inside effector: ", effector.is_inside_effector(target))

# #plot the effector points
# import matplotlib.pyplot as plt
# plt.plot(effector.effector_location[:,0], effector.effector_location[:,1], 'o')
# plt.plot(target[0], target[1], 'o')
# plt.show()


