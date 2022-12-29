"""
https://python.plainenglish.io/reference-frame-transformations-in-python-with-numpy-and-matplotlib-6adeb901e0b0

roll,pitch,yaw rotation is inertial frame to body frame

body frame to inertial frame is yaw,pitch,roll rotation
g

"""

import numpy as np
import math as m

#2d rotation matrix
def rot2d(psi):
    return np.array([[np.cos(psi), -np.sin(psi)],
                     [np.sin(psi), np.cos(psi)]])

#rotate a point by a given angle
def rotate_point(point, angle):
    return rot2d(angle) @ point

#create 2d triangle with a given angle and distance
def create_triangle(angle, distance):
    p_x = distance * np.cos(angle)
    p_y = distance * np.sin(angle)
    p_1 = np.array([p_x, p_y])

    p_y = -distance * np.sin(angle)
    p_2 = np.array([p_x, p_y]) #the other point on the the triangle

    #compute the midpoint between p_1 and p_2
    p_3 = (p_1 + p_2) / 2
    return p_1, p_2, p_3


def translate_point(point, translation):
    return point + translation


def create_semicircle(mid_point, current_psi, diameter, num_points=100):
    r = diameter / 2
    print("mid point: ", mid_point)
    #create a half circle
    t = np.linspace(current_psi - np.pi/2, current_psi + np.pi/2, num_points)
    x = mid_point[0] + r * np.cos(t)
    y = mid_point[1] + r * np.sin(t)
    #return as a list of points
    return np.array([x, y])



current_psi = np.deg2rad(340)
effector_angle = np.deg2rad(30)
effector_range = 1
ref_point = np.array([0, 0])

triangle_points = create_triangle(effector_angle, effector_range)
translated_points = [translate_point(point, ref_point) for point in triangle_points]
rotated_points = [rotate_point(point, current_psi) for point in translated_points]

#distance between triangle points
#norm distance
d = np.linalg.norm(triangle_points[0] - triangle_points[1])
#create half a circle based on reference point and diameter
semicircle = create_semicircle(rotated_points[2], current_psi, d, 20)


#plot the initial reference point and effector point
import matplotlib.pyplot as plt

plt.plot(ref_point[0], ref_point[1], 'o', label='reference point')
for point in rotated_points:
    plt.plot(point[0], point[1], 'o', label='rotated point', color='red')

plt.plot(semicircle[0], semicircle[1], label='semicircle', color='green')
#plt.plot(rotated_semicircle[0], rotated_semicircle[1], label='rotated_semicircle', color='green')

plt.title('Psi: ' + str(np.rad2deg(current_psi)) + ' degrees')
plt.legend()
plt.show()




