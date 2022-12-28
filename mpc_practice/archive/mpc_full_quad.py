import casadi as ca
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

"""
https://www.researchgate.net/publication/311545161_Model_Predictive_Control_for_Trajectory_Tracking_of_Unmanned_Aerial_Vehicles_Using_Robot_Operating_System/link/59682043458515e9af9eba66/download

States:
[
    x, y, z,
    vx, vy, vz,
    phi, theta
]

"""

class QuadCopterModel():
    def __init__(self):
        self.define_states()

    
