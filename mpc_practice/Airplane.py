import casadi as ca
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time


class AirplaneModel():
    def __init__(self) -> None:
        pass
    
    
    def define_states(self):
        #body velocities
        self.u = ca.SX.sym('u')
        self.v = ca.SX.sym('v')
        self.z = ca.SX.sym('z')
        
        #attitude rates
        self.p = ca.SX.sym('p')
        self.q = ca.SX.sym('q')
        self.r = ca.SX.sym('r')
        
        #attitude
        self.theta = ca.SX.sym('theta')
        self.psi = ca.SX.sym('psi')
        
    
    def define_controls(self):
        pass
    
    def set_state_space(self):
        pass
    
    
