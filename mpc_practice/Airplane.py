import casadi as ca
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time


class AirplaneModel():
    def __init__(self) -> None:
        self.define_states()
        self.define_controls()
    
    
    def define_states(self):
        """define the states of your system"""
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
        
        self.states = ca.vertcat(
            self.u,
            self.v,
            self.z,
            self.p,
            self.q,
            self.r,
            self.theta,
            self.psi
        )
            
    def define_controls(self):
        """controls for system"""
        self.b1 = ca.horzcat(
            20.3929   -0.4694   -0.2392   -0.7126
        )
        
        self.da = ca.SX.sym('da')
        self.de = ca.SX.sym('de')
        self.dr = ca.SX.sym('dr')
        self.dT = ca.SX.sym('dT')
        self.controls = ca.vertcat(
            self.da,
            self.de, 
            self.dr, 
            self.dT
        )
        
    
    def set_state_space(self):
        """set state space """
        self.a1 = ca.horzcat(
          -0.0404, 0.0618, 0.0501, -0.0000, -0.0005, 0.0000  0.0
        )
        
        self.a2 = ca.horzcat(
        -0.1686, -1.1889, 7.6870, 0, 0.0041, 0.0 , 0.0
        )
        
        self.a3 = ca.horzcat(
            0.1633, -2.6139,   -3.8519,    0.0000,    0.0489,   -0.0000
        ) 
        
        self.a4 = ca.horzcat(
            -0.0000, -0.0000,   -0.0000,   -0.3386,   -0.0474,   -6.5405  
        )
        
        self.a5 = ca.horzcat(
           -0.0000,    0.0000,   -0.0000,   -1.1288,   -0.9149,   -0.3679  
        )
        
        self.a6 = ca.horzcat(
           -0.0000,   -0.0000,   -0.0000,    0.9931,   -0.1763,   -1.2047 
        )
        
        self.a7 = ca.horzcat(
            0,         0,    0.9056,         0,         0,   -0.0000
        )
            
        self.a8 = ca.horzcat(
            0,         0,   -0.0000,         0,    0.9467,   -0.0046

        )
    
        self.A = ca.vertcat(
            self.a1,
            self.a2,
            self.a3,
            self.a4, 
            self.a5,
            self.a6, 
            self.a7, 
            self.a8
        )
    
    
if __name__ == '__main__':
    airplane = AirplaneModel()
    airplane.set_state_space()
    

