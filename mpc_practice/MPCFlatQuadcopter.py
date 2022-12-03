from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt

class ToyCar():
    def __init__(self) -> None:
        
        self.define_states()
        self.define_controls()
        self.compute_dynamics()
    
    def define_states(self) -> None:
        """states"""
        self.states = ca.MX.sym("X", 3) #X matrix
        self.x = self.states[0]
        self.y = self.states[1]
        self.psi = self.states[2]
        
    def define_controls(self) -> None:
        """2 inputs"""
        self.controls = ca.MX.sym("U",4)
        self.velocity_command = self.controls[0] #velocity command
        self.psi_command = self.controls[1] #angular velocity command
        
    def compute_dynamics(self) -> None:
        """compute dynamics of system"""
        self.x_dot = self.velocity_command * ca.cos(self.psi)
        self.y_dot = self.velocity_command * ca.sin(self.psi)
        self.dpsi = self.psi_command
        
        #known as RHS 
        self.z_dot = ca.vertcat(
            self.x_dot, 
            self.y_dot, 
            self.dpsi
        )
        
        #xdot = Ax + bu
        self.f = ca.Function('f',
                             [self.states, self.controls],
                             )
        
if __name__ == '__main__':
    toycar = ToyCar()
    
    
        
        