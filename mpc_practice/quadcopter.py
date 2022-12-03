import casadi as ca
import math 
"""
https://robotics.stackexchange.com/questions/11509/how-to-find-state-space-representation-of-quadcopter

"""    

## GLOBAL VARIABLES

#quadcopter info
## moment of inertiasa
IXX = 0.04 #kg/m^2
IYY = 0.04 #kg/m^2
IZZ = 4.5 #kg/m^2

#masses
M = 1.2 #kg mass 

#propeller info

## Classeess
class Propeller():
    def __init__(self, prop_dia, prop_pitch, thrust_unit='N'):
        self.dia = prop_dia
        self.pitch = prop_pitch
        self.thrust_unit = thrust_unit
        self.rpm = 0 #RPM
        self.thrust = 0

    def set_speed(self,rpm):
        self.rpm = rpm
        # From http://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
        self.thrust = 4.392e-8 * self.rpm * math.pow(self.dia,3.5)/(math.sqrt(self.pitch))
        self.thrust = self.thrust*(4.23e-4 * self.rpm * self.pitch)
        self.thrust = self.thrust*0.101972
            
class QuadCopter():
    
    def __init__(self):
        self.states = ca.MX.sym("X", 12) #X matrix
        self.control_inputs = ca.MX.sym("U", 4) #U matrix

        self.define_states()
        self.define_control_inputs()

    def define_states(self)->None:
        """define 12 DOF state space model of system"""
        
        self.x = self.states[0]
        self.x_dot = self.states[1]
        
        self.y = self.states[2]
        self.y_dot = self.states[3]
        
        self.z = self.states[4]
        self.z_dot = self.states[5]
        
        self.psi =self.states[6]
        self.psi_dot = self.states[7]
        
        self.theta = self.states[8]
        self.theta_dot = self.states[9]
        
        self.psi = self.states[10]
        self.psi_dot = self.states[11] 
        
        
    def define_control_inputs(self) ->None:
        """This is for the quadcopter system 4 motors"""
        self.control_0 = self.control_inputs[0]
        self.control_1 = self.control_inputs[0]
        self.control_2 = self.control_inputs[0]
        self.control_3 = self.control_inputs[0]
        

if __name__ == '__main__':
    quad = QuadCopter()


    """
    testing state space formulation
    - I need to simulate the force output of the propellers 
    for my system base on my motors spinning
    """
    x_dot = ca.MX.sym("X_dot", quad.states.size()) 
    
    g = 9.81 #m/s^2
    
    #this is where I do the dynamics for state space

    u_dot = quad.states[0]
    

    rhs = x_dot
    


    