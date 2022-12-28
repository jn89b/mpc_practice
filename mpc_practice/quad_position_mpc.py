import quad_config
import casadi as ca

mq = quad_config.mq #mass of quadrotor
g = quad_config.g #gravity
Ix = quad_config.Ix # moment of inertia about x axis
Iy = quad_config.Iy #  moment of inertia about y axis
Iz = quad_config.Iz # moment of inertia about z axis
length = quad_config.la #length of quadrotor arm
b = quad_config.b #thrust coefficient
d = quad_config.d #drag coefficient


"""
file:///home/justin/Downloads/MPC_for_UAV_ROS_v2.pdf

"""

class QuadPositionModel():
    def __init__(self):
        self.define_states()
        self.define_controls()

    def define_states(self):
        """define the states of your system"""
        #body position
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.z = ca.SX.sym('z')

        #body velocities
        self.u = ca.SX.sym('u')
        self.v = ca.SX.sym('v')
        self.z = ca.SX.sym('z')

        #attitudes, psi is held 
        self.theta = ca.SX.sym('theta')
        self.phi = ca.SX.sym('phi')

        self.states = ca.vertcat(
            self.u,
            self.v,
            self.z
        )

        self.n_states = self.states.size()[0]

    def define_controls(self):
        """controls for system"""
        self.psi_desired = ca.SX.sym('psi_desired')
        self.phi_desired = ca.SX.sym('phi_desired')
        #max normalized thrust 
        self.thrust_total = ca.SX.sym('thrust_total')

        self.controls = ca.vertcat(
            self.psi_desired,
            self.phi_desired,
            self.thrust_total
        )

        self.n_controls = self.controls.size()[0]

    def set_state_space(self):
        self.a1 = ca.horzcat(
            0, 0 , 0, 1, 0, 0, 0, 0)
        
        self.a2 = ca.horzcat(
            0, 0, 0, 0, 1, 0, 0, 0)

        self.a3 = ca.horzcat(
            0, 0, 0, 0, 0, 1, 0, 0)
        
        self.a4 = ca.horzcat(
            0, 0, 0, -d, 0, 0, g, 0)

        self.a5 = ca.horzcat(
            0, 0, 0, 0, -d, 0, 0, -g)
        
        self.a6 = ca.horzcat(
            0, 0, 0, 0, 0, -d, 0, 0)

        self.a7 = ca.horzcat(


