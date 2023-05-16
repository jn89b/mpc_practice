import casadi as ca 
import numpy as np
import matplotlib.pyplot as plt

def construct_polynomial_basis(degree_polynomial:int,
                               collocation_method:str="radau"):
    
    """
    http://archive.control.lth.se/media/Education/DoctorateProgram/2011/OptimizationWithCasadi/mx_exercises.pdf
    """
    
    tau_root = [0] + ca.collocation_points(
        degree_polynomial, collocation_method)
    
    # Coefficients of the collocation equation
    C = np.zeros((degree_polynomial+1,degree_polynomial+1))
    
    # Coefficients of the continuity equation
    D = np.zeros(degree_polynomial+1)

    # Coefficients of the quadrature function
    F = np.zeros(degree_polynomial+1)

    # Construct polynomial basis
    for j in range(degree_polynomial+1):
        #refer to equation 7 in the paper
        p = np.poly1d([1])
        for r in range(degree_polynomial+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        #equation 9 in the paper
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        # this is equation 10 in the paper
        pder = np.polyder(p)
        for r in range(degree_polynomial+1):
            C[j,r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        #this is the gauss quadrature
        pint = np.polyint(p)
        F[j] = pint(1.0)

    return C, D, F, tau_root


def define_collocation_time_points(n_steps:int, 
                                   tfinal:float,
                                   h:float, 
                                   degree_polynomial:int):


    #refer to equation 6 in the paper
    T = np.zeros((n_steps, degree_polynomial+1))
    for k in range(n_steps):
        for j in range(degree_polynomial+1):
            T[k,j] = h*(k + tau_root[j])
    
    return T

if __name__ == "__main__":
    
    degree_polynomial = 3

    #control discretization
    nk = 20

    #end time
    tf = 10.0

    inter_axle = 0.5

    #step size 
    h = tf/nk

    C, D, F, tau_root = construct_polynomial_basis(degree_polynomial)

    #states 
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    phi = ca.SX.sym("phi")
    delta = ca.SX.sym("delta")
    vx = ca.SX.sym("vx")
    theta = ca.SX.sym("theta")
    
    z = ca.vertcat(x, y, phi, delta, vx, theta)

    #controls 
    alphaux = ca.SX.sym("alphaux")
    aux = ca.SX.sym("aux")
    dt = ca.SX.sym("dt")

    u = ca.vertcat(alphaux, aux, dt)

    zdot = ca.vertcat(vx*ca.cos(phi), 
                      vx*ca.sin(phi), 
                      (vx/inter_axle)*ca.tan(delta), 
                      alphaux, 
                      aux, 
                      vx*dt)
    
    #



