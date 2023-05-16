import casadi as cas
from os import system
# Define the state-space system
A = cas.DM([[0.0, 1.0], [-1.0, -1.0]])
B = cas.DM([[0.0], [1.0]])
x = cas.MX.sym('x', 2)
u = cas.MX.sym('u', 1)
xdot = A @ x + B @ u

# Create a CasADi function
ss_func = cas.Function('ss_func', [x, u], [xdot])

# Generate C code for the function
ss_func.generate()

# Compile the generated C code
system('gcc -fPIC -shared -O3 ss_func.c -o ss_func.so')

#load the compiled function
ss_func_compiled = cas.external('ss_func', './ss_func.so')

# Call the compiled function
print(ss_func_compiled([1.0, 2.0], [3.0]))

# RK4 with the compiled function
def rk4_compiled(f, x, u, h):

    k1 = f(x, u)
    k2 = f(x + h / 2 * k1, u)
    k3 = f(x + h / 2 * k2, u)
    k4 = f(x + h * k3, u)

    x_next = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next

#use rk4_compiled
x = cas.MX.sym('x', 2)
u = cas.MX.sym('u', 1)
h = cas.MX.sym('h', 1)
x_next = rk4_compiled(ss_func_compiled, x, u, h)
print(x_next)