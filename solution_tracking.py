from casadi import *
from quadcopter import *
from casadi.tools import *
from pylab import *

show_3d = False

# Construct the quadcopter model
model = Quadcopter()

f = model.f

# Construct the explicit ODE
R = jacobian(model.r,model.dx)
rhs = - casadi.solve(R, substitute(model.r, model.dx, 0) ) 
f = SXFunction( [ model.x, model.u ], [ rhs ] )
f.init()

T = 3.0
N = 200

DT = T/N

# This constructs an object that behaves like an MX,
# but has convenient accessors
# e.g  W["X",0] returns the state at the first control interval
# 
# For more information about this CasADi feature,
# you could consult an online tutorial
#    docs.casadi.org    ->   scroll to bottom
#                       ->   click tutorials
#                       ->   tools
#                       ->   structure.pdf
W = struct_symMX([
    (
     entry("X",struct=model.x,repeat=N+1),
     entry("Y",struct=model.x,repeat=N),
     entry("U",struct=model.u,repeat=N)
    )
])  

ts = [0]
t = 0

# Build up the list of constraints
g = []
for i in range(N):
  # Collocation equations
  C = [-2,2]
  xp_j = C[0]*W["X",i] + C[1]*W["Y",i]
  [f_j] = f([ W["Y",i], W["U",i] ])
  g.append(DT*f_j - xp_j)
  
  # Continuity equations
  D = [-1,2]
  xf = D[0]*W["X",i] + D[1]*W["Y",i]
  g.append(xf - W["X",i+1])
  
  t = t + DT
  ts.append(t)
  
g = vertcat(g)

# Construct the reference to track
traj = [ array([sin(i*DT*2*pi/3),i*DT*2*pi/3,0]) for i in range(N) ]

# Construct the objective

R1 = vertcat([ p-t for p,t in zip(W["X",:,"p"],traj) ])
R2 = vertcat(W["U",:])

alpha = 0.05

f = (T/N) * ( mul(R1.T, R1) + alpha*mul(R2.T, R2) )

# Create the NLP
nlp = MXFunction( nlpIn(x=W), nlpOut(f=f, g=g) )
nlp.init()

# Create an IPOPT NLP solver
solver = NlpSolver("ipopt",nlp)
solver.setOption("linear_solver","mumps") # todo: remove
# If we need more than 100 iterations, something is wrong
solver.setOption("max_iter",100)
solver.init()

# All constraints are equality constraints in this case
solver.setInput(0, "lbg")
solver.setInput(0, "ubg")

# Construct and populate the vectors with
# upper and lower simple bounds
#
# lbx is an array in disguise.
# You can view the underlying array as lbx.cat
lbx = W(-inf)
ubx = W(inf)

#  0 <= u_k <= 0.5 
lbx["U",:] =  0
ubx["U",:] =  0.5

#  p_0 = [0,0,0]^T
lbx["X",0,"p"] = 0.0
ubx["X",0,"p"] = 0.0

#  v_0 = [0,0,0]^T
lbx["X",0,"v"] = 0.0
ubx["X",0,"v"] = 0.0

# Construct a vector with the initial guess
x0 = W(0)
x0["X",:] = model.x0
x0["Y",:] = model.x0
x0["U",:] = model.u0

solver.setInput(x0,  "x0")
solver.setInput(lbx, "lbx")
solver.setInput(ubx, "ubx")

# Solve the NLP
solver.evaluate()

# Cast the result vector in a form
# that we can easily access
sol = W(solver.getOutput("x"))


# Save solution to a file
import pickle
pickle.dump(sol,file('tracking.pkl','w'))

X = sol["X",:,"p","x"]
Y = sol["X",:,"p","y"]
Z = sol["X",:,"p","z"]

# 2D plots

figure()
plot(ts,X,label="p_x")
plot(ts,Y,label="p_y")
plot(ts,Z,label="p_z")
plot(ts[:-1],horzcat(traj)[0,:].T,label="p_ref_x")
plot(ts[:-1],horzcat(traj)[1,:].T,label="p_ref_y")
xlabel("Time [s]")
legend(loc="upper left")

title("State trajectories")

figure()
plot(X,Y,'.',label="optimized")
plot(horzcat(traj)[0,:].T,horzcat(traj)[1,:].T,'.',label="reference")
legend()

title("Top down trajectory view")
xlabel("x [m]")
xlabel("y [m]")
axis('equal')

figure()
step(ts, horzcat(sol["U",:]+[ sol["U",-1] ]).T,where='post')
xlabel("Time [s]")
title("Control trajectories")
ylim([-0.1,0.6])

if show_3d:
  # 3D plots
  from mpl_toolkits.mplot3d import Axes3D
  figure()
  ax = gca(projection='3d')

  ax.plot(array(X),array(Y),array(Z),"b.",label="optimized")

  Traj = array(horzcat(traj)) 
  ax.plot(Traj[:,0],Traj[:,1],Traj[:,2],'g.',label="reference")

  # Plot the rotors
  circle = array([ [cos(t),sin(t),0] for t in linspace(0,2*pi) ]).T*0.1

  for p, q in zip(sol["X",::20,"p"],sol["X",::20,"q"]):
    for offset in [ array([[1,0,0]]),array([[0,1,0]]),array([[-1,0,0]]),array([[0,-1,0]]) ]:
      circle_offset = circle + 0.1*offset.T
      circle_3D = mul(quat(*q), circle_offset)
      ax.plot(
        array(p[0]+circle_3D[0,:]).squeeze(),
        array(p[1]+circle_3D[1,:]).squeeze(),
        array(p[2]+circle_3D[2,:]).squeeze(),
        'k'
      )

  ax.set_xlim([-pi,pi])
  ax.set_ylim([0,2*pi])
  ax.set_zlim([-pi,pi])

  legend()

show()
