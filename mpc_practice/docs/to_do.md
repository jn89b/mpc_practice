# To Do
- [X] Get Trajectory Control to work 
- [X] Get MPC to work
- [X] Modularize code to take in states of system 


## Get MPC to work
How the algorithm works

### Algorithm
```Latex
- Inputs:
	- Objective $J$
	- Dynamics $f$
	- Horizon time $T$
	- Initial guess of  controls $\hat{u_{1,T}}$
- Set:
	- $u_{1,T}$ = $\hat{u_{1,T}}$
- While not at goal:
	- $x_{init}$ = getCurrentState() 
	- $u_1$ = solveOptimizationProblem($J$, $f$, $x_{init}$, $u_{1,T}$)  
	- $u$ = set($u_1$)
	- applyInput($u$)
```

### Multiple Obstacles
- Define input of obstacle layout
- Override the cost function for calculate obstacles 
- 