# Senquential-Convex-Programming-for-Trajectory-Planning

We would like to acknowledge Dr. Bassam Alrifaee's research works including his doctoral dissertation entitled as "Networked Model Predictive Control for Vehicle Collision Avoidance" and his GitHub repository "https://github.com/balrifaee/Net-MPC_Collision-Avoidance", which is the basis of our implementation in this repository.

This repo is implemented in Python 3.7 and is the Python version of the afrementioned Dr. Bassam Alrifaee's Github repository. In this repo, we utlize the CVXOPT package and Gurobi solver to address the optimization problem, and amend the function to add noise on the vehicle dynamics model. The decision variable is control input $u$.
