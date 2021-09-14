# %%
import numpy as np
from math import *
from autograd import jacobian
import sympy

# %%
class DefaultVehicle():
    def __init__(self):
        self.u0 = 0 # initial steering angle [radians]    
        
        self.x_start = 0 # [m]
        self.y_start = 0 # [m]
        self.heading = 0 # [radians]
        
        #  A list of (x,y) points that make up a piecewise linear curve. 
        # The curve is the vehicles desired trajectory. [m]
        self.referenceTrajectory = np.array([[0,0],[1,0],[3,1]]) # (3,2)
        
        self.speed = 4 # [m/s]
        self.acceleration = 0 # [m/s^2]
        self.Length = .98 # Vehicle length (bumper to bumper)[m]
        self.Width = .88 # Vehicle width [m]
        self.Lf = .34 # Distance between vehicle center and front axle center [m]
        self.Lr = .34 # Distance between vehicle center and rear axle center [m]
        self.Q = 1 # Trajectory deviation weights for MPC
        self.Q_final = 20 # Trajectory deviation weight for the final prediction step
        self.R = 4000 # Steering rate weights for MPC
        
        self.labelOffset = np.array([[0,0]])

# %%
class BicyleModel():
    def __init__(self, is_noise):
        """ the argument x,u,Lf,Lr are to compute the jacobian of this model """
        self.nx = 6
        self.nu = 1
        self.ny = 2
        self.is_noise = is_noise
        # self.jacobian = auto_diff(x,u,Lf,Lr)

    def makeInitState(self, veh):
        self.makeInitStateVector = np.array([veh.x_start,veh.y_start,veh.heading,veh.speed,veh.acceleration,0]).reshape(-1,1) # (6,1)

    def comp_jacobian(self,x,u,Lf,Lr):  
        Ac = np.array([
            [ 0, 0, -x[3]*sin(x[2] + atan((Lr*tan(x[5]))/(Lf + Lr)))*sqrt((Lr**2*tan(x[5])**2)/(Lf + Lr)**2 + 1), cos(x[2] + atan((Lr*tan(x[5]))/(Lf + Lr)))*sqrt((Lr**2*tan(x[5])**2)/(Lf + Lr)**2 + 1), 0, (Lr**2*x[3]*cos(x[2] + atan((Lr*tan(x[5]))/(Lf + Lr)))*tan(x[5])*(tan(x[5])**2 + 1))/(sqrt((Lr**2*tan(x[5])**2)/(Lf + Lr)**2 + 1)*(Lf + Lr)**2) - (Lr*x[3]*sin(x[2] + atan((Lr*tan(x[5]))/(Lf + Lr)))*(tan(x[5])**2 + 1))/(sqrt((Lr**2*tan(x[5])**2)/(Lf + Lr)**2 + 1)*(Lf + Lr))],
            [ 0, 0,  x[3]*cos(x[2] + atan((Lr*tan(x[5]))/(Lf + Lr)))*sqrt((Lr**2*tan(x[5])**2)/(Lf + Lr)**2 + 1), sin(x[2] + atan((Lr*tan(x[5]))/(Lf + Lr)))*sqrt((Lr**2*tan(x[5])**2)/(Lf + Lr)**2 + 1), 0, (Lr*x[3]*cos(x[2] + atan((Lr*tan(x[5]))/(Lf + Lr)))*(tan(x[5])**2 + 1))/(sqrt((Lr**2*tan(x[5])**2)/(Lf + Lr)**2 + 1)*(Lf + Lr)) + (Lr**2*x[3]*sin(x[2] + atan((Lr*tan(x[5]))/(Lf + Lr)))*tan(x[5])*(tan(x[5])**2 + 1))/(sqrt((Lr**2*tan(x[5])**2)/(Lf + Lr)**2 + 1)*(Lf + Lr)**2)],
            [ 0, 0, 0, tan(x[5])/(Lf + Lr), 0, (x[3]*(tan(x[5])**2 + 1))/(Lf + Lr)],
            [ 0, 0, 0, 0, 1, 0],
            [ 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, -10]])
        Bc = np.array([[0],[0],[0],[0],[0],[10]])
        # Ac = jacobian(self.ode(x,t,u,Lf,Lr),x)
        # Bc = jacobian(self.ode(x,t,u,Lf,Lr),u)
        Cc = np.eye(self.ny,self.nx)
        t=0
        Ec = self.ode(x,t,u,Lf,Lr).reshape(-1,1) - Ac@x.reshape([-1,1]) - Bc@u.reshape(-1,1)
        return Ac,Bc,Cc,Ec
    
    def ode(self,x,t,u_ref,Lf,Lr):
        # From Vehicle Dynamics and Control, Rajesh Rajamani, p.24
        # With modifications:
        # * Added steering dynamic: U/U_ref = 1/(1+Ts), T = 0.1 sec.
        # * Added correction for velocity: The measured speed on the buggy is that of the rear axle. 
        # In this model we need the speed of the center.
        # Note: There is a time delay between the controller output and the actuator's reaction. 
        # It is not modeled here, but taken care of in the simulation setup.
        L = Lf+Lr
        R = Lr/L
        phi = x[2]
        a = x[4]
        u = x[5]
        v_rear = x[3]
        v_center = v_rear * sqrt(  1 + (R*tan(u))**2  )
        
        dx = x.copy()
        dx[0] = v_center*cos(phi+atan(R*tan(u)))
        dx[1] = v_center*sin(phi+atan(R*tan(u)))
        dx[2] = v_center*tan(u)*cos(atan(R*tan(u)))/L
        dx[3] = a
        dx[4] = 0
        dx[5] = (u_ref-u)/0.1
        if self.is_noise:
            dx[0] += np.random.normal(0,0.000003)
            dx[1] += np.random.normal(0,0.000003)
        return dx
    
    def odes_(self,t,x,u_ref,Lf,Lr):
        # From Vehicle Dynamics and Control, Rajesh Rajamani, p.24
        # With modifications:
        # * Added steering dynamic: U/U_ref = 1/(1+Ts), T = 0.1 sec.
        # * Added correction for velocity: The measured speed on the buggy is that of the rear axle. 
        # In this model we need the speed of the center.
        # Note: There is a time delay between the controller output and the actuator's reaction. 
        # It is not modeled here, but taken care of in the simulation setup.
        L = Lf+Lr
        R = Lr/L
        phi = x[2]
        a = x[4]
        u = x[5]
        v_rear = x[3]
        v_center = v_rear * sqrt(  1 + (R*tan(u))**2  )
        
        dx = x.copy()
        dx[0] = v_center*cos(phi+atan(R*tan(u))) 
        dx[1] = v_center*sin(phi+atan(R*tan(u))) 
        dx[2] = v_center*tan(u)*cos(atan(R*tan(u)))/L
        dx[3] = a
        dx[4] = 0
        dx[5] = (u_ref-u)/0.1
        if self.is_noise:
            dx[0] += np.random.normal(0,0.000003)
            dx[1] += np.random.normal(0,0.000003)
        return dx
        
# %%
