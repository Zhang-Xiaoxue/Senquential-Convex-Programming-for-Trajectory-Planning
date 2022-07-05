# %%
import numpy as np
from math import floor,cos,sin
import ode
import scipy as sci
from sympy import Matrix
from scipy import linalg

from Scenarios import Indices
from SampleReferTraj import sampleReferenceTrajectory

# %%
class IterClass():
    def __init__(self, scenario, x_measured, u_path, obstacleState, uMax):
        """ 
        self.x0: array (nVeh, 6)
        """
        idx = Indices()
        self.x0 = np.zeros([scenario.nVeh,scenario.model.nx])
        self.u0 = np.zeros([scenario.nVeh,scenario.model.nu])
        steps = 10
        self.MPC_delay_compensation_trajectory = np.zeros([steps,scenario.model.nx,scenario.nVeh])
        assert (u_path.shape[1] * scenario.tick_length - (scenario.delay_x + scenario.dt + scenario.delay_u) < 1e-10)

        for v in range(scenario.nVeh):
            Y = sci.integrate.odeint(scenario.model.ode, 
                                     x_measured[v,:],
                                     np.linspace(0, scenario.delay_x+scenario.dt+scenario.delay_u,steps), 
                                     args=(u_path[v,-1],scenario.Lf[v],scenario.Lr[v]))

            self.x0[v,:] = Y[-1,:]
            self.u0[v,:] = u_path[v,-1]
            self.MPC_delay_compensation_trajectory[:,:,v] = Y
        
        self.ReferenceTrajectoryPoints = np.zeros([scenario.Hp,2,scenario.nVeh])
        for v in range(scenario.nVeh):
            # Find equidistant points on the reference trajectory.
            self.ReferenceTrajectoryPoints[:,:,v] = sampleReferenceTrajectory(
                scenario.Hp,  # number of prediction steps
                scenario.referenceTrajectories[v], 
                self.x0[v,idx.x],  # vehicle position x
                self.x0[v,idx.y],  # vehicle position y
                self.x0[v,idx.speed]*scenario.dt)  # distance traveled in one timestep
        
        if scenario.nObst:
            # Determine Obstacle positions (x = x0 + v*t)
            self.obstacleFutureTrajectories = np.zeros([scenario.nObst,2,scenario.Hp]);
            for k in range(scenario.Hp):
                step = ((k+1)*scenario.dt+scenario.delay_x + scenario.dt + scenario.delay_u)*scenario.obstacles[:,idx.speed]
                self.obstacleFutureTrajectories[:,idx.x,k] = np.squeeze(step*np.cos( scenario.obstacles[:,idx.heading] ) + obstacleState[:,idx.x].reshape(-1,1))
                self.obstacleFutureTrajectories[:,idx.y,k] = np.squeeze(step*np.sin( scenario.obstacles[:,idx.heading] ) + obstacleState[:,idx.y].reshape(-1,1))
        
        self.uMax = uMax
        self.reset = 0

# %%
class MPCclass():
    # Compute all relevant discretization and MPC matrices for all vehicles.
    def __init__(self,scenario,Iter):
        # same as "generate_mpc_matrixces"  
        nx = scenario.model.nx
        nu = scenario.model.nu
        ny = scenario.model.ny
        nVeh = scenario.nVeh
        Hp = scenario.Hp
        Hu = scenario.Hu
        dt = scenario.dt

        # GENARATE MATRICES
        self.A = np.zeros([nx,nx,Hp,nVeh])
        self.B = np.zeros([nx,nu,Hp,nVeh])
        self.E = np.zeros([nx,Hp,nVeh])
        self.Mathcal_A = np.zeros([ny*Hp,nx, nVeh])
        self.Mathcal_B = np.zeros([ny*Hp,nu*Hu,nVeh]) 
        self.Mathcal_C = np.zeros([ny*Hp,1,nVeh])
        self.Phi_0 = np.zeros([nu*Hu,nu*Hu,nVeh]) 
        self.Psi_0 = np.zeros([nu*Hu,1,nVeh]) 
        self.gamma_0 = np.zeros([1,nVeh]) 
        self.const_term = np.zeros([ny*Hp,1,nVeh])
        self.Reference = np.zeros([Hp*ny,nVeh])

        for v in range(nVeh):
            for i in range(Hp):
                self.Reference[slice(ny*(i),ny*(i+1),1), v] = Iter.ReferenceTrajectoryPoints[i,0:ny,v].T

            A,B,C,E = self.discretize( Iter.x0[v,:], Iter.u0[v,:], scenario.Lf[v], scenario.Lr[v], dt, scenario.model )
            E[abs(E)<=1e-30]=0
            self.Mathcal_A[:,:,v], self.Mathcal_C[:,:,v], self.Mathcal_B[:,:,v] = self.prediction_matrices( A,B,C,E, nx,nu,ny,Hp,Hu )  

            self.const_term[:,:,v] = self.Mathcal_A[:,:,v] @ Iter.x0[v,:].reshape(-1,1) + self.Mathcal_C[:,:,v] # Constant term

            self.Phi_0[:,:,v] , self.Psi_0[:,:,v], self.gamma_0[:,v] = self.mpc_cost_function_matrices( scenario.Q[v], scenario.R[v], nu,ny,Hp,Hu, v, scenario.Q_final[v] )

            for i in range(Hp):
                self.A[:,:,i,v] = A
                self.B[:,:,i,v] = B
                self.E[:,i,v] = np.squeeze(E)

    def discretize(self,x0,u0,Lf,Lr, dt, model ):
        
        if x0.shape[0]==1:
            x0=x0.T
        
        Ac,Bc,Cc,Ec = model.comp_jacobian(x0,u0,Lf,Lr)

        tmp = sci.linalg.expm(dt*np.vstack([ np.hstack([Ac,Bc]), np.zeros([Ac.shape[1]+Bc.shape[1]-Ac.shape[0],Ac.shape[1]+Bc.shape[1]])] ))
        Ad = tmp[0:Ac.shape[0],0:Ac.shape[1]]
        Bd = tmp[0:Bc.shape[0],Ac.shape[1]:Ac.shape[1]+Bc.shape[1]]
        Cd = Cc
        
        tmp = sci.linalg.expm(dt* np.vstack([ np.hstack([Ac,Ec]), np.zeros([Ac.shape[1]+Ec.shape[1]-Ac.shape[0], Ac.shape[1]+Ec.shape[1] ]) ]) )
        Ed = tmp[0:Ec.shape[0],Ac.shape[1]:Ac.shape[1]+Ec.shape[1]]
        return Ad,Bd,Cd,Ed


    def mpc_cost_function_matrices(self, Q_weight, R_weight, nu,ny,Hp,Hu,idx_veh,Q_final ):
        # Compute G and H matrices in the quadratic cost function.
        Q = Q_weight * np.eye(ny*Hp)
        for i in range(ny*(Hp-1),ny*Hp):
            Q[i, i] = Q_final
        
        R = R_weight * np.eye(nu*Hu)
        Error = self.Reference[:,idx_veh].reshape(-1,1) - self.const_term[:,:,idx_veh]
        Phi_0 = 0.5*((self.Mathcal_B[:,:,idx_veh].T @ Q @ self.Mathcal_B[:,:,idx_veh] + R)+(self.Mathcal_B[:,:,idx_veh].T @ Q @ self.Mathcal_B[:,:,idx_veh] + R).T)
        Psi_0 = -2*self.Mathcal_B[:,:,idx_veh].T@Q@Error
        gamma_0 = Error.T @ Q @ Error
        return Phi_0, Psi_0, gamma_0

    def prediction_matrices(self, A,B,C,E,nx,nu,ny,Hp, Hu):
        assert (Hu <= Hp)
        Mathcal_B = np.zeros([ny*Hp,nu*Hu]) 
        Mathcal_A = np.zeros([ny*Hp,nx]) 
        Mathcal_C = np.zeros([ny*Hp,nu])

        power_C_A = np.zeros([ny,nx,Hp+1])
        power_C_A[:,:,0] = C@np.eye(nx)
        sum_power_C_A = np.zeros([ny,nx,Hp+1])
        sum_power_C_A[:,:,0] = C@np.eye(nx)
        for i in range(1,Hp+1):
            power_C_A[:,:,i] = C @ np.linalg.matrix_power(A,i) # CA^i
            sum_power_C_A[:,:,i] = power_C_A[:,:,i] + sum_power_C_A[:,:,i-1] # sum_i(CA^i)
        
        for i in range(Hp):
            Mathcal_A[slice(ny*(i),ny*(i+1),1),:] = power_C_A[:,:,i+1]
            Mathcal_C[slice(ny*(i),ny*(i+1),1),:] = sum_power_C_A[:,:,i]@E
            for j in range(i+1):
                Mathcal_B[slice(ny*(i),ny*(i+1),1), slice(nu*(j),nu*(j+1),1)] =  power_C_A[:,:,i-j]@B
        
        return Mathcal_A, Mathcal_C, Mathcal_B

