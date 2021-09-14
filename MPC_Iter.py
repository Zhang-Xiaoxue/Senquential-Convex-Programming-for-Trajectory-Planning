# %%
import numpy as np
from math import floor,cos,sin
import ode
import scipy as sci
from sympy import Matrix

from Scenarios import Indices
from SampleReferTraj import sampleReferenceTrajectory

# %%
class IterClass():
    def __init__(self, scenario, x_measured, u_path, obstacleState, uMax):
        """ 
        arguments: u_path is a list w.r.t nVeh, each indice is a matrix 
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
        self.Theta = np.zeros([ny*Hp,nu*Hu,nVeh])
        self.H = np.zeros([nu*Hu,nu*Hu,nVeh])
        self.g = np.zeros([nu*Hu,1,nVeh])
        self.r = np.zeros([1,nVeh])
        self.freeResponse = np.zeros([ny*Hp,1,nVeh])
        self.Reference = np.zeros([Hp*ny,nVeh])

        for v in range(nVeh):
            for i in range(Hp):
                self.Reference[slice(ny*(i),ny*(i+1),1), v] = Iter.ReferenceTrajectoryPoints[i,0:ny,v].T

            A,B,C,E = self.discretize( Iter.x0[v,:], Iter.u0[v,:], scenario.Lf[v], scenario.Lr[v], dt, scenario.model )
            E[abs(E)<=1e-30]=0
            Psi, Gamma, self.Theta[:,:,v], Pie = self.prediction_matrices( A,B,C,nx,nu,ny,Hp,Hu )  

            self.freeResponse[:,:,v] = Psi @ Iter.x0[v,:].reshape(-1,1) + Gamma * Iter.u0[v,:].T + Pie @ E

            self.H[:,:,v] , self.g[:,:,v], self.r[:,v] = self.mpc_cost_function_matrices( scenario.Q[v], scenario.R[v], nu,ny,Hp,Hu, v, scenario.Q_final[v] )

            # For the simple linearization, the same A,B,E are used in every prediction step.
            for i in range(Hp):
                self.A[:,:,i,v] = A
                self.B[:,:,i,v] = B
                self.E[:,i,v] = np.squeeze(E)

    def discretize(self,x0,u0,Lf,Lr, dt, model ):
        # Compute the linearization and discretization of a non-linear continous model with a known jacobian around a given point.
        # Model form: dx/dt = f(x,u)
        # Discretization form: x(k+1) = Ad*x(k) + Bd*u(k) + Ed
        
        if x0.shape[0]==1:
            x0=x0.T
        
        Ac,Bc,Cc,Ec = model.comp_jacobian(x0,u0,Lf,Lr)

        # Formula from wikipedia https://en.wikipedia.org/wiki/Discretization#cite_ref-1
        # expm([A B; 0 0] * dt) == [Ad Bd; 0 eye]
        # Cited from
        # Raymond DeCarlo: Linear Systems: A State Variable Approach with Numerical Implementation, Prentice Hall, NJ, 1989, page 215
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
        Error = self.Reference[:,idx_veh].reshape(-1,1) - self.freeResponse[:,:,idx_veh]
        H = 0.5*((self.Theta[:,:,idx_veh].T @ Q @ self.Theta[:,:,idx_veh] + R)+(self.Theta[:,:,idx_veh].T @ Q @ self.Theta[:,:,idx_veh] + R).T)
        g = -2*self.Theta[:,:,idx_veh].T@Q@Error
        r = Error.T @ Q @ Error
        return H, g, r

    def prediction_matrices(self, A,B,C,nx,nu,ny,Hp, Hu):
        assert (Hu <= Hp)
        Theta = np.zeros([ny*Hp,nu*Hu])
        Psi = np.zeros([ny*Hp,nx])
        Gamma = np.zeros([ny*Hp,nu])
        Pie = np.zeros([ny*Hp,nx])

        powersCA = np.zeros([ny,nx,Hp+1])
        powersCA[:,:,0] = C@np.eye(nx)
        summedPowersCA = np.zeros([ny,nx,Hp+1])
        summedPowersCA[:,:,0] = C@np.eye(nx)
        for i in range(1,Hp+1):
            powersCA[:,:,i] = C @ np.linalg.matrix_power(A,i)
            summedPowersCA[:,:,i] = powersCA[:,:,i] + summedPowersCA[:,:,i-1]
        
        for i in range(Hp):
            Psi[slice(ny*(i),ny*(i+1),1),:] = powersCA[:,:,i+1]
            Gamma[slice(ny*(i),ny*(i+1),1),:] = summedPowersCA[:,:,i]@B
            for iu in range(min(i,Hu)+1):
                Theta[slice(ny*(i),ny*(i+1),1), slice(nu*(iu),nu*(iu+1),1)] =  summedPowersCA[:,:,i-iu]@B
            Pie[slice(ny*(i),ny*(i+1),1),:] = summedPowersCA[:,:,i]
        
        return Psi, Gamma, Theta, Pie

