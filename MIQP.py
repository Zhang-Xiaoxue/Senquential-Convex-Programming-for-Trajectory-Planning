# %%
import numpy as np
from math import floor,cos,sin, sqrt, pi
import scipy as sci
from qpsolvers import solve_qp
from sympy import Matrix
import time
from scipy import io
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB

from Scenarios import Indices
from Config import Config, MIP_CPLEX
from MPC_Iter import MPCclass

cfg = Config()
mip_setting = MIP_CPLEX()

# %%
class MIQPcontroller():
    def __init__(self, scenario, Iter, prevOutput):

        self.dsafeExtra = 2

        self.scenario = scenario
        self.Iter = Iter
        self.prevOutput = prevOutput

        self.nx = scenario.model.nx
        self.nu = scenario.model.nu
        self.ny = scenario.model.ny
        self.Hp = scenario.Hp
        self.Hu = scenario.Hu
        self.nVeh = scenario.nVeh
        self.nObst = scenario.nObst
        self.dsafeExtra = scenario.dsafeExtra

        self.scenario_duLim = scenario.duLim

        self.mpc = MPCclass(scenario, Iter)
        self.qcqp = self.convert_to_QCQP(scenario)
        self.du = np.zeros([self.nVeh*self.Hu,1])
    
    def MIQP_controller(self, Iter):
        MIP, bObstAvoidStart, NOV = self.convert_to_MIP(Iter)

        # Solving the Optimization Problem
        controllerOutput = {}
        optimizerTimer = time.time()
        x_continuous = cp.Variable((bObstAvoidStart,1))
        x_binary = cp.Variable((NOV-bObstAvoidStart,1), boolean=True)
        x = cp.vstack((x_continuous,x_binary))
        obj = cp.Minimize(0.5*cp.quad_form(x,MIP['H_MIQP'])+MIP['f_MIQP'].T@x)
        
        constr  = [ MIP['Aeq']@x == MIP['Beq']]
        constr += [ MIP['Aineq']@x <= MIP['Bineq'] ]
        constr += [ x <= MIP['ub'] ]
        constr += [ x >= MIP['lb'] ]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.GUROBI)
        x_sol = x.value

        # [x,fval,exitflag,output] = cplexmiqp (MIP['H_MIQP'], MIP['f_MIQP'], MIP['Aineq'], MIP['Bineq'], MIP['Aeq'], MIP['Beq'],[ ], [ ], [ ], MIP['lb'], MIP['ub'], MIP['ctype'], [ ], CPLEXoptions)
        controllerOutput['optimizerTime'] = time.time() - optimizerTimer
        controllerOutput['CPLEX_fval'] = prob.value

        # Extracting the data that will be returned from the optimized variables array
        min_avoid_constr = []
        if x_sol is not None:
            min_avoid_constr = min(MIP['Bineq'][MIP['avoidanceContraintsStart']:MIP['Bineq'].shape[0]] - MIP['Aineq'][MIP['avoidanceContraintsStart']:MIP['Aineq'].shape[0],:]@x_sol)

        print('miqp status: fval {} | min_avoid_constr {}\n'.format(prob.value+MIP['r_MIQP'], min_avoid_constr) )
        
        # when exitflag > 0, then x is a solution
        if x_sol is not None :
            # extract results
            U = np.zeros((self.Hp,self.nVeh))
            trajectoryPrediction = np.zeros((self.Hp,self.ny,self.nVeh))
            for v in range(self.nVeh):
                for k in range(self.Hp):
                    trajectoryPrediction[k,:,v] = np.squeeze(x_sol[MIP['varIdx']['y'](v,k)])
                    U[k,v] = x_sol[MIP['varIdx']['u'](v,min(k,self.Hu))]     
        else:
            trajectoryPrediction,U = self.decode_deltaU(np.zeros((self.Hu*self.nVeh,1)))
            U = np.squeeze(U[:,0,:])

        return U,trajectoryPrediction,controllerOutput
    
    def convert_to_MIP(self, Iter):
        MIP = {}
        bigM = mip_setting.bigM
        polyDegree = mip_setting.polygonalNormApproximationDegree

        idx = Indices()

        state0   = Iter.x0
        ctrl0    = Iter.u0
        refPoint = Iter.ReferenceTrajectoryPoints

        dt = self.scenario.dt
        RVeh = self.scenario.RVeh

        W = np.vstack([np.sin(np.array(range(1,polyDegree+1))*2*pi/polyDegree),np.cos(np.array(range(1,polyDegree+1))*2*pi/polyDegree)]).T

        ## Variable Index Mapping
        statesStart = 0
        NOStates = self.Hp*self.nx*self.nVeh
        ctrlStart = NOStates
        NOCtrl = self.Hu*self.nu*self.nVeh
        deltaCtrlStart =  NOStates + NOCtrl
        NODeltaCtrl = self.Hu*self.nu*self.nVeh
        refDistStart =  NOStates + NOCtrl + NODeltaCtrl
        NORefDist = self.Hp*self.nVeh
        bObstAvoidStart = NOStates + NOCtrl + NODeltaCtrl + NORefDist
        bObstAvoid = self.Hp*self.ny*self.nObst*self.nVeh
        bVehAvoidStart = NOStates + NOCtrl + NODeltaCtrl + NORefDist + bObstAvoid
        bVehAvoid = 0
        if self.nVeh > 1:
            bVehAvoid = self.Hp*self.ny*self.nVeh*self.nVeh
        NOV = NOStates + NOCtrl + NODeltaCtrl + NORefDist+ bObstAvoid + bVehAvoid

        # Index maps to calculate the position of each variable in the optimization problem.
        varIdx = {}
        varIdx['x'] = lambda v,k : slice(statesStart+(self.nx*v*self.Hp)+self.nx*k, statesStart+(self.nx*v*self.Hp)+self.nx*k+self.nx, 1)
        varIdx['y'] = lambda v,k : slice(statesStart+(self.nx*v*self.Hp)+self.nx*k, statesStart+(self.nx*v*self.Hp)+self.nx*k+2, 1)
        varIdx['u'] = lambda v,k : ctrlStart + self.Hu*v + k+1
        varIdx['refDist'] = lambda v,k : refDistStart + v*self.Hp + k+1
        varIdx['deltaCtrl'] = lambda v,k : deltaCtrlStart + v*self.Hu + k+1
        varIdx['bObstAvoid'] = lambda v,o,k : slice(bObstAvoidStart+2*(self.Hp*self.nObst*v+self.Hp*o+k), bObstAvoidStart+2*(self.Hp*self.nObst*v+self.Hp*o+k)+2, 1)
        varIdx['bVehAvoid'] = lambda vi,nj,k : slice(bVehAvoidStart+2*self.Hp*self.nVeh*vi+2*self.Hp*vj+2*k, bVehAvoidStart+2*self.Hp*self.nVeh*vi+2*self.Hp*vj+2*k+2, 1)

        # Objective
        f_MILP = np.zeros((NOV,1))
        f_MIQP = np.zeros((NOV,1))
        H_MIQP = np.zeros((NOV,NOV))
        r_MIQP = 0
        for v in range(self.nVeh):
            
            cooperationCoeff = 1  
            if hasattr(self.scenario,'CooperationCoefficients'):
                assert(self.scenario.CooperationCoefficients.shape[0] == 1)
                assert(self.scenario.CooperationCoefficients.shape[1] == self.nVeh)
                cooperationCoeff = self.scenario.CooperationCoefficients[v]
            
            for k in range(self.Hp-1):
                f_MILP[varIdx['refDist'](v,k)] = cooperationCoeff * self.scenario.Q[v]
                f_MIQP[varIdx['y'](v,k)] = -cooperationCoeff * 2*self.scenario.Q[v]*refPoint[k,:,v].reshape(-1,1)
                H_MIQP[varIdx['y'](v,k),varIdx['y'](v,k)] = 2 * cooperationCoeff * np.eye(2) * self.scenario.Q[v]
                r_MIQP = r_MIQP + cooperationCoeff * self.scenario.Q[v] * np.linalg.norm(refPoint[k,:,v])**2

            f_MILP[varIdx['refDist'](v,self.Hp)] = cooperationCoeff * self.scenario.Q_final[v]
            f_MIQP[varIdx['y'](v,self.Hp)] = -cooperationCoeff * 2*self.scenario.Q_final[v]*refPoint[self.Hp-1,:,v].reshape(-1,1)
            H_MIQP[varIdx['y'](v,self.Hp),varIdx['y'](v,self.Hp)] = cooperationCoeff * np.eye(2) * 2 * self.scenario.Q_final[v]
            r_MIQP = r_MIQP + cooperationCoeff * self.scenario.Q_final[v] * np.linalg.norm(refPoint[self.Hp-1,:,v])**2
            for k in range(self.Hu):
                f_MILP[varIdx['deltaCtrl'](v,k)] = cooperationCoeff * mip_setting.R_Gain * self.scenario.R[v]
                H_MIQP[varIdx['deltaCtrl'](v,k),varIdx['deltaCtrl'](v,k)] = cooperationCoeff * 2 * self.scenario.R[v]
        
        ###################### The equality constraints ###############################
        Aeq = np.zeros ((self.nx*self.Hp*self.nVeh,NOV))
        Beq = np.zeros ((self.nx*self.Hp*self.nVeh,1))
        rows = 0  # Equation/Constraint counter
        # Predictions of the system states
        for v in range(self.nVeh):
            rows_slice = slice(rows,rows+self.nx,1)
            Aeq[rows_slice,  varIdx['x'](v,0)] =  np.eye(self.nx)
            Aeq[rows_slice,  varIdx['u'](v,0)] = np.squeeze(-self.mpc.B[:,:,0,v])
            Beq[rows_slice,  0]                =  self.mpc.A[:,:,0,v]@state0[v,:].T + self.mpc.E[:,0,v]
            rows += self.nx
            
            for k in range(self.Hp-1):
                rows_slice = slice(rows,rows+self.nx,1)
                Aeq[rows_slice, varIdx['x'](v,k+1)]         =  np.eye(self.nx)
                Aeq[rows_slice, varIdx['x'](v,k)]           = -self.mpc.A[:,:,k+1,v]
                Aeq[rows_slice, varIdx['u'](v,min(self.Hu,k+1))] = -np.squeeze(self.mpc.B[:,:,k+1,v])
                Beq[rows_slice, 0]                       =  self.mpc.E[:,k+1,v]
                rows += self.nx
        assert (rows == self.nx*self.Hp*self.nVeh)

        ######################### The inequality Constraints ##############################
        ## Specifying the Number of the required Constraints
        nConstraints = self.nVeh*(
            self.Hp*polyDegree              # Trajectory deviation slack variable
            + 2*self.Hu                     # Delta U slack variable
            + 4*self.nObst*self.Hp          # Obstacle avoidance
            + 2*(self.nVeh-1)*self.Hp)      # Vehicle avoidance

        ## Inequality Constraints Matrices
        Aineq = np.zeros((nConstraints,NOV))
        Bineq = np.zeros((nConstraints,1))
        rows = 0 # Equation/Constraint counter

        ## Distance of vehicles positions to Reference Trajectory Constraints
        for v in range(self.nVeh):
            for k in range(self.Hp):
                rows_slice = slice(rows,rows+polyDegree,1)
                Aineq[rows_slice, varIdx['refDist'](v,k)] = -1
                Aineq[rows_slice, varIdx['y'](v,k)]       = W
                Bineq[rows_slice, 0]                      = W@refPoint[k,:,v].T
                rows += polyDegree

        ## Defining the change of Control variables
        for v in range(self.nVeh):
            rows_slice = slice(rows,rows+2)
            Aineq[rows_slice, varIdx['u'](v,1)]          = np.array([1,-1])
            Aineq[rows_slice, varIdx['deltaCtrl'](v,1)]  = np.array([-1,-1])
            Bineq[rows_slice, 0]                         = ctrl0[v,0] * np.array([1,-1])
            rows += 2
            
            for k in range(1, self.Hu):
                rows_slice = slice(rows,rows+2)
                Aineq[rows_slice, varIdx['u'](v,k)]         = np.array([ 1,-1])
                Aineq[rows_slice, varIdx['deltaCtrl'](v,k)] = np.array([-1,-1])
                Aineq[rows_slice, varIdx['u'](v,k-1)]       = np.array([-1, 1])
                rows += 2

        MIP['avoidanceContraintsStart'] = rows

        ## Obstacle Avoidance Constraints
        for v in range(self.nVeh):
            for o in range(self.nObst):
                # - Always the l and w here are expressing the half length and half width
                if (mip_setting.obstAsQCQP): # consider obst. as in QCQP
                    c, s = 1, 0
                    l = self.scenario.dsafeObstacles[v,o]
                    w = self.scenario.dsafeObstacles[v,o]
                else: # consider obst. as rectangle
                    # - Augmenting the obstacle dimensions for enough safe obstacle avoidance
                    l = self.scenario.obstacles[o,idx.length]/2 + RVeh[v]  # Squaring to make sure that the l is positive as
                    w = self.scenario.obstacles[o,idx.width]/2 + RVeh[v]  # it may be negative dependingon the orientation
                    # - this term for considering the sampling time effect consodering the velocity of the vehicles and the obstacle
                    l_cord = (state0[v,3] + self.scenario.obstacles[o][3,0])*dt
                    # - Including the cord length in the obstaccles dimensions
                    l = l + l_cord*cos(pi/4)/2
                    w = w + l_cord*cos(pi/4)/2
                    # - Checking if the calculated dimensions are enough or the obstacles was originally small
                    if (l < l_cord/2):
                        l = l_cord/2
                    # l = l + cfg.MILP_CPLEX.dsafeObstacle;
                    if (w < l_cord/2):
                        w = l_cord/2
                    # w = w + cfg.MILP_CPLEX.dsafeObstacle;
                    c = cos(self.scenario.obstacles[o][2,0])
                    s = sin(self.scenario.obstacles[o][2,0])

                for k in range(self.Hp):
                    obst_x = Iter.obstacleFutureTrajectories[o,idx.x,k]
                    obst_y = Iter.obstacleFutureTrajectories[o,idx.y,k]            
                    rows_slice = slice(rows, rows+4, 1)
                    Aineq[rows_slice, varIdx['y'](v,k)] = np.array([[-c,-s], [c,s], [-s,c], [s,-c]])
                    Aineq[rows_slice, varIdx['bObstAvoid'](v,o,k)] = bigM*np.array([[-1,-1], [1,-1], [-1,1], [1,1]])
                    Bineq[rows_slice, 0] = -np.array([l,l,w,w]) + obst_x*np.array([-c,c,-s,s]) + obst_y*np.array([-s,s,c,-c]) + bigM*np.array([0,1,1,2])
                    rows += 4
                
        ## Vehicle Avoidance Constraints
        if (self.nVeh > 1):
            for vi in range(self.nVeh):
                for vj in range(self.nVeh):
                    avoidanceDist = self.scenario.dsafeVehicles[vi,vj]
                    for k in range(self.Hp):
                        if vi<vj:
                            rows_slice = slice(rows, rows+4, 1)
                            Aineq[rows_slice, varIdx['y'](vi,k)] = np.array([[ 1,0],[0, 1],[-1,0],[0,-1]])
                            Aineq[rows_slice, varIdx['y'](vj,k)] = np.array([[-1,0],[0,-1],[ 1,0],[0, 1]])
                            Aineq[rows_slice, varIdx['bVehAvoid'](vi,vj,k)] = bigM*np.array([[-1,-1],[1,-1],[-1,1],[1,1]])
                            Bineq[rows_slice, 0] = bigM*np.array([0,1,1,2]) - avoidanceDist*np.ones(4)
                            rows += 4

        assert (rows == nConstraints)

        ## Lower and Upper Bounds
        # Setting the Lower bounds and Upper Bounds of the control variables and the change of the control variables
        lb = np.full((NOV,1),-np.inf)
        ub = np.full((NOV,1), np.inf)

        for v in range(self.nVeh):
            for k in range(self.Hu):
                lb[varIdx['u'](v,k)] = -Iter.uMax[0,v]
                ub[varIdx['u'](v,k)] =  Iter.uMax[0,v]
                ub[varIdx['deltaCtrl'](v,k)] = self.scenario.duLim

        ## Specifying the type of each variable in the X 'variables array'.
        ctype = np.empty(NOV)
        ctype[0:bObstAvoidStart] = 0
        ctype[bObstAvoidStart:NOV] = 1
        MIP['f_MILP'] = f_MILP
        MIP['f_MIQP'] = f_MIQP
        MIP['H_MIQP'] = H_MIQP
        MIP['r_MIQP'] = r_MIQP
        MIP['Aineq'] = Aineq
        MIP['Bineq'] = Bineq
        MIP['Aeq'] = Aeq
        MIP['Beq'] = Beq
        MIP['lb'] = lb
        MIP['ub'] = ub
        MIP['ctype'] = ctype
        MIP['varIdx'] = varIdx
        
        return MIP, bObstAvoidStart, NOV

    def decode_deltaU(self, du):
        U = np.zeros([self.Hp,self.nu,self.nVeh])
        Traj = np.zeros([self.Hp,self.ny,self.nVeh])
        du = du.reshape([self.nu,self.Hu,self.nVeh],order='F')

        # Control values
        for v in range(self.nVeh):
            U[0,:,v] = du[:,0,v].T + self.Iter.u0[v,:]
            for k in range(1,self.Hu):
                U[k,:,v] = du[:,k,v].T + U[k-1,:,v]
            for k in range(self.Hu,self.Hp):
                U[k,:,v] = U[self.Hu-1,:,v]

        # Predicted trajectory
        for v in range(self.nVeh):
            X = self.mpc.freeResponse[:,:,v] + self.mpc.Theta[:,:,v] @ du[:,:,v].T
            X = X.reshape([self.ny,self.Hp],order='F')
            for i in range(self.ny):
                Traj[:,i,v] = X[i,:]
        
        return Traj, U
    
    def convert_to_QCQP(self, scenario):
        # RESULT FORM:
        #  x = [ du1; du2;...;dunVeh]
        #  minimize    x'*p0*x + q0'*x + r0
        #  subject to  x'*pi*x + qi'*x + ri <= 0,   i = 1,...,m
        
        # result matrices
        p0 = np.zeros([self.nVeh*(self.nu*self.Hu),self.nVeh*(self.nu*self.Hu)])
        q0 = np.zeros([self.nVeh*(self.nu*self.Hu),1])
        r0 = 0

        p = np.zeros([self.nVeh-1, self.nVeh, self.Hp, self.nVeh*(self.nu*self.Hu), self.nVeh*(self.nu*self.Hu)])
        q = np.zeros([self.nVeh-1, self.nVeh, self.Hp, self.nVeh*(self.nu*self.Hu), 1 ])
        r = np.zeros([self.nVeh-1, self.nVeh, self.Hp])

        p_obst = np.zeros([self.nVeh, self.nObst, self.Hp, self.nVeh*(self.nu*self.Hu), self.nVeh*(self.nu*self.Hu)])
        q_obst = np.zeros([self.nVeh, self.nObst, self.Hp, self.nVeh*(self.nu*self.Hu), 1])
        r_obst = np.zeros([self.nVeh, self.nObst, self.Hp])

        for v in range(self.nVeh):
            # OBJECTIVE FUNCTION MATRICES    
            veh1_slice = slice(self.nu*self.Hu*v, self.nu*self.Hu*(v+1),1)
            cooperationCoeff = 1
            if hasattr(scenario,'CooperationCoefficients'):
                assert (scenario.CooperationCoefficients.shape[0] == 1) 
                assert (scenario.CooperationCoefficients.shape[1] == self.nVeh)
                cooperationCoeff = scenario.CooperationCoefficients[v,0]
            p0[veh1_slice,veh1_slice] = cooperationCoeff * self.mpc.H[:,:,v]
            q0[veh1_slice,0] = np.squeeze(cooperationCoeff*self.mpc.g[:,:,v])
            r0 = r0 + cooperationCoeff*self.mpc.r[:,v]

            # CONSTRAINTS MATRICES
            for k in range(self.Hp):
                # VEHICLES AVOIDANCE
                intv = slice(k*self.ny, (k+1)*self.ny, 1)
                for v2 in range(v+1, self.nVeh):

                    veh2_slice = slice(v2*self.nu*self.Hu, (v2+1)*self.nu*self.Hu, 1)
                                
                    p[v,v2,k,veh1_slice,veh1_slice]  = - self.mpc.Theta[intv,:, v].T @ self.mpc.Theta[intv,:, v]
                    p[v,v2,k,veh2_slice,veh2_slice]  = - self.mpc.Theta[intv,:,v2].T @ self.mpc.Theta[intv,:,v2]
                    p[v,v2,k,veh1_slice,veh2_slice]  =   self.mpc.Theta[intv,:, v].T @ self.mpc.Theta[intv,:,v2]
                    p[v,v2,k,veh2_slice,veh1_slice]  =   self.mpc.Theta[intv,:,v2].T @ self.mpc.Theta[intv,:, v]
                    
                    b =  self.mpc.freeResponse[intv,0,v] - self.mpc.freeResponse[intv,0,v2]

                    q[v,v2,k,veh1_slice,0] = -2*self.mpc.Theta[intv,:, v].T @ b
                    q[v,v2,k,veh2_slice,0] =  2*self.mpc.Theta[intv,:,v2].T @ b
                    r[v,v2,k] = (scenario.dsafeVehicles[v,v2] + self.dsafeExtra)**2 - b.T @ b
                
                # OBSTACLE AVOIDANCE
                # Obstacle in next time step
                if self.nObst:
                    for o in range(self.nObst): 
                        p_obst[v,o,k,veh1_slice,veh1_slice] = - self.mpc.Theta[intv,:,v].T @ self.mpc.Theta[intv,:,v]
                        b = self.mpc.freeResponse[intv,0,v] - self.Iter.obstacleFutureTrajectories[o,:,k].T
                        q_obst[v,o,k,veh1_slice,0] = -2*self.mpc.Theta[intv,:,v].T @ b
                        r_obst[v,o,k] = (scenario.dsafeObstacles[v,o] + self.dsafeExtra)**2 - b.T @ b

        for v in range(self.nVeh):
            for k in range(self.Hp):
                for v2 in range(v+1, self.nVeh):
                    p[v,v2,k,:,:] = 0.5*(p[v,v2,k,:,:]+p[v,v2,k,:,:].T)
                if self.nObst:
                    for o in range(self.nObst):
                        p_obst[v,o,k,:,:] = 0.5*(p_obst[v,o,k,:,:]+p_obst[v,o,k,:,:].T)

        p[abs(p)<=1e-30] = 0
        q[abs(q)<=1e-30] = 0
        
        qcqp_dict = {'p0':p0, 'q0':q0, 'r0':r0, 'p':p, 'q':q, 'r':r, 'p_o':p_obst, 'q_o':q_obst, 'r_o':r_obst}
            
        return qcqp_dict
    
    def QCQP_evaluate(self, deltaU):
        c_linear = 0
        c_quad = 1e9
        objectiveTradeoffCoefficient = 1
        sum_violations = 0
        max_violation = 0

        feasible = True

        constraintValuesVehicle = np.full([self.nVeh,self.nVeh,self.Hp], -np.inf)
        constraintValuesObstacle = np.full([self.nVeh,self.nObst,self.Hp], -np.inf)
        
        objValue = deltaU.T @ self.qcqp['p0'] @ deltaU + self.qcqp['q0'].T @ deltaU + self.qcqp['r0']
        feasibilityScore = objectiveTradeoffCoefficient*objValue
        feasibilityScoreGradient = objectiveTradeoffCoefficient*((self.qcqp['p0']+self.qcqp['p0'].T) @ deltaU + self.qcqp['q0'])

        for v in range(self.nVeh):
            for k in range(self.Hp):
                # VEHICLES
                for v2 in range((v+1),self.nVeh):
                    ci = deltaU.T @ self.qcqp['p'][v,v2,k,:,:] @ deltaU + self.qcqp['q'][v,v2,k,:,:].T @ deltaU + self.qcqp['r'][v,v2,k]
                    constraintValuesVehicle[v,v2,k] = ci
                    constraintValuesVehicle[v2,v,k] = ci
                    feasibilityScore = feasibilityScore + c_quad*max(ci,0)**2 + c_linear*max(ci,0)
                    if (ci > 0):
                        feasibilityScoreGradient = feasibilityScoreGradient + \
                            (c_quad*2*ci+c_linear)*((self.qcqp['p'][v,v2,k,:,:]+self.qcqp['p'][v,v2,k,:,:].T)@deltaU
                             + self.qcqp['q'][v,v2,k,:,:])

                    if (ci > cfg.QCQP.constraintTolerance):
                        feasible = False
                        sum_violations = sum_violations + ci
                        max_violation = max(max_violation,ci)     
                    
                    # OBSTACLES
                    if self.nObst:
                        for ob in range(self.nObst):
                            ci = deltaU.T @ self.qcqp['p_o'][v,ob,k,:,:] @ deltaU + self.qcqp['q_o'][v,ob,k,:,:].T @ deltaU + self.qcqp['r_o'][v,ob,k]
                            constraintValuesObstacle[v,ob,k] = ci
                            feasibilityScore = feasibilityScore + c_quad*max(ci,0)**2 + c_linear*max(ci,0)
                            if (ci > 0):
                                feasibilityScoreGradient = feasibilityScoreGradient + \
                                    (c_quad*2*ci+c_linear)*((self.qcqp['p_o'][v,ob,k,:,:]+self.qcqp['p_o'][v,ob,k,:,:].T)@deltaU 
                                    + self.qcqp['q_o'][v,ob,k])

                            if (ci > cfg.QCQP.constraintTolerance):
                                feasible = False
                                sum_violations = sum_violations + ci
                                max_violation = max(max_violation,ci)

        return feasible, objValue, feasibilityScore, feasibilityScoreGradient, max_violation, sum_violations, constraintValuesVehicle, constraintValuesObstacle
    
    def evaluateInOriginalProblem(self, controlPrediction, trajectoryPrediction,options):
        evaluation = {}
        evaluation['predictionObjectiveValueX'] = 0
        evaluation['predictionObjectiveValueDU'] = 0
        
        # trajectory Prediction  error term
        sqRefErr = (self.Iter.ReferenceTrajectoryPoints-trajectoryPrediction)**2
        for v in range(self.nVeh):
            evaluation['predictionObjectiveValueX'] = evaluation['predictionObjectiveValueX'] + \
                self.scenario.Q[v] * sqRefErr[0:-1,:,v].sum() + \
                self.scenario.Q_final[v] * sqRefErr[-1,:,v].sum()
        
        # steering Prediction term
        du = np.diff(np.vstack([self.Iter.u0.T, controlPrediction]), axis=0)
        du = du[0:self.Hu,:]
        sqDeltaU = du**2
        for v in range(self.nVeh):
            evaluation['predictionObjectiveValueDU'] = evaluation['predictionObjectiveValueDU'] + \
                self.scenario.R[v] * sqDeltaU[:,v].sum()
        
        evaluation['predictionObjectiveValue'] = evaluation['predictionObjectiveValueX'] + evaluation['predictionObjectiveValueDU']    
        
        # crash prediction check based on QCQP
        du = du.reshape(du.shape[0]*du.shape[1],1,order='F')
        evaluation['predictionFeasibleQCQP'],_,_,_,_,_,evaluation['constraintValuesVehicleQCQP'],evaluation['constraintValuesObstacleQCQP'] = self.QCQP_evaluate(du)
        
        evaluation['constraintValuesVehicle_trajPred'] = np.zeros([self.nVeh,self.nVeh,self.Hp])
        if self.nObst:
            evaluation['constraintValuesObstacle_trajPred'] = np.zeros([self.nVeh,self.nObst,self.Hp])
        
        # crash prediction check based on the predicted trajectory
        evaluation['predictionFeasible_trajPred'] = True
        for k in range(self.Hp):
            for v in range(self.nVeh):
                for v2 in range((v+1),self.nVeh):
                    veh_dist_sq = ((trajectoryPrediction[k,:,v]-trajectoryPrediction[k,:,v2])**2).sum()
                    ci = self.scenario.dsafeVehicles[v,v2]**2 - veh_dist_sq
                    evaluation['constraintValuesVehicle_trajPred'][v,v2,k] = ci
                    evaluation['constraintValuesVehicle_trajPred'][v2,v,k] = ci
                    if (ci > cfg.QCQP.constraintTolerance):
                        evaluation['predictionFeasible_trajPred'] = False
                if self.nObst:
                    for o in range(self.nObst):
                        obst_dist_sq = ((trajectoryPrediction[k,:,v]-self.Iter.obstacleFutureTrajectories[o,:,k])**2).sum()
                        ci = self.scenario.dsafeObstacles[v,o]**2 - obst_dist_sq
                        evaluation['constraintValuesObstacle_trajPred'][v,o,k] = ci
                        if (ci > cfg.QCQP.constraintTolerance):
                            evaluation['predictionFeasible_trajPred'] = False
        
        if not hasattr(options, 'ignoreQCQPcheck'):
            if (evaluation['predictionFeasibleQCQP'] != evaluation['predictionFeasible_trajPred']):
                print('feasibility criteria disagree\n')
        
        evaluation['predictionFeasible'] = evaluation['predictionFeasible_trajPred']
        evaluation['constraintValuesVehicle'] = evaluation['constraintValuesVehicle_trajPred']
        if self.nObst:
            evaluation['constraintValuesObstacle'] = evaluation['constraintValuesObstacle_trajPred']
    
        return evaluation