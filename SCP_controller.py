# %%
import numpy as np
from math import floor,cos,sin, sqrt
import scipy as sci
from qpsolvers import solve_qp
from sympy import Matrix
import time
from scipy import io
import cvxpy as cp

from Scenarios import Indices
from Config import Config
from MPC_Iter import MPCclass

cfg = Config()

# %%
class SCPcontroller():
    def __init__(self, scenario, Iter, prevOutput):

        self.dsafeExtra = 2

        self.scenario = scenario
        self.Iter = Iter
        self.prevOutput = prevOutput

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
    
    def SCP_controller(self, Iter):
        init_method = 'previous'
        # if hasattr(self.scenario, 'SCP_init_method'):
        #     init_method = self.scenario.SCP_init_method
        
        if (init_method == 'previous') and self.prevOutput and ('du' in self.prevOutput):
            self.du = self.prevOutput['du']
            self.du = self.du.reshape([self.Hu,self.nVeh],order='F')
            self.du[0:-1,:] = self.du[1:self.Hu,:]
            self.du[-1,:] = 0
            self.du = self.du.reshape([self.Hu*self.nVeh,1],order='F')
        
        controllerOutput = {}
        controllerOutput['resultInvalid'] = False
        optimizerTimer = time.time()
        
        # try last MPC sol.
        self.du, feasible , _, controllerOutput['optimization_log'] = self.SCP_optimizer( self.du )
        # heuristic: try left and right if no. of veh. is 1
        if self.nVeh == 1: 
            if feasible == False:
                # try left
                self.du = np.ones([self.du.shape[0],self.du.shape[0]])*self.scenario_duLim
                du_pos , feasible_pos, _ = self.SCP_optimizer( self.du )
                if feasible_pos :
                    self.du = du_pos
                else:
                    # try right
                    self.du = -np.ones([self.du.shape[0],self.du.shape[0]])*self.scenario_duLim
                    du_neg , feasible_neg, _ = self.SCP_optimizer( self.du )
                    if feasible_neg:
                        self.du = du_neg
                    else:
                        print('INFEASIBLE PROBLEM')
                        controllerOutput['resultInvalid'] = True
        
        controllerOutput['du'] = self.du
        trajectoryPrediction, U = self.decode_deltaU(self.du)  # U: (14,1,8)
        U = np.squeeze(U[:,0,:])
        controllerOutput['optimizerTime'] = time.time() - optimizerTimer
        return U, trajectoryPrediction, controllerOutput
    
    def SCP_optimizer(self, x_0):
        if abs(x_0[0,0]) < np.spacing(1): # avoid numerical issues
            x_0[0] = np.spacing(1)

        n_du = x_0.shape[0]
        nVars = x_0.shape[0]+1
        nCons = int(self.nVeh * ( self.Hp*(self.nVeh-1)/2 + self.Hp*self.nObst + 2*self.Hu ))

        _, objValue_0, _, _, max_violation_0, _, _, _ = self.QCQP_evaluate(x_0)
        delta_tol = 1e-4
        slack_weight = 1e5
        slack_ub = 1e30
        slack_lb = 0
        max_SCP_iter = 20

        optimization_log = {'P':[], 'q':[], 'Aineq':[], 'bineq':[], 'lb':[], 'ub':[], 'x':[], 'slack':[],
                            'SCP_ObjVal':[], 'QCQP_ObjVal':[], 'delta_hat':[], 'delta':[], 'du':[], 'feasible':[],
                            'prev_du':[], 'Traj':[], 'U':[], 'prevTraj':[], 'prevU':[]}

        for i in range(max_SCP_iter):
            Aineq = np.zeros([nCons,nVars])
            bineq = np.zeros([nCons,1])
            row = 0
            # VEHICLE AVOIDANCE
            for v in range(self.nVeh-1):
                for v2 in range(v+1,self.nVeh):
                    for k in range(self.Hp):
                        Aineq[row,0:n_du] = (self.qcqp['q'][v,v2,k].T + 2*x_0.T @ self.qcqp['p'][v,v2,k])
                        bineq[row,0] = -(self.qcqp['r'][v,v2,k]-x_0.T@self.qcqp['p'][v,v2,k]@x_0)
                        row += 1

            # OBSTACLE AVOIDANCE
            if self.nObst:
                for v in range(self.nVeh):
                    for o in range(self.nObst):
                        for k in range(self.Hp):
                            Aineq[row,slice(0,n_du,1)] = (self.qcqp['q_o'][v,o,k].T + 2*x_0.T @ self.qcqp['p_o'][v,o,k])
                            bineq[row,0] = -(self.qcqp['r_o'][v,o,k]-x_0.T @ self.qcqp['p_o'][v,o,k] @ x_0)
                            row += 1
            
            # U limits
            for v in range(self.nVeh):
                row_slice = slice(row, row+self.Hu,1)
                Aineq[row_slice,slice(v*(self.nu*self.Hu),(v+1)*(self.nu*self.Hu),1)] = np.tril(np.ones([self.Hu,self.Hu]))
                bineq[row_slice,0] = -self.Iter.u0[v,:] + self.Iter.uMax[:,v]
                
                row_slice1 = slice(row+self.Hu, row+self.Hu+self.Hu,1)
                Aineq[row_slice1,slice(v*(self.nu*self.Hu),(v+1)*(self.nu*self.Hu),1)] =  -(np.tril(np.ones([self.Hu,self.Hu])))
                bineq[row_slice1,0] = -(-self.Iter.u0[v,:] - self.Iter.uMax[:,v])
                row = row+self.Hu+self.Hu
            
            lb = -np.ones([n_du,1])*self.scenario_duLim
            ub =  np.ones([n_du,1])*self.scenario_duLim
            P = 2*self.qcqp['p0']
            
            # slack var
            q = np.vstack([self.qcqp['q0'], slack_weight])
            P = sci.linalg.block_diag(P,0)
            Aineq[0:nCons-2*self.Hu*self.nVeh,-1] = -1 # <--  enable slack var just for collision avoidance constraints
            lb = np.vstack([lb, slack_lb])
            ub = np.vstack([ub, slack_ub])
            Aineq[abs(Aineq)<=1e-20] = 0

            # res = {'P_py': P, 'q_py':q, 'Aineq_py':Aineq, 'bineq_py':bineq, 'lb_py':lb, 'ub_py':ub, \
            #        'qcqp_p':self.qcqp['p'], 'qcqp_q':self.qcqp['q'], 'qcqp_r':self.qcqp['r'], \
            #        'qcqp_q0':self.qcqp['q0'], 'qcqp_p0':self.qcqp['p0'], 'qcqp_r0':self.qcqp['r0']}
            # io.savemat('res_py.mat', res)

            x = cp.Variable([P.shape[0],1])
            cost = 0.5*cp.quad_form(x,P) + q.T@x
            constr = [Aineq@x <= bineq]
            constr += [x <= ub]
            constr += [x >= lb]
            prob = cp.Problem(cp.Minimize(cost), constr)
            res = prob.solve(solver=cp.GUROBI, verbose=False)
            # if not res :
            #     print('wowowowowowowow------------cplex failed!')
            #     res = prob.solve(solver=cp.CVXOPT, verbose=True)
            x = x.value
            fval = prob.value

            slack = x[-1]
            prev_x = x_0
            x_0 = x[0:-1]
            # Eval. progress
            feasible, objValue, _, _, max_violation, sum_violations, constraintValuesVehicle, constraintValuesObstacle = self.QCQP_evaluate(x_0)
            
            if self.nObst:
                max_constraint = max(constraintValuesVehicle.max(), constraintValuesObstacle.max() ) 
            else:
                max_constraint = 0 # since no obstacles
            
            fval = fval + self.qcqp['r0']
            delta_hat = (objValue_0 + slack_weight*max_violation_0) - fval # predicted decrease of obj
            delta = (objValue_0 + slack_weight*max_violation_0) - (objValue + slack_weight*max_violation) # real decrease of obj
            print('slack %8f max_violation %8f sum_violations %8f feas %d objVal %8f fval %8f max_constraint %e\n' %(slack, max_violation, sum_violations, feasible, objValue, fval, max_constraint) )
            
            objValue_0 = objValue
            max_violation_0 = max_violation
                    
            # Log context : dict

            optimization_log['P'].append(P)
            optimization_log['q'].append(q)
            optimization_log['Aineq'].append(Aineq)
            optimization_log['bineq'].append(bineq)
            optimization_log['lb'].append(lb)
            optimization_log['ub'].append(ub)
            optimization_log['x'].append(x)
            optimization_log['slack'].append(slack)
            optimization_log['SCP_ObjVal'].append(fval)
            optimization_log['QCQP_ObjVal'].append(objValue)
            optimization_log['delta_hat'].append(delta_hat)
            optimization_log['delta'].append(delta)
            optimization_log['du'].append(x_0)
            optimization_log['feasible'].append(feasible)
            optimization_log['prev_du'].append(prev_x)
            log_Traj, log_U = self.decode_deltaU(x_0)
            optimization_log['Traj'].append(log_Traj)
            optimization_log['U'].append(log_U)
            log_prevTraj, log_prevU = self.decode_deltaU(prev_x)
            optimization_log['prevTraj'].append(log_prevTraj)
            optimization_log['prevU'].append(log_prevU)
            
            if self.nVeh == 1:
                if  (abs(delta) < delta_tol) and (max_violation > cfg.QCQP.constraintTolerance) :
                    break
            if  (abs(delta) < delta_tol) and (max_violation <= cfg.QCQP.constraintTolerance): # max_violation is constraintTolerance in QCQP_evaluate.m.
                break
        print('iterations: ', i)
        return x_0 , feasible, objValue[0,0], optimization_log
    
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
    