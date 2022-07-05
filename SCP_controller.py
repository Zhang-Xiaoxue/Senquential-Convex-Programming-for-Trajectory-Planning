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

        # self.dsafeExtra = 2

        self.scenario = scenario
        self.Iter = Iter
        self.prevOutput = prevOutput

        self.nu = scenario.model.nu
        self.ny = scenario.model.ny
        self.Hp = scenario.Hp
        self.nVeh = scenario.nVeh
        self.nObst = scenario.nObst
        self.dsafeExtra = scenario.dsafeExtra

        self.scenario_uLim = scenario.uLim

        self.mpc = MPCclass(scenario, Iter)
        self.qcqp = self.QCQP_formulate(scenario)
        self.u = np.zeros([self.nVeh*self.Hp,1])
    
    def SCP_controller(self, Iter):
        
        if self.prevOutput and ('u' in self.prevOutput):
            self.u = self.prevOutput['u'].reshape([self.Hp*self.nVeh,1],order='F')
         
        controllerOutput = {}
        controllerOutput['resultInvalid'] = False
        optimizerTimer = time.time()
        
        self.u, feasible , _, controllerOutput['optimization_log'] = self.SCP_optimizer( self.u )

        if self.nVeh == 1: 
            if feasible == False:
                # try left
                self.u = np.ones([self.u.shape[0],self.u.shape[0]])*self.scenario_uLim
                u_pos , feasible_pos, _ = self.SCP_optimizer( self.u )
                if feasible_pos :
                    self.u = u_pos
                else:
                    # try right
                    self.u = -np.ones([self.u.shape[0],self.u.shape[0]])*self.scenario_uLim
                    u_neg , feasible_neg, _ = self.SCP_optimizer( self.u )
                    if feasible_neg:
                        self.u = u_neg
                    else:
                        print('INFEASIBLE PROBLEM')
                        controllerOutput['resultInvalid'] = True
        
        controllerOutput['u'] = self.u
        trajectoryPrediction, U = self.forward_U(self.u)  # U: (14,1,8)
        U = np.squeeze(U[:,0,:])
        controllerOutput['optimizerTime'] = time.time() - optimizerTimer
        return U, trajectoryPrediction, controllerOutput
    
    def SCP_optimizer(self, u_approx):
        if abs(u_approx[0,0]) < np.spacing(1): # avoid numerical issues
            u_approx[0] = np.spacing(1)

        num_var = u_approx.shape[0]
        num_var_add_omega = u_approx.shape[0]+1
        num_constr = int(self.nVeh * ( self.Hp*(self.nVeh-1)/2 + self.Hp*self.nObst))

        _, objValue_0, _, _, max_violation_0, _, _, _ = self.QCQP_evaluate(u_approx)
        delta_tol = 1e-3
        psi_omega_weight = 1e5
        upper_bound_omega, lower_bound_omega = 1e25, 0
        max_SCP_iter = 20

        optimization_log = {'P':[], 'q':[], 'Aineq':[], 'bineq':[], 'lb':[], 'ub':[], 'x':[], 'slack':[],
                            'SCP_ObjVal':[], 'QCQP_ObjVal':[], 'delta_hat':[], 'delta':[], 'u':[], 'feasible':[],
                            'prev_u':[], 'Traj':[], 'U':[], 'prevTraj':[], 'prevU':[]}

        for i in range(max_SCP_iter):
            Aineq = np.zeros([num_constr,num_var_add_omega]) 
            bineq = np.zeros([num_constr,1])
            row = 0
            # VEHICLE AVOIDANCE
            for i in range(self.nVeh-1):
                for j in range(i+1,self.nVeh): # host vehicle is connected with the following vehicles.
                    for k in range(self.Hp):
                        Aineq[row,0:num_var] = (self.qcqp['Psi'][i,j,k].T + 2*u_approx.T @ self.qcqp['Phi'][i,j,k])
                        bineq[row,0] = -(self.qcqp['gamma'][i,j,k]-u_approx.T@self.qcqp['Phi'][i,j,k]@u_approx)
                        row += 1
            # if (A == Aineq): print("true")

            # OBSTACLE AVOIDANCE
            if self.nObst:
                # A_matrix_obs = [self.qcqp['q_o'][i,o,k].T + 2*u_approx.T @ self.qcqp['p_o'][i,o,k] for i in range(self.nVeh) for o in range(self.nObst) for k in range(self.Hp)]
                # b_vector_obs = [-(self.qcqp['r_o'][i,o,k]-u_approx.T @ self.qcqp['p_o'][i,o,k] @ u_approx) for i in range(self.nVeh) for o in range(self.nObst) for k in range(self.Hp)]
                for i in range(self.nVeh):
                    for o in range(self.nObst):
                        for k in range(self.Hp):
                            Aineq[row,slice(0,num_var,1)] = (self.qcqp['Psi_o'][i,o,k].T + 2*u_approx.T @ self.qcqp['Phi_o'][i,o,k])
                            bineq[row,0] = -(self.qcqp['gamma_o'][i,o,k]-u_approx.T @ self.qcqp['Phi_o'][i,o,k] @ u_approx)
                            row += 1
            # A_matrix = A_matrix_vehicle + A_matrix_obs
            # b_vector = b_vector_vehicle + b_vector_obs
            
            lb = -np.ones([num_var,1])*self.scenario_uLim
            ub =  np.ones([num_var,1])*self.scenario_uLim
            P = 2*self.qcqp['Phi0']
            
            # slack var
            q = np.vstack([self.qcqp['Psi0'], psi_omega_weight])
            P = sci.linalg.block_diag(P,0)
            Aineq[0:num_constr,-1] = -1 # <--  enable slack var just for collision avoidance constraints
            lb = np.vstack([lb, lower_bound_omega])
            ub = np.vstack([ub, upper_bound_omega])
            Aineq[abs(Aineq)<=1e-20] = 0

            # res = {'P_py': P, 'q_py':q, 'Aineq_py':Aineq, 'bineq_py':bineq, 'lb_py':lb, 'ub_py':ub, \
            #        'qcqp_p':self.qcqp['p'], 'qcqp_q':self.qcqp['q'], 'qcqp_r':self.qcqp['r'], \
            #        'qcqp_Psi0':self.qcqp['Psi0'], 'qcqp_Phi0':self.qcqp['Phi0'], 'qcqp_gamma0':self.qcqp['gamma0']}
            # io.savemat('res_py.mat', res)

            u_var = cp.Variable([P.shape[0],1])
            cost = 0.5*cp.quad_form(u_var,P) + q.T@u_var
            constr = [Aineq@u_var <= bineq]
            constr += [u_var <= ub]
            constr += [u_var >= lb]
            prob = cp.Problem(cp.Minimize(cost), constr)
            res = prob.solve(solver=cp.GUROBI, verbose=False)
            # if not res :
            #     print('wowowowowowowow------------cplex failed!')
            #     res = prob.solve(solver=cp.CVXOPT, verbose=True)
            u_var = u_var.value
            fval = prob.value

            slack = u_var[-1]
            prev_x = u_approx
            u_approx = u_var[0:-1]
            # Eval. progress
            feasible, objValue, _, _, max_violation, sum_violations, constraintValuesVehicle, constraintValuesObstacle = self.QCQP_evaluate(u_approx)
            
            if self.nObst:
                max_constraint = max(constraintValuesVehicle.max(), constraintValuesObstacle.max() ) 
            else:
                max_constraint = 0 # since no obstacles
            
            fval = fval + self.qcqp['gamma0']
            delta_hat = (objValue_0 + psi_omega_weight*max_violation_0) - fval # predicted decrease of obj
            delta = (objValue_0 + psi_omega_weight*max_violation_0) - (objValue + psi_omega_weight*max_violation) # real decrease of obj
            print('slack %8f sum_violations %8f feasible %d objVal %8f fval %8f \n' %(slack,  sum_violations, feasible, objValue, fval) )
            
            objValue_0 = objValue
            max_violation_0 = max_violation
                    
            # Log context : dict

            optimization_log['P'].append(P)
            optimization_log['q'].append(q)
            optimization_log['Aineq'].append(Aineq)
            optimization_log['bineq'].append(bineq)
            optimization_log['lb'].append(lb)
            optimization_log['ub'].append(ub)
            optimization_log['x'].append(u_var)
            optimization_log['slack'].append(slack)
            optimization_log['SCP_ObjVal'].append(fval)
            optimization_log['QCQP_ObjVal'].append(objValue)
            optimization_log['delta_hat'].append(delta_hat)
            optimization_log['delta'].append(delta)
            optimization_log['u'].append(u_approx)
            optimization_log['feasible'].append(feasible)
            optimization_log['prev_u'].append(prev_x)
            log_Traj, log_U = self.forward_U(u_approx)
            optimization_log['Traj'].append(log_Traj)
            optimization_log['U'].append(log_U)
            log_prevTraj, log_prevU = self.forward_U(prev_x)
            optimization_log['prevTraj'].append(log_prevTraj)
            optimization_log['prevU'].append(log_prevU)
            
            if self.nVeh == 1:
                if  (abs(delta) < delta_tol) and (max_violation > cfg.QCQP.constraintTolerance) :
                    break
            if  (abs(delta) < delta_tol) and (max_violation <= cfg.QCQP.constraintTolerance): # max_violation is constraintTolerance in QCQP_evaluate.m.
                break
        print('iterations: ', i)
        return u_approx , feasible, objValue[0,0], optimization_log
    
    def forward_U(self, u):
        U = np.zeros([self.Hp,self.nu,self.nVeh])
        Traj = np.zeros([self.Hp,self.ny,self.nVeh])
        u = u.reshape([self.nu,self.Hp,self.nVeh],order='F')
        for v in range(self.nVeh):
            U[:,:,v] = u[:,:,v].T

        # Predicted trajectory
        for v in range(self.nVeh):
            X = self.mpc.const_term[:,:,v] + self.mpc.Mathcal_B[:,:,v] @ u[:,:,v].T
            X = X.reshape([self.ny,self.Hp],order='F')
            for i in range(self.ny):
                Traj[:,i,v] = X[i,:]
        
        return Traj, U

    def QCQP_evaluate(self, U):
        c_linear = 0
        c_quad = 1e9
        objectiveTradeoffCoefficient = 1
        sum_violations = 0
        max_violation = 0

        feasible = True

        constraintValuesVehicle = np.full([self.nVeh,self.nVeh,self.Hp], -np.inf)
        constraintValuesObstacle = np.full([self.nVeh,self.nObst,self.Hp], -np.inf)
        
        objValue = U.T @ self.qcqp['Phi0'] @ U + self.qcqp['Psi0'].T @ U + self.qcqp['gamma0']
        feasibilityScore = objectiveTradeoffCoefficient*objValue
        feasibilityScoreGradient = objectiveTradeoffCoefficient*((self.qcqp['Phi0']+self.qcqp['Phi0'].T) @ U + self.qcqp['Psi0'])

        for v in range(self.nVeh):
            for k in range(self.Hp):
                # VEHICLES
                for v2 in range((v+1),self.nVeh):
                    ci = U.T @ self.qcqp['Phi'][v,v2,k,:,:] @ U + self.qcqp['Psi'][v,v2,k,:,:].T @ U + self.qcqp['gamma'][v,v2,k]
                    constraintValuesVehicle[v,v2,k] = ci
                    constraintValuesVehicle[v2,v,k] = ci
                    feasibilityScore = feasibilityScore + c_quad*max(ci,0)**2 + c_linear*max(ci,0)
                    if (ci > 0):
                        feasibilityScoreGradient = feasibilityScoreGradient + \
                            (c_quad*2*ci+c_linear)*((self.qcqp['Phi'][v,v2,k,:,:]+self.qcqp['Phi'][v,v2,k,:,:].T)@U
                             + self.qcqp['Psi'][v,v2,k,:,:])

                    if (ci > cfg.QCQP.constraintTolerance):
                        feasible = False
                        sum_violations = sum_violations + ci
                        max_violation = max(max_violation,ci)     
                    
                    # OBSTACLES
                    if self.nObst:
                        for ob in range(self.nObst):
                            ci = U.T @ self.qcqp['Phi_o'][v,ob,k,:,:] @ U + self.qcqp['Psi_o'][v,ob,k,:,:].T @ U + self.qcqp['gamma_o'][v,ob,k]
                            constraintValuesObstacle[v,ob,k] = ci
                            feasibilityScore = feasibilityScore + c_quad*max(ci,0)**2 + c_linear*max(ci,0)
                            if (ci > 0):
                                feasibilityScoreGradient = feasibilityScoreGradient + \
                                    (c_quad*2*ci+c_linear)*((self.qcqp['Phi_o'][v,ob,k,:,:]+self.qcqp['Phi_o'][v,ob,k,:,:].T)@U 
                                    + self.qcqp['Psi_o'][v,ob,k])

                            if (ci > cfg.QCQP.constraintTolerance):
                                feasible = False
                                sum_violations = sum_violations + ci
                                max_violation = max(max_violation,ci)

        return feasible, objValue, feasibilityScore, feasibilityScoreGradient, max_violation, sum_violations, constraintValuesVehicle, constraintValuesObstacle

    def extractor_itau(self, i, tau):
        assert(tau>self.Hp-1)
        assert(i>self.nVeh-1)
        return np.hstack([np.zeros([self.ny,tau*self.ny]), np.eye(self.ny), np.zeros([self.ny,(self.Hp-tau-1)*self.ny])])

    def extractor_ijtau(self, i, j, tau):
        assert(j<=i)
        E_itau = self.extractor_itau(i, tau)
        E_jtau = self.extractor_itau(j, tau)
        return np.hstack([np.zeros([self.ny, (i)*self.Hp*self.ny]), E_itau, np.zeros([self.ny, (j-i-1)*self.Hp*self.ny]), -E_jtau, self.zeros([self.ny, (self.nVeh-j-1)*self.Hp*self.ny]) ])

    def QCQP_formulate(self, scenario):
        
        # result matrices
        Phi0 = np.zeros([self.nVeh*self.nu*self.Hp,self.nVeh*self.nu*self.Hp])
        Psi0 = np.zeros([self.nVeh*self.nu*self.Hp,1])
        gamma0 = 0

        Phi = np.zeros([self.nVeh-1, self.nVeh, self.Hp, self.nVeh*self.nu*self.Hp, self.nVeh*self.nu*self.Hp])
        Psi = np.zeros([self.nVeh-1, self.nVeh, self.Hp, self.nVeh*self.nu*self.Hp, 1 ])
        gamma = np.zeros([self.nVeh-1, self.nVeh, self.Hp])

        Phi_obst = np.zeros([self.nVeh, self.nObst, self.Hp, self.nVeh*self.nu*self.Hp, self.nVeh*self.nu*self.Hp])
        Psi_obst = np.zeros([self.nVeh, self.nObst, self.Hp, self.nVeh*self.nu*self.Hp, 1])
        gamma_obst = np.zeros([self.nVeh, self.nObst, self.Hp])
        
        for v in range(self.nVeh):
            # OBJECTIVE FUNCTION MATRICES    
            veh1_slice = slice(self.nu*self.Hp*v, self.nu*self.Hp*(v+1),1)
            Phi0[veh1_slice,veh1_slice] = self.mpc.Phi_0[:,:,v]
            Psi0[veh1_slice,0] = np.squeeze(self.mpc.Psi_0[:,:,v])
            gamma0 = gamma0 + self.mpc.gamma_0[:,v]

            # CONSTRAINTS MATRICES
            for k in range(self.Hp):
                # VEHICLES AVOIDANCE
                intv = slice(k*self.ny, (k+1)*self.ny, 1)
                for v2 in range(v+1, self.nVeh):

                    veh2_slice = slice(v2*self.nu*self.Hp, (v2+1)*self.nu*self.Hp, 1)
                                
                    Phi[v,v2,k,veh1_slice,veh1_slice]  = - self.mpc.Mathcal_B[intv,:, v].T @ self.mpc.Mathcal_B[intv,:, v]
                    Phi[v,v2,k,veh2_slice,veh2_slice]  = - self.mpc.Mathcal_B[intv,:,v2].T @ self.mpc.Mathcal_B[intv,:,v2]
                    Phi[v,v2,k,veh1_slice,veh2_slice]  =   self.mpc.Mathcal_B[intv,:, v].T @ self.mpc.Mathcal_B[intv,:,v2]
                    Phi[v,v2,k,veh2_slice,veh1_slice]  =   self.mpc.Mathcal_B[intv,:,v2].T @ self.mpc.Mathcal_B[intv,:, v]
                    
                    b =  self.mpc.const_term[intv,0,v] - self.mpc.const_term[intv,0,v2]

                    Psi[v,v2,k,veh1_slice,0] = -2*self.mpc.Mathcal_B[intv,:, v].T @ b
                    Psi[v,v2,k,veh2_slice,0] =  2*self.mpc.Mathcal_B[intv,:,v2].T @ b
                    gamma[v,v2,k] = (scenario.dsafeVehicles[v,v2] + self.dsafeExtra)**2 - b.T @ b
                
                # OBSTACLE AVOIDANCE
                # Obstacle in next time step
                if self.nObst:
                    for o in range(self.nObst): 
                        Phi_obst[v,o,k,veh1_slice,veh1_slice] = - self.mpc.Mathcal_B[intv,:,v].T @ self.mpc.Mathcal_B[intv,:,v]
                        b = self.mpc.const_term[intv,0,v] - self.Iter.obstacleFutureTrajectories[o,:,k].T
                        Psi_obst[v,o,k,veh1_slice,0] = -2*self.mpc.Mathcal_B[intv,:,v].T @ b
                        gamma_obst[v,o,k] = (scenario.dsafeObstacles[v,o] + self.dsafeExtra)**2 - b.T @ b

        for v in range(self.nVeh):
            for k in range(self.Hp):
                for v2 in range(v+1, self.nVeh):
                    Phi[v,v2,k,:,:] = 0.5*(Phi[v,v2,k,:,:]+Phi[v,v2,k,:,:].T)
                if self.nObst:
                    for o in range(self.nObst):
                        Phi_obst[v,o,k,:,:] = 0.5*(Phi_obst[v,o,k,:,:]+Phi_obst[v,o,k,:,:].T)

        Phi[abs(Phi)<=1e-30] = 0
        Psi[abs(Psi)<=1e-30] = 0
        
        qcqp_dict = {'Phi0':Phi0, 'Psi0':Psi0, 'gamma0':gamma0, 'Phi':Phi, 'Psi':Psi, 'gamma':gamma, 'Phi_o':Phi_obst, 'Psi_o':Psi_obst, 'gamma_o':gamma_obst}
            
        return qcqp_dict
    
    def evaluateInOriginalProblem(self, controlPrediction, trajectoryPrediction,options):
        evaluation = {}
        evaluation['predictionObjectiveValueX'] = 0
        evaluation['predictionObjectiveValueU'] = 0
        
        # trajectory Prediction  error term
        sqRefErr = (self.Iter.ReferenceTrajectoryPoints-trajectoryPrediction)**2
        for v in range(self.nVeh):
            evaluation['predictionObjectiveValueX'] = evaluation['predictionObjectiveValueX'] + \
                self.scenario.Q[v] * sqRefErr[0:-1,:,v].sum() + \
                self.scenario.Q_final[v] * sqRefErr[-1,:,v].sum()
        
        # steering Prediction term
        u = controlPrediction[0:self.Hp,:]
        sqU = u**2
        for v in range(self.nVeh):
            evaluation['predictionObjectiveValueU'] = evaluation['predictionObjectiveValueU'] + \
                self.scenario.R[v] * sqU[:,v].sum()
        
        evaluation['predictionObjectiveValue'] = evaluation['predictionObjectiveValueX'] + evaluation['predictionObjectiveValueU']    
        
        # crash prediction check based on QCQP
        u = u.reshape(u.shape[0]*u.shape[1],1,order='F')
        evaluation['predictionFeasibleQCQP'],_,_,_,_,_,evaluation['constraintValuesVehicleQCQP'],evaluation['constraintValuesObstacleQCQP'] = self.QCQP_evaluate(u)
        
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
    