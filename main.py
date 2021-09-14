# %%
import numpy as np
from math import pi, atan, ceil, sin, cos
from scipy import integrate
from scipy import io
import time
import pickle
from sympy import Matrix
import matplotlib.pyplot as plt
import json

from Scenarios import Scenario
from SCP_controller import SCPcontroller
from MIQP import MIQPcontroller
from MPC_Iter import MPCclass, IterClass
from Config import Config
from Scenarios import Indices
from Model import BicyleModel
from plotOnline import plotOnline

import warnings

warnings.filterwarnings('ignore')


# %%
class Simulation():
    def __init__(self, scenarios, doOnlinePlot, isNoise):
        self.isNoise = isNoise
        self.doOnlinePlot = doOnlinePlot

        # Create the result.scenario.
        self.scenario = scenarios
        self.scenario.complete_scenario()
            
        # Variables and Parameters
        Hp   = self.scenario.Hp
        Nsim = self.scenario.Nsim
        dt   = self.scenario.dt
        nVeh = self.scenario.nVeh
        nObst = self.scenario.nObst
        nx   = self.scenario.model.nx
        nu   = self.scenario.model.nu

        # if that ever changes, the code needs to be fixed in many places 
        # as this assumption was often implicitly made.
        assert (nu == 1) 
        
        ny   = self.scenario.model.ny
        self.cfg = Config()
        self.idx = Indices()

        # Result variables
        self.controllerOutputs = []  # np.array([Nsim,1]) # cell
        self.iterationStructs =  []  # np.array([Nsim,1]) # cell
        self.obstaclePathFullRes = np.full([nObst,2,self.scenario.ticks_total+1],np.nan)
        self.vehiclePathFullRes = np.full([nx,nVeh,self.scenario.ticks_total+1],np.nan)
        self.controlPathFullRes = np.full([nVeh,self.scenario.ticks_total+1],np.nan)
        self.controlPredictions = np.zeros([Hp,nVeh,Nsim])
        self.trajectoryPredictions = np.zeros([Hp,ny,nVeh,Nsim])
        self.controllerRuntime = np.zeros([Nsim,1])
        self.stepTime = np.zeros([Nsim,1])
        self.evaluations = []    # np.array([Nsim,1]) # cell
        self.steeringLimitsExceeded = False
        
        
        # Calculate obstacle trajectories
        if nObst:
            for tik in range(self.scenario.ticks_total+1):
                self.obstaclePathFullRes[:,self.idx.x,tik] = np.squeeze((tik*self.scenario.tick_length)*self.scenario.obstacles[:,self.idx.speed,:]*
                                                                            np.cos( self.scenario.obstacles[:,self.idx.heading,:] )
                                                                             + self.scenario.obstacles[:,self.idx.x,:])
                self.obstaclePathFullRes[:,self.idx.y,tik] = np.squeeze((tik*self.scenario.tick_length)*self.scenario.obstacles[:,self.idx.speed,:]*
                                                                            np.sin( self.scenario.obstacles[:,self.idx.heading,:] ) 
                                                                            + self.scenario.obstacles[:,self.idx.y,:])
        
        # Initialize the first step.
        for v in range(nVeh):
            self.vehiclePathFullRes[:,v,0] = self.scenario.x0[v][:,0]
            self.controlPathFullRes[v,0:self.scenario.ticks_delay_u+self.scenario.ticks_per_sim+1] = self.scenario.u0[v]
        
    def runsimulation(self, controllerName):
        # save data
        QCQP_obj_value_list = []
        evaluations_obj_value = []
        ReferenceTrajectory = np.zeros((self.scenario.Hp,2,self.scenario.nVeh,self.scenario.Nsim))
        initial_pos = np.zeros((2, self.scenario.nVeh, self.scenario.Nsim))

        if self.doOnlinePlot:
            fig1,ax1 = plt.subplots(self.scenario.nVeh,1)
            plt.ion()
            fig2,ax2 = plt.subplots()
            plt.ion()
        # begin simulation loop
        print('Beginning simulation ... ')
        steps = 10
        self.MPC_delay_compensation_trajectory = np.zeros([steps,self.scenario.model.nx,self.scenario.nVeh,self.scenario.Nsim])
        for i in range(self.scenario.Nsim):
            print('###################### iter: {} ###################'.format(i))
            self.stepTimer = time.time()
            tick_now = i*self.scenario.ticks_per_sim # tick index of the current moment.
            tick_of_measurement = max(0,tick_now - self.scenario.ticks_delay_x)  # tick index from a past moment, where the state x was measured.
            tick_of_actucator = min(self.scenario.ticks_total+1,tick_now+1+self.scenario.ticks_delay_u+self.scenario.ticks_per_sim) # tick index in the future when the result of a MPC calculation that starts now is applied.
                        
            # determine controller inputs
            uMax = np.zeros([1,self.scenario.nVeh])
            for v in range(self.scenario.nVeh):
                # Steering constraints: At high speeds lateral acceleration needs to be limited.
                dynamicSteeringLimit = atan((self.scenario.lateralAccelerationLimit*(self.scenario.Lf[v]+self.scenario.Lr[v]))/(self.vehiclePathFullRes[self.idx.speed,v,tick_now]**2))
                uMax[0,v] = min(self.scenario.mechanicalSteeringLimit, dynamicSteeringLimit)
            
            x_measured = self.vehiclePathFullRes[:,:,tick_of_measurement].T        
            # Extract the control output values for the timeslice that starts at (T_now - T_x) and ends at (T_now + T_MPC + T_u).
            # The index slicing is a little confusing because of the truncation to the simulation timespan:
            # In the beginning of the simulation there are no control outputs, so they are assumed to be zero.
            u_path = np.zeros([self.scenario.nVeh,self.scenario.ticks_delay_x + self.scenario.ticks_per_sim + self.scenario.ticks_delay_u])       
            u_path[:,slice(max(self.scenario.ticks_delay_x-tick_now,0),max(self.scenario.ticks_delay_x-tick_now,0)+tick_of_actucator-1-tick_of_measurement,1)] = self.controlPathFullRes[:,tick_of_measurement+1:tick_of_actucator]
            
            # call controller
            controllerTimer = time.time()        
            # MPC_init is part of the controller and thus inside 'controllerTimer'.
            # since it is a preprocessing step necessary for every MPC controller, it is called here.
            Iter = IterClass(self.scenario, x_measured, u_path, self.obstaclePathFullRes[:,:,tick_of_measurement], uMax)
            self.MPC_delay_compensation_trajectory[:,:,:,i] = Iter.MPC_delay_compensation_trajectory
            if i == 0:
                Iter.reset = 1
            else:
                Iter.reset = 0
            if controllerName == 'SCP':
                if i == 0:
                    self.controller = SCPcontroller(self.scenario,Iter,self.controllerOutputs)
                else:
                    self.controller = SCPcontroller(self.scenario,Iter,self.controllerOutputs[i-1])
                U, self.trajectoryPredictions[:,:,:,i],controllerOutputs_i = self.controller.SCP_controller(Iter)
            if controllerName == 'MIQP':
                if i == 0:
                    self.controller = MIQPcontroller(self.scenario,Iter,self.controllerOutputs)
                else:
                    self.controller = MIQPcontroller(self.scenario,Iter,self.controllerOutputs[i-1])
                U, self.trajectoryPredictions[:,:,:,i], controllerOutputs_i = self.controller.MIQP_controller(Iter)
            self.controllerOutputs.append(controllerOutputs_i)
            self.controllerRuntime[i] = time.time() - controllerTimer
            
            # check steering constraints
            for v in range(self.scenario.nVeh):
                if U.ndim == 1:
                    U = U.reshape(-1,1)
                if (abs(U[0,v]) > Iter.uMax[:,v]+1e-3):
                    print('Steering limit exceeded for vehicle {}: |{}|>{}\n'.format(v, U[0,v], Iter.uMax[:,v]))
                    self.steeringLimitsExceeded = True
                if (abs(U[0,v]-Iter.u0[v,:]) > self.scenario.duLim+1e-3):
                    print('Steering rate exceeded for vehicle {}: |{}|>{}\n'.format(v, U[1,v]-Iter.u0[v,:],self.scenario.duLim))
                    self.steeringLimitsExceeded = True

                for j in range(1,self.scenario.Hp):

                    if (abs(U[j,v]) > Iter.uMax[:,v]+1e-3):
                        print('Steering limit exceeded for vehicle {}: |{}|>{}\n'.format(v, U[j,v], Iter.uMax[:,v]))
                    self.steeringLimitsExceeded = True
                    if (abs(U[j,v]-U[j-1,v]) > self.scenario.duLim+1e-3):
                        print('Steering rate exceeded for vehicle {}: |{}|>{}\n'.format(v,U[j,v]-U[j-1,v],self.scenario.duLim))
                        self.steeringLimitsExceeded = True
            
            # enforce steering constraints
            for v in range(self.scenario.nVeh):
                U[0,v] = min(U[0,v],  Iter.uMax[:,v])
                U[0,v] = max(U[0,v], -Iter.uMax[:,v])
                U[0,v] = min(U[0,v],  Iter.u0[v,:]+self.scenario.duLim)
                U[0,v] = max(U[0,v],  Iter.u0[v,:]-self.scenario.duLim)         
                for j in range(1,self.scenario.Hp):
                    U[j,v] = min(U[j,v],  Iter.uMax[:,v])
                    U[j,v] = max(U[j,v], -Iter.uMax[:,v])
                    U[j,v] = min(U[j,v], U[j-1,v]+self.scenario.duLim)
                    U[j,v] = max(U[j,v], U[j-1,v]-self.scenario.duLim)            
            
            for v in range(self.scenario.nVeh):
                # Save the controller's signal. The controller's signal is saved with a shift of 'ticks_per_sim+ticks_delay_u' into the future. 
                # This is done to simulate controller and actuator delay: 
                #    * It takes 'ticks_per_sim' to execute the MPC controller and 'ticks_delay_u' for the signal to propagate to the actual steering angle.              
                slice_array =  np.array(range(i*self.scenario.ticks_per_sim+1+self.scenario.ticks_delay_u+self.scenario.ticks_per_sim,(i+1)*self.scenario.ticks_per_sim+1+self.scenario.ticks_delay_u+self.scenario.ticks_per_sim))
                slice_array[slice_array>=self.controlPathFullRes.shape[1]-1] = self.controlPathFullRes.shape[1]-1
                self.controlPathFullRes[v,slice_array] = U[0,v]
                            
                # simulate with the controller's signal
                ode_prob = integrate.ode(self.scenario.model.odes_).set_integrator('dopri5',atol=1e-8,rtol=1e-8)
                model_step = np.zeros([ self.scenario.ticks_per_sim+1, self.scenario.model.nx])
                timelist = np.linspace(i*self.scenario.dt, (i+1)*self.scenario.dt, self.scenario.ticks_per_sim+1)
                for k in range(self.scenario.ticks_per_sim+1):
                    ode_prob.set_initial_value(self.vehiclePathFullRes[:,v,tick_now], t=i*self.scenario.dt).set_f_params(self.controlPathFullRes[v,min(self.controlPathFullRes.shape[1]-1,ceil(timelist[k]/self.scenario.tick_length)+1)], self.scenario.Lf[v],self.scenario.Lr[v])
                    model_step[k,:] = ode_prob.integrate(timelist[k])
                self.vehiclePathFullRes[:,v,slice(self.scenario.ticks_per_sim*i+1,self.scenario.ticks_per_sim*(i+1)+1,1)] = model_step[1:self.scenario.ticks_per_sim+1,:].T
            self.controlPredictions[:,:,i] = U
            
            self.stepTime[i,:] = time.time() - self.stepTimer

            # Check if self.scenario is initially feasible
            IsFeasible,_,_,_,_,_,_,_ = self.controller.QCQP_evaluate(np.zeros([self.scenario.nVeh*self.scenario.Hu,1]))
            if i==0 and not IsFeasible:
                raise Exception('scenario initially infeasible!')
            options = {'ignoreQCQPcheck': True}
            self.evaluations.append(self.controller.evaluateInOriginalProblem(U, self.trajectoryPredictions[:,:,:,i], options ))
            evaluations_obj_value.append(self.controller.evaluateInOriginalProblem(U, self.trajectoryPredictions[:,:,:,i], options )['predictionObjectiveValue'])
                
            self.iterationStructs.append(Iter)
            ReferenceTrajectory[:,:,:,i] = Iter.ReferenceTrajectoryPoints
            initial_pos[:,:,i] = Iter.x0[:,0:2].T
            
            if self.doOnlinePlot:
                plotOnline(self,i,ax1,ax2)  # ax1 to plot control input for each vehicle, ax2 to plot the trajectory
                  
        # End of simulation loop

        result_for_plot1 = {'vehiclePathFullRes':self.vehiclePathFullRes.tolist(), 
                           'obstaclePathFullRes':self.obstaclePathFullRes.tolist(), 
                           'controlPathFullRes':self.controlPathFullRes.tolist(),
                           'controlPredictions':self.controlPredictions.tolist(),                           
                           'trajectoryPredictions':self.trajectoryPredictions.tolist(),
                           'initial_pos': initial_pos.tolist(),
                           'ReferenceTrajectory': ReferenceTrajectory.tolist(),
                           'MPC_delay_compensation_trajectory':self.MPC_delay_compensation_trajectory.tolist(),
                           'evaluations_obj_value':evaluations_obj_value, 
                           'controllerRuntime':self.controllerRuntime.tolist(),
                           'stepTime':self.stepTime.tolist(),
                           }
        # result_for_plot = [result_for_plot1, self.controllerOutputs, self.controllerOutputs]
        if self.isNoise:
            with open('Data/'+scenario_choice+'_num_'+str(self.scenario.nVeh)+'_control_'+controllerName+'_with_noise.json', 'w') as f:
                json.dump(result_for_plot1, f)
        else:
            with open('Data/'+scenario_choice+'_num_'+str(self.scenario.nVeh)+'_control_'+controllerName+'.json', 'w') as f:
                json.dump(result_for_plot1, f)

# %%
if __name__ == '__main__':
    is_noise = False
    num_veh_list = range(3,9)
    # for num_vehicles in num_veh_list:
    num_vehicles = 8
    print('### num_veh : ', num_vehicles)
    angles = [2*pi/num_vehicles*(i+1) for i in range(num_vehicles)]
    scenarios = Scenario(is_noise)
    scenario_choice = 'Circle'
    if scenario_choice == 'Circle':
        scenarios.get_circle_scenario(angles)
    elif scenario_choice == 'Frog':
        scenarios.get_frog_scenario()
    elif scenario_choice == 'Parallel':
        num_vehicles = 11
        scenarios.get_parallel_scenario(num_vehicles)
        scenarios.dsafeExtra = 0.9
    controllerName = 'SCP' # 'MIQP', 'SCP'
    if controllerName == 'MIQP':
        scenarios.dsafeExtra = 0
    simu = Simulation(scenarios, doOnlinePlot=False, isNoise=is_noise)
    simu.runsimulation(controllerName)
        
    print('----------------------------------------------------------------------------------------------------------------------------')

# %%
