# %%
import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy import Matrix
import json
import os

from Scenarios import Scenario
from Scenarios import Indices
from plotOnline import transformedRectangle

mypath = 'C:\\Users\\Lisnol\\National University of Singapore\\Ma Jun - Research-XX\\SCP\\'
# mypath = 'D:\\SoftWare\\DropBox\\Dropbox\\[5]SCP\\paper\\'

# %%
# load data to plot figure 1-4
num_vehicles = 8
scenario_choice = 'Circle'
controllerName = 'SCP' # 'MIQP', 'SCP'
angles = [2*pi/num_vehicles*(i+1) for i in range(num_vehicles)]

idx = Indices()
scenario = Scenario()
if scenario_choice == 'Circle':
    scenario.get_circle_scenario(angles)
elif scenario_choice == 'Frog':
    scenario.get_frog_scenario()
elif scenario_choice == 'Parallel':
    num_vehicles = 11
    scenario.get_parallel_scenario(num_vehicles)
    scenario.dsafeExtra = 0.9
scenario.complete_scenario()
Hp   = scenario.Hp
Nsim = scenario.Nsim
dt   = scenario.dt
nVeh = scenario.nVeh
nObst = scenario.nObst
nx   = scenario.model.nx
nu   = scenario.model.nu
steps = 10

with open('Data\\'+scenario_choice+'_num_'+str(scenario.nVeh)+'_control_'+controllerName+'.json', 'r') as f:
    result = json.load(f)
vehiclePathFullRes  = np.reshape(result['vehiclePathFullRes'],(nx, nVeh, scenario.ticks_total+1),order='F')          # (nx, nVeh, ticks_total+1)
obstaclePathFullRes = np.reshape(result['obstaclePathFullRes'], (nObst, 2, scenario.ticks_total+1) , order='F')      # (nObst, 2, ticks_total+1)
controlPathFullRes  = np.reshape(result['controlPathFullRes'], (nVeh, scenario.ticks_total+1), order='F')            # (nVeh, ticks_total+1)
controlPrediction = np.reshape(result['controlPredictions'], (Hp, nVeh, Nsim), order='F')                            # (Hp, nVeh, Nsim)
trajectoryPredictions = np.reshape(result['trajectoryPredictions'], (Hp, scenario.model.ny, nVeh, Nsim), order='F')  # (Hp, ny, nVeh, Nsim)
initial_pos = np.reshape(result['initial_pos'], (1, 2, nVeh, Nsim), order='F')                                       # (1, 2, nVeh, Nsim)
MPC_delay_compensation_trajectory = np.reshape(result['MPC_delay_compensation_trajectory'], (steps, nx, nVeh, Nsim), order='F')  # (steps, nx, nVeh, Nsim)
evaluations_obj_value =  np.reshape(result['evaluations_obj_value'], (Nsim,1), order='F')                            # Nsim
controllerRuntime = np.reshape(result['controllerRuntime'], (Nsim,1), order='F')                                     # (Nsim, 1)
stepTime = np.reshape(result['stepTime'], (Nsim,1), order='F')                                                       # (Nsim, 1)
ReferenceTrajectory = np.reshape(result['ReferenceTrajectory'], (Hp, 2, nVeh, Nsim), order='F')                      # (Hp, 2, nVeh, Nsim)
trajectoryPrediction_with_x0 = np.zeros((Hp+1, scenario.model.ny, nVeh, Nsim))

for step_idx in range(Nsim):
    trajectoryPrediction_with_x0[:,:,:,step_idx] = np.vstack([initial_pos[:,:,:,step_idx],trajectoryPredictions[:,:,:,step_idx] ])

## Colors
colorVehmap = cm.get_cmap('rainbow', nVeh)
colorVeh = colorVehmap(range(nVeh))

# %%
"""
############################################################################################################
############################ Plot One scenario One controller name  (No Compare)############################
############################################################################################################
"""

# %%
""" 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    Plot trajectories
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" 
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams['font.size'] = 18

nrows, ncols = 1, 1
figsize = (8,8)

for step_idx in range(Nsim):
    fig1, ax1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    ax1.set_aspect('equal', adjustable='box')

    tick_now = step_idx*scenario.ticks_per_sim
    vehiclePositions = vehiclePathFullRes[:,:,tick_now]
    obstaclePositions = obstaclePathFullRes[:,:,tick_now]

    for v in range(nVeh):
        # Sampled trajectory points
        ax1.scatter( ReferenceTrajectory[:,idx.x,v,step_idx], ReferenceTrajectory[:,idx.y,v,step_idx], marker='o', s=9, color=colorVeh[v,:])

        # predicted trajectory
        ax1.plot( trajectoryPrediction_with_x0[:,idx.x,v,step_idx],trajectoryPrediction_with_x0[:,idx.y,v,step_idx], color=colorVeh[v,:] )
    
        # vehicle trajectory delay prediction
        ax1.plot( MPC_delay_compensation_trajectory[:,idx.x,v,step_idx],  MPC_delay_compensation_trajectory[:,idx.y,v,step_idx], color=colorVeh[v,:], linewidth=2 )

        # Vehicle rectangles
        x = vehiclePositions[:,v]
        vehiclePolygon = transformedRectangle(x[idx.x],x[idx.y],x[idx.heading], scenario.Length[v],scenario.Width[v])
        ax1.fill(vehiclePolygon[0,:], vehiclePolygon[1,:], fc=colorVeh[v,:], ec='k')

    # Obstacle rectangles
    if nObst:
        for i in range(nObst):
            obstaclePolygon = transformedRectangle( obstaclePositions[i,idx.x], obstaclePositions[i,idx.y], 
                                scenario.obstacles[i,idx.heading], scenario.obstacles[i,idx.length], scenario.obstacles[i,idx.width])
            ax1.fill(obstaclePolygon[0,:],obstaclePolygon[1,:], color='gray')

    ax1.set_ylabel(r'$y$ [m]')
    ax1.set_xlabel(r'$x$ [m]')

    if scenario_choice == 'Parallel':
        ax1.set_xlim(-40,40)
        ax1.set_ylim(-25,25)
    
    plt.tight_layout()
    plt.savefig('figs\\'+str(step_idx)+'.png')

# %%
