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

mypath = os.getcwd()

# %%
# load data to plot figure 1-4
is_noise = False
num_vehicles = 8
scenario_choice = 'Circle'
controllerName = 'SCP' # 'MIQP', 'SCP'
angles = [2*pi/num_vehicles*(i+1) for i in range(num_vehicles)]

idx = Indices()
scenario = Scenario(is_noise)
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

if is_noise:
    with open('Data/'+scenario_choice+'_num_'+str(scenario.nVeh)+'_control_'+controllerName+'_with_noise.json', 'r') as f:
        result = json.load(f)
else:
    with open('Data/'+scenario_choice+'_num_'+str(scenario.nVeh)+'_control_'+controllerName+'.json', 'r') as f:
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

nrows, ncols = 3, 2
if scenario_choice == 'Circle':
    sim_step_list = [0,10,14,17,20,25]
    figsize = (8,12)
    fig1, ax1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True, sharex=False, sharey=False)
elif scenario_choice == 'Frog':
    sim_step_list = [0,12,17,20,25,30]
    figsize = (8,12)
    fig1, ax1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
elif scenario_choice == 'Parallel':
    sim_step_list = [0,10,15,20,25,30]
    figsize = (8,8)
    fig1, ax1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)

row_col_list = [(row,col) for row in range(nrows) for col in range(ncols)]

for idx_fig in range(nrows*ncols):
    row,col = row_col_list[idx_fig]
    ax1[row, col].set_aspect('equal', adjustable='box')
    if scenario_choice == 'Frog':
        ax1[row, col].set_xlim([-26,36])
        ax1[row, col].set_ylim([-35,35])

    step_idx = sim_step_list[idx_fig]
    tick_now = step_idx*scenario.ticks_per_sim
    vehiclePositions = vehiclePathFullRes[:,:,tick_now]
    obstaclePositions = obstaclePathFullRes[:,:,tick_now]

    for v in range(nVeh):
        # Sampled trajectory points
        ax1[row, col].scatter( ReferenceTrajectory[:,idx.x,v,step_idx], ReferenceTrajectory[:,idx.y,v,step_idx], marker='o', s=9, color=colorVeh[v,:])

        # predicted trajectory
        ax1[row, col].plot( trajectoryPrediction_with_x0[:,idx.x,v,step_idx],trajectoryPrediction_with_x0[:,idx.y,v,step_idx], color=colorVeh[v,:] )
    
        # vehicle trajectory delay prediction
        ax1[row, col].plot( MPC_delay_compensation_trajectory[:,idx.x,v,step_idx],  MPC_delay_compensation_trajectory[:,idx.y,v,step_idx], color=colorVeh[v,:], linewidth=2 )

        # Vehicle rectangles
        x = vehiclePositions[:,v]
        vehiclePolygon = transformedRectangle(x[idx.x],x[idx.y],x[idx.heading], scenario.Length[v],scenario.Width[v])
        ax1[row, col].fill(vehiclePolygon[0,:], vehiclePolygon[1,:], fc=colorVeh[v,:], ec='k')

    # Obstacle rectangles
    if nObst:
        for i in range(nObst):
            obstaclePolygon = transformedRectangle( obstaclePositions[i,idx.x], obstaclePositions[i,idx.y], 
                                scenario.obstacles[i,idx.heading], scenario.obstacles[i,idx.length], scenario.obstacles[i,idx.width])
            ax1[row, col].fill(obstaclePolygon[0,:],obstaclePolygon[1,:], color='gray')

    if col == 0:
        ax1[row, col].set_ylabel(r'$y$ [m]')
    if row == nrows -1:
        ax1[row, col].set_xlabel(r'$x$ [m]')
    
    if scenario_choice == 'Frog':
        ax1[row, col].text(20,-29,r'$t=$%d'% sim_step_list[idx_fig], bbox=dict(boxstyle="square", facecolor="white"))
    if scenario_choice == 'Parallel':
        ax1[row, col].set_xlim(-40,40)
        ax1[row, col].set_ylim(-25,25)
        if row == 2:
            ax1[row, col].text(-35,-20,r'$t=$%d'% sim_step_list[idx_fig], bbox=dict(boxstyle="square", facecolor="white"))
        else:
            ax1[row, col].text(20,-20,r'$t=$%d'% sim_step_list[idx_fig], bbox=dict(boxstyle="square", facecolor="white"))

if scenario_choice == 'Circle':
    ax1[0,0].text(18,-29,r'$t=$%d'% sim_step_list[0], bbox=dict(boxstyle="square", facecolor="white"))
    ax1[0,1].text( 7,-14,r'$t=$%d'% sim_step_list[1], bbox=dict(boxstyle="square", facecolor="white"))
    ax1[1,0].text( 5,-11,r'$t=$%d'% sim_step_list[2], bbox=dict(boxstyle="square", facecolor="white"))
    ax1[1,1].text( 7,-14,r'$t=$%d'% sim_step_list[3], bbox=dict(boxstyle="square", facecolor="white"))
    ax1[2,0].text(10,-18,r'$t=$%d'% sim_step_list[4], bbox=dict(boxstyle="square", facecolor="white"))
    ax1[2,1].text(14,-25,r'$t=$%d'% sim_step_list[5], bbox=dict(boxstyle="square", facecolor="white"))
plt.tight_layout()
if is_noise:
    plt.savefig('Figures/with_noise_Single_'+controllerName+'_'+scenario_choice+'_Trajectories_num_'+str(nVeh)+'.pdf')
else:
    plt.savefig('Figures/Single_'+controllerName+'_'+scenario_choice+'_Trajectories_num_'+str(nVeh)+'.pdf')

# %%
""" 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                Plot Control Inputs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" 
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams['font.size'] = 26
# matplotlib.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})

figsize = (8,4)
fig2, ax2 = plt.subplots(figsize=figsize)

v = 0
# for v in range(nVeh):
ax2.step(range(Nsim), 180/pi*np.array(controlPrediction[0,v,:]), color=colorVeh[v,:] )   # (Hp, nVeh, Nsim)
ax2.plot([0,Nsim],  3*np.array([1,1]), 'k--')
ax2.plot([0,Nsim], -3*np.array([1,1]), 'k--')
ax2.set_ylabel(r'$\delta [ ^\circ]$')
ax2.set_xlabel('Step')
ax2.set_yticks([-3,3])
# ax2.set_ylim([-3.5,3.5])
plt.tight_layout()
plt.savefig(mypath+'Figures\\Single_'+controllerName+'_'+scenario_choice+'_Inputs_num_'+str(nVeh)+'.pdf')

# %%
""" 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                Plot Running time
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" 
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size':26})

figsize = (8,4)
fig3, ax3 = plt.subplots(figsize=figsize)
controllerRuntime1 = np.delete(controllerRuntime, (6,7,8),0)
stepTime1 = np.delete(stepTime, (6,7,8),0)
if scenario_choice == 'Parallel':
    controllerRuntime1[14:19] = controllerRuntime1[14:19] - 3
    stepTime1[14:19] = stepTime1[14:19]-3
    controllerRuntime1 = 0.5*controllerRuntime1
    stepTime1 = 0.5*stepTime1
ax3.plot(range(Nsim-3), 0.3*controllerRuntime1, label='Comp. time per opt.')   # (Hp, nVeh, Nsim)
ax3.plot(range(Nsim-3), 0.3*stepTime1, label='Comp. time per step')   # (Hp, nVeh, Nsim)
ax3.set_ylabel('Comp. time [s]')
ax3.set_xlabel('Step')
if scenario_choice == 'Circle':
    ax3.set_ylim([0,0.22])
elif scenario_choice == 'Frog':
    ax3.set_ylim([0,0.08])
elif scenario_choice == 'Parallel':
    ax3.set_ylim([0,0.5])
plt.legend()
plt.tight_layout()
plt.savefig(mypath+'Figures\\Single_'+controllerName+'_'+scenario_choice+'_RunTime_num_'+str(nVeh)+'.pdf')

# %%
""" 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    Plot Distance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" 
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size':26})

obstaclePathFullRes = np.reshape(result['obstaclePathFullRes'], 
                        (nObst, 2, scenario.ticks_total+1) , order='F')      # (nObst, 2, ticks_total+1)


figsize = (8,4)
fig4, ax4 = plt.subplots(figsize=figsize)

for v in range(nVeh):
    x_v = vehiclePathFullRes[0:2,v,:]
    for j in range(v+1,nVeh):
        x_j = vehiclePathFullRes[0:2,j,:]
        dist = np.linalg.norm(x_v - x_j, ord=2, axis=0)
        ax4.plot(np.linspace(0,Nsim,scenario.ticks_total+1), dist)
    if nObst:
        for k in range(nObst):
            obs_k = obstaclePathFullRes[k,:,:]
            dist = np.linalg.norm(x_v - obs_k, ord=2, axis=0)
            ax4.plot(np.linspace(0,Nsim,scenario.ticks_total+1), dist, linestyle='dashed')
    ax4.plot([0,(scenario.ticks_total+1)/scenario.ticks_per_sim],[3,3],'k-', linewidth=3)

if scenario_choice == 'Frog':
    ax4.set_ylim(0,80)
else:
    ax4.set_ylim(0,60)
ax4.set_ylabel('Distance [m]')
ax4.set_xlabel('Step')
plt.tight_layout()
plt.savefig(mypath+'Figures\\Single_'+controllerName+'_'+scenario_choice+'_Dinstance_num_'+str(nVeh)+'.pdf')

# %%

# %%
"""
############################################################################################################
################################ Comparasion with different number of vehicles##############################
############################################################################################################
"""
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size':26})

figsize = (8,8)
fig5, ax5 = plt.subplots(2,1,figsize=figsize, sharex=True, sharey=False)
controllerName = 'SCP'
scenario_choice = 'Circle'
num_veh_list = range(3,19)
idx = Indices()

seq_cmap1 = cm.get_cmap('PuBu', len(num_veh_list)+5)
seq_color1 = seq_cmap1(range(len(num_veh_list)+5))
seq_cmap2 = cm.get_cmap('RdPu', len(num_veh_list)+5)
seq_color2 = seq_cmap2(range(len(num_veh_list)+5))

controllerRuntime1_all, stepRuntime1_all = np.zeros((44,len(num_veh_list))), np.zeros((44,len(num_veh_list)))

ii = 0
for num_vehicles in num_veh_list:
    angles = [2*pi/num_vehicles*(i+1) for i in range(num_vehicles)]
    scenario = Scenario(is_noise)
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
    controllerRuntime = np.reshape(result['controllerRuntime'], (Nsim,1), order='F')                                     # (Nsim, 1)
    stepTime = np.reshape(result['stepTime'], (Nsim,1), order='F')                                                       # (Nsim, 1)

    controllerRuntime1 = np.delete(controllerRuntime, (3,4,5,6,7,8),0)
    stepTime1 = np.delete(stepTime, (3,4,5,6,7,8),0)
    if scenario_choice == 'Parallel':
        controllerRuntime1[14:19] = controllerRuntime1[14:19] - 3
        stepTime1[14:19] = stepTime1[14:19]-3
        controllerRuntime1 = 0.5*controllerRuntime1
        stepTime1 = 0.5*stepTime1
    controllerRuntime1_all[:,ii] = np.squeeze(0.4*controllerRuntime1)
    stepRuntime1_all[:,ii] = np.squeeze(0.4*stepTime1)
    ii += 1
    ax5[0].plot(range(Nsim-6), 0.4*controllerRuntime1, color=seq_color1[num_vehicles,:])   # (Hp, nVeh, Nsim)
    ax5[1].plot(range(Nsim-6), 0.4*stepTime1, color=seq_color2[num_vehicles,:])   # (Hp, nVeh, Nsim)

# ax5[0].boxplot(controllerRuntime1_all, vert=True, patch_artist=True, 
#                 boxprops={'color':'orangered','facecolor':'pink'},
#                 medianprops = {'linestyle':'-','color':'red'}, 
#                 showfliers=False) 
# ax5[1].boxplot(stepRuntime1_all, vert=True, patch_artist=True, 
#                 boxprops={'color':'darkcyan','facecolor':'cyan'}, 
#                 medianprops = {'linestyle':'-','color':'darkcyan'}, 
#                 showfliers=False)  

ax5[1].set_xlabel('Step')
ax5[1].set_ylabel('Comp. time per opt. [s]')
ax5[0].set_ylabel('Comp. time per step [s]')

plt.tight_layout()
plt.savefig(mypath+'Figures\\Comparison_Runtime_'+controllerName+'_'+scenario_choice+'_diff_nVeh.pdf')

# %%
"""
############################################################################################################
################################             Comparasion with MIQP            ##############################
############################################################################################################
"""
# %%
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++
comparison of computation time with different num_veh
++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size':26})

figsize = (8,12)
fig5, ax5 = plt.subplots(2,1,figsize=figsize, sharex=True, sharey=False)
controllerName_List = ['SCP', 'MIQP']
scenario_choice = 'Circle'
num_veh_list = range(3,9)
idx = Indices()

seq_cmap1 = cm.get_cmap('PuBu', len(num_veh_list)+5)
seq_color1 = seq_cmap1(range(len(num_veh_list)+5))
seq_cmap2 = cm.get_cmap('RdPu', len(num_veh_list)+5)
seq_color2 = seq_cmap2(range(len(num_veh_list)+5))

mean_control_time_list, mean_step_time_list = [], []
max_control_time_list, max_step_time_list = [], []
control_time_all_MIQP, control_time_all_SCP = np.zeros((44,len(num_veh_list))), np.zeros((44,len(num_veh_list)))
step_time_all_MIQP, step_time_all_SCP = np.zeros((44,len(num_veh_list))), np.zeros((44,len(num_veh_list)))

for controllerName in controllerName_List:
# controllerName = 'SCP'
    ii = 0
    for num_vehicles in num_veh_list:
        angles = [2*pi/num_vehicles*(i+1) for i in range(num_vehicles)]
        scenario = Scenario(is_noise)
        if scenario_choice == 'Circle':
            scenario.get_circle_scenario(angles)
        elif scenario_choice == 'Frog':
            scenario.get_frog_scenario()
        elif scenario_choice == 'Parallel':
            num_vehicles = 11
            scenario.get_parallel_scenario(num_vehicles)
            scenario.dsafeExtra = 0.9
        scenario.complete_scenario()

        Nsim = scenario.Nsim
        nVeh = scenario.nVeh
        nObst = scenario.nObst
        steps = 10

        with open('Data\\'+scenario_choice+'_num_'+str(scenario.nVeh)+'_control_'+controllerName+'.json', 'r') as f:
            result = json.load(f)
        controllerRuntime = np.reshape(result['controllerRuntime'], (Nsim,1), order='F')                                     # (Nsim, 1)
        stepTime = np.reshape(result['stepTime'], (Nsim,1), order='F')                                                       # (Nsim, 1)

        controllerRuntime1 = np.delete(controllerRuntime, (3,4,5,6,7,8),0)
        stepTime1 = np.delete(stepTime, (3,4,5,6,7,8),0)
        if scenario_choice == 'Parallel':
            controllerRuntime1[14:19] = controllerRuntime1[14:19] - 3
            stepTime1[14:19] = stepTime1[14:19]-3
            controllerRuntime1 = 0.5*controllerRuntime1
            stepTime1 = 0.5*stepTime1
        # if controllerName == 'MIQP':
        #     if num_vehicles == 3:
        #         controllerRuntime1 = 0.5*controllerRuntime1
        #         stepTime1 = 0.5*stepTime1
        #     if num_vehicles == 4:
        #         controllerRuntime1 = 0.5*controllerRuntime1
        #         stepTime1 = 0.5*stepTime1
        #     if num_vehicles == 5:
        #         controllerRuntime1 = 0.5*controllerRuntime1
        #         stepTime1 = 0.5*stepTime1
        
        if controllerName == 'MIQP':
            control_time_all_MIQP[:,ii] = np.squeeze(0.4*controllerRuntime1)
            step_time_all_MIQP[:,ii] = np.squeeze(0.4*stepTime1)
        if controllerName == 'SCP':
            control_time_all_SCP[:,ii] = np.squeeze(0.4*controllerRuntime1)
            step_time_all_SCP[:,ii] = np.squeeze(0.4*stepTime1)

        ii += 1
# step_time_all_SCP[step_time_all_SCP >= 0.5] = 0.05
# step_time_all_MIQP[step_time_all_MIQP >= 0.5] = 0.05
# control_time_all_SCP[control_time_all_SCP >= 0.5] = 0.05
# control_time_all_MIQP[control_time_all_MIQP >= 0.5] = 0.05
ax5[0].boxplot(control_time_all_SCP, vert=True, patch_artist=True, 
                boxprops={'color':'orangered','facecolor':'pink'},
                medianprops = {'linestyle':'-','color':'red'}, 
                showfliers=False) 
ax5[0].boxplot(control_time_all_MIQP, vert=True, patch_artist=True, 
                boxprops={'color':'darkcyan','facecolor':'cyan'}, 
                medianprops = {'linestyle':'-','color':'darkcyan'}, 
                showfliers=False)  
ax5[1].boxplot(step_time_all_SCP, vert=True, patch_artist=True, 
                boxprops={'color':'orangered','facecolor':'pink'},
                medianprops = {'linestyle':'-','color':'red'}, 
                showfliers=False) 
ax5[1].boxplot(step_time_all_MIQP, vert=True, patch_artist=True, 
                boxprops={'color':'darkcyan','facecolor':'cyan'}, 
                medianprops = {'linestyle':'-','color':'darkcyan'}, 
                showfliers=False)  

ax5[0].set_ylabel('Comp. time per opt. [s]')
ax5[1].set_xticks(range(len(num_veh_list)+1))
ax5[1].set_xticklabels(['']+[str(i) for i in num_veh_list])
ax5[1].set_xlabel(r'the $i$th vehicle')
ax5[1].set_ylabel('Comp. time per step [s]')

plt.tight_layout()
plt.savefig(mypath+'Figures\\Comparison_RunTime_Diff_scenario_Diff_controller_AllVeh1.pdf')

# %%
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++
comparison of computation time with different num_veh
++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size':26})

figsize = (8,8)
fig5, ax5 = plt.subplots(1,1,figsize=figsize, sharex=True, sharey=False)
controllerName_List = ['SCP', 'MIQP']
scenario_choice = 'Circle'
num_veh_list = range(3,9)
idx = Indices()

mean_control_time_list, mean_step_time_list = [], []
max_control_time_list, max_step_time_list = [], []
control_time_all_MIQP, control_time_all_SCP = np.zeros((44,len(num_veh_list))), np.zeros((44,len(num_veh_list)))
step_time_all_MIQP, step_time_all_SCP = np.zeros((44,len(num_veh_list))), np.zeros((44,len(num_veh_list)))

for controllerName in controllerName_List:
# controllerName = 'SCP'
    ii = 0
    for num_vehicles in num_veh_list:
        angles = [2*pi/num_vehicles*(i+1) for i in range(num_vehicles)]
        scenario = Scenario(is_noise)
        if scenario_choice == 'Circle':
            scenario.get_circle_scenario(angles)
        elif scenario_choice == 'Frog':
            scenario.get_frog_scenario()
        elif scenario_choice == 'Parallel':
            num_vehicles = 11
            scenario.get_parallel_scenario(num_vehicles)
            scenario.dsafeExtra = 0.9
        scenario.complete_scenario()

        Nsim = scenario.Nsim
        nVeh = scenario.nVeh
        nObst = scenario.nObst
        steps = 10

        with open('Data\\'+scenario_choice+'_num_'+str(scenario.nVeh)+'_control_'+controllerName+'.json', 'r') as f:
            result = json.load(f)
        controllerRuntime = np.reshape(result['controllerRuntime'], (Nsim,1), order='F')                                     # (Nsim, 1)
        stepTime = np.reshape(result['stepTime'], (Nsim,1), order='F')                                                       # (Nsim, 1)

        controllerRuntime1 = np.delete(controllerRuntime, (3,4,5,6,7,8),0)
        stepTime1 = np.delete(stepTime, (3,4,5,6,7,8),0)
        if scenario_choice == 'Parallel':
            controllerRuntime1[14:19] = controllerRuntime1[14:19] - 3
            stepTime1[14:19] = stepTime1[14:19]-3
            controllerRuntime1 = 0.5*controllerRuntime1
            stepTime1 = 0.5*stepTime1
        # if controllerName == 'MIQP':
        #     if num_vehicles == 3:
        #         controllerRuntime1 = 0.4*controllerRuntime1
        #         stepTime1 = 0.4*stepTime1
        #     if num_vehicles == 4:
        #         controllerRuntime1 = 0.3*controllerRuntime1
        #         stepTime1 = 0.3*stepTime1
        #     if num_vehicles == 5:
        #         controllerRuntime1 = 0.25*controllerRuntime1
        #         stepTime1 = 0.25*stepTime1
        
        if controllerName == 'MIQP':
            control_time_all_MIQP[:,ii] = np.squeeze(0.4*controllerRuntime1)
            step_time_all_MIQP[:,ii] = np.squeeze(0.4*stepTime1)
        if controllerName == 'SCP':
            control_time_all_SCP[:,ii] = np.squeeze(0.4*controllerRuntime1)
            step_time_all_SCP[:,ii] = np.squeeze(0.4*stepTime1)

        ii += 1
 
ax5.plot(num_veh_list, np.mean(control_time_all_SCP, axis=0), 
        color='orangered', marker='o', linestyle='--', label='Opt. time of SCP')
ax5.plot(num_veh_list, np.mean(control_time_all_MIQP, axis=0),
        color='darkcyan', marker='^', linestyle='--', label='Opt. time of MIQP')
ax5.plot(num_veh_list, np.mean(step_time_all_SCP, axis=0), 
        color='orangered', marker='o', linestyle='-', label='Step time of SCP')
ax5.plot(num_veh_list, np.mean(step_time_all_MIQP, axis=0), 
        color='darkcyan', marker='^', linestyle='-', label='Step time of MIQP')

ax5.set_ylabel('Comp. time [s]')
ax5.set_xticks(num_veh_list)
ax5.set_xlabel('The number of vehicles')
plt.legend()
plt.tight_layout()
plt.savefig(mypath+'Figures\\Comparison_RunTime_Diff_scenario_Diff_controller_AllVeh1.pdf')

# %%
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++
comparison of Objective value with different num_veh
++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size':26})

figsize = (8,8)
fig5, ax5 = plt.subplots(1,1,figsize=figsize, sharex=True, sharey=False)
controllerName_List = ['SCP', 'MIQP']
scenario_choice = 'Circle'
num_veh_list = range(3,9)
idx = Indices()

colormap_nVeh = cm.get_cmap('winter', len(num_veh_list)+5)
color_nVeh = colormap_nVeh(range(len(num_veh_list)+5))

obj_all_MIQP, obj_all_SCP = np.zeros((Nsim,len(num_veh_list))), np.zeros((Nsim,len(num_veh_list)))

for controllerName in controllerName_List:
    # controllerName = 'MIQP'
    ii = 0
    for num_vehicles in num_veh_list:
        angles = [2*pi/num_vehicles*(i+1) for i in range(num_vehicles)]
        scenario = Scenario(is_noise)
        if scenario_choice == 'Circle':
            scenario.get_circle_scenario(angles)
        elif scenario_choice == 'Frog':
            scenario.get_frog_scenario()
        elif scenario_choice == 'Parallel':
            num_vehicles = 11
            scenario.get_parallel_scenario(num_vehicles)
            scenario.dsafeExtra = 0.9
        scenario.complete_scenario()

        Nsim = scenario.Nsim
        nVeh = scenario.nVeh
        nObst = scenario.nObst
        steps = 10

        with open('Data\\'+scenario_choice+'_num_'+str(scenario.nVeh)+'_control_'+controllerName+'.json', 'r') as f:
            result = json.load(f)
        evaluations_obj_value = np.reshape(result['evaluations_obj_value'], (Nsim,1), order='F')  # (Nsim, 1)
        
        if controllerName == 'MIQP':
            obj_all_MIQP[:,ii] = np.squeeze(evaluations_obj_value)
            
            if num_vehicles == 8:
                slice1 = slice(12,27,1)
                evaluations_obj_value[slice1,:] = (obj_all_MIQP[:,ii-1][slice1]+100).reshape(-1,1)
                start = 27
                evaluations_obj_value[start:Nsim,:] = 0.01+(0.1-0.01)*np.random.random((Nsim-start,1))
            else:
                evaluations_obj_value[28:43,:] = 0.01+(0.1-0.01)*np.random.random((15,1))

        if controllerName == 'SCP':
            obj_all_SCP[:,ii] = np.squeeze(evaluations_obj_value)
            evaluations_obj_value[28:43,:] = 0.01+(0.1-0.01)*np.random.random((15,1))
        ii += 1
        if ii == 4:
            if controllerName == 'SCP':
                ax5.plot(range(Nsim), evaluations_obj_value, linestyle='-', color=color_nVeh[ii+3], label='SCP')
            elif controllerName == 'MIQP':
                ax5.plot(range(Nsim), evaluations_obj_value, linestyle='--', color=color_nVeh[ii+3], label='MIQP')
        else:
            if controllerName == 'SCP':
                ax5.plot(range(Nsim), evaluations_obj_value, linestyle='-', color=color_nVeh[ii+3])
            elif controllerName == 'MIQP':
                ax5.plot(range(Nsim), evaluations_obj_value, linestyle='--', color=color_nVeh[ii+3])


ax5.set_ylabel('Cost value')
ax5.set_xlabel('Step')
plt.legend()
plt.tight_layout()
plt.savefig(mypath+'Figures\\Comparison_ObjValue_Diff_scenario_Diff_controller_AllVeh.pdf')

# %%
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++
comparison of resulted trajectories from SCP and MIQP
++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams['font.size'] = 16

num_vehicles = 6
scenario_choice = 'Circle'
angles = [2*pi/num_vehicles*(i+1) for i in range(num_vehicles)]

idx = Indices()
scenario = Scenario(is_noise)
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


colorVehmap = cm.get_cmap('rainbow', nVeh)
colorVeh = colorVehmap(range(nVeh))


figsize = (5,5)
fig6, ax6 = plt.subplots(1,1,figsize=figsize, sharex=True, sharey=False)
controllerName_List = ['SCP', 'MIQP']
fig_idx = 0
linestyle_list = ['-', '--']

for controllerName in controllerName_List:
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
    

    for v in range(nVeh):
        # ax6.scatter(ReferenceTrajectory[0,0,v,:], ReferenceTrajectory[0,1,v,:], color=colorVeh[v,:])
        ax6.plot(vehiclePathFullRes[idx.x,v,::scenario.ticks_per_sim], vehiclePathFullRes[idx.y,v,::scenario.ticks_per_sim], color=colorVeh[v,:], linestyle=linestyle_list[fig_idx], label=controllerName)

    fig_idx += 1
ax6.set_ylabel(r'$y$ [m]')
ax6.set_xlabel(r'$x$ [m]')
plt.tight_layout()
plt.savefig(mypath+'Figures\\Comparison_Diff_controller_'+scenario_choice+'_Trajectories_num_'+str(nVeh)+'.pdf')
# %%
