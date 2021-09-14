# %%
import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy import Matrix

from Scenarios import Indices


# %%
idx = Indices()

def plotOnline(result,step_idx, ax1, ax2):
    # ax1 to plot control input for each vehicle, ax2 to plot the trajectory
    Iter = result.iterationStructs[step_idx]
    tick_now = step_idx*result.scenario.ticks_per_sim

    scenario = result.scenario
    vehiclePositions = result.vehiclePathFullRes[:,:,tick_now]
    obstaclePositions = result.obstaclePathFullRes[:,:,tick_now]
    controlPrediction = result.controlPredictions[:,:,step_idx]
    evaluation = result.evaluations[step_idx]
    
    # add initial position to trajectoryPrediction (not part of the
    # controller output).
    trajectoryPrediction_with_x0 = np.vstack([Iter.x0[:,0:2].T.reshape(1,2,result.scenario.nVeh,order='F'),result.trajectoryPredictions[:,:,:,step_idx] ])
    trajectoryPrediction = result.trajectoryPredictions[:,:,:,step_idx]
    delayPrediction = result.MPC_delay_compensation_trajectory[:,:,:,step_idx]

    nVeh = scenario.nVeh
    nObst = scenario.nObst
    Hp = scenario.Hp

    ## Colors
    colorVehmap = cm.get_cmap('jet', nVeh)
    colorVeh = colorVehmap(range(nVeh))
    
    ## Controller steering inputs
    if nVeh == 1:
        ax1.cla()
        ax1.step(range(scenario.Hp), 180/pi*np.array(controlPrediction[:,0]), color=colorVeh[0,:] )
        ax1.plot([0,scenario.Hp-1], 180/pi*Iter.uMax[0,0]*np.array([1,1]), 'k--')
        ax1.plot([0,scenario.Hp-1],-180/pi*Iter.uMax[0,0]*np.array([1,1]), 'k--')
        ax1.set_ylim(-1.5*180/pi*scenario.mechanicalSteeringLimit, 1.5*180/pi*scenario.mechanicalSteeringLimit)
        ax1.set_xlim(0,scenario.Hp-1)
        ax1.set_ylabel(r'$u [ ^\circ]$')
        ax1.set_title('Predicted steering angles')
        ax1.set_xlabel('Prediction steps')
        ax1.set_yticks(180/pi*scenario.mechanicalSteeringLimit*np.array([-1,0,1]))
        plt.pause(0.00000000000000000000000000000000000000001)
        # plt.show()
    else:
        for v in range(nVeh):
            ax1[v].cla()
            ax1[v].step(range(scenario.Hp), 180/pi*np.array(controlPrediction[:,v]), color=colorVeh[v,:] )
            ax1[v].plot([0,scenario.Hp-1], 180/pi*Iter.uMax[0,v]*np.array([1,1]), 'k--')
            ax1[v].plot([0,scenario.Hp-1],-180/pi*Iter.uMax[0,v]*np.array([1,1]), 'k--')
            ax1[v].set_ylim(-1.5*180/pi*scenario.mechanicalSteeringLimit, 1.5*180/pi*scenario.mechanicalSteeringLimit)
            ax1[v].set_xlim(0,scenario.Hp-1)
            ax1[v].set_ylabel(r'$u_{' +str(v+1)+ '} [ ^\circ]$')
            if v == 0:
                ax1[v].set_title('Predicted steering angles')
            elif v == nVeh:
                ax1[v].set_xlabel('Prediction steps')
            ax1[v].set_yticks(180/pi*scenario.mechanicalSteeringLimit*np.array([-1,0,1]))
            plt.pause(0.00000000000000000000000000000000000000001)
            # plt.show()

    """ plot second figure for trajectory """
    ## Simulation state / scenario plot
    ax2.cla()
    ax2.set_aspect('equal','box')
    
    ax2.set_xlabel(r'$x$ [m]')
    ax2.set_ylabel(r'$y$ [m]')

    ax2.set_xlim(scenario.plotLimits[0,:])
    ax2.set_ylim(scenario.plotLimits[1,:])
    
    for v in range(nVeh):
        # Sampled trajectory points
        ax2.scatter( Iter.ReferenceTrajectoryPoints[:,idx.x,v], Iter.ReferenceTrajectoryPoints[:,idx.y,v], marker='o', s=9, color=colorVeh[v,:])

        # predicted trajectory
        ax2.plot( trajectoryPrediction_with_x0[:,idx.x,v],trajectoryPrediction_with_x0[:,idx.y,v], color=colorVeh[v,:] )
    
        # vehicle trajectory delay prediction
        ax2.plot(  delayPrediction[:,idx.x,v],  delayPrediction[:,idx.y,v], color=colorVeh[v,:], linewidth=2 )

        # Vehicle rectangles
        x = vehiclePositions[:,v]
        vehiclePolygon = transformedRectangle(x[idx.x],x[idx.y],x[idx.heading], scenario.Length[v],scenario.Width[v])
        ax2.fill(vehiclePolygon[0,:], vehiclePolygon[1,:], fc=colorVeh[v,:], ec='k')
    
    # Obstacle rectangles
    if nObst:
        for i in range(nObst):
            obstaclePolygon = transformedRectangle( obstaclePositions[i,idx.x], obstaclePositions[i,idx.y], 
                                scenario.obstacles[i,idx.heading], scenario.obstacles[i,idx.length], scenario.obstacles[i,idx.width])
            ax2.fill(obstaclePolygon[0,:],obstaclePolygon[1,:], color='k')
    
    # Constraint Violation Markers
    cfg = result.cfg
    tol = cfg.QCQP.constraintTolerance
    maxConstraints = np.max(evaluation['constraintValuesVehicle'],axis=1)
    if nObst > 0:
        maxConstraints = np.maximum(maxConstraints, np.max(evaluation['constraintValuesObstacle'],axis=1))

    violations = maxConstraints>tol
    for v in range(nVeh):
        for k in range(Hp):
            if violations[v,k]:
                x = trajectoryPrediction[k,0,v]
                y = trajectoryPrediction[k,1,v]
                ax2.plot(x,y,'r*')
    plt.pause(0.00000000000000000000000000000000000000001)
    # plt.show()

def transformedRectangle( x,y,angle, Length, Width ):

    unitSquare = np.array([ [0,0,0,1],[1,0,0,1],[1,1,0,1],[0,1,0,1]]).T
    
    # Read this bottom-up
    translate_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]])
    rotate_z_matrix = np.array([[cos(angle), -sin(angle), 0, 0],[sin(angle), cos(angle), 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    scale_matrix = np.array([[Length, 0, 0, 0], [0, Width, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    fix_translate_matrix = np.array([[1, 0, 0, -0.5], [0, 1, 0, -0.5], [0, 0, 1, 0], [0, 0, 0, 1]])
    polygon_XY = translate_matrix @ rotate_z_matrix @ scale_matrix @ fix_translate_matrix @ unitSquare
            
    polygon_XY = polygon_XY[0:2,:]
    return polygon_XY

