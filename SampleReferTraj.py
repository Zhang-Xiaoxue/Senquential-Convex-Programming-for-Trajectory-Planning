import numpy as np
from math import sqrt

# %%
def normalize(x):
    return x/np.linalg.norm(x,2)

def sampleReferenceTrajectory(nSamples, referenceTrajectory, vehicle_x,vehicle_y, stepSize):
        ReferencePoints = np.zeros([nSamples,2])
    
        _, _, x, y, TrajectoryIndex = getShortestDistance(referenceTrajectory[:,0],referenceTrajectory[:,1],vehicle_x,vehicle_y)
        
        nLinePieces = referenceTrajectory.shape[0]
        currentPoint = np.array([[x,y]])
        
        # All line-segments are assumed to be longer than stepSize. 
        # Should it become necessary to have short line-segments this algorithm needs to be changed.
        for i in range(nLinePieces-1):
            assert (np.linalg.norm(referenceTrajectory[i+1,:]-referenceTrajectory[i,:],ord=2)>stepSize)         
        for i in range(nSamples):
            # make a step
            remainingLength = np.linalg.norm(currentPoint-referenceTrajectory[TrajectoryIndex,:],ord=2)
            if (remainingLength > stepSize)  or (TrajectoryIndex == nLinePieces):
                currentPoint = currentPoint + stepSize*normalize(referenceTrajectory[TrajectoryIndex,:]-referenceTrajectory[TrajectoryIndex-1,:])
            else:
                currentPoint = referenceTrajectory[TrajectoryIndex,:]
                TrajectoryIndex = min(TrajectoryIndex, nLinePieces-1)          
                currentPoint = currentPoint + (stepSize-remainingLength)*normalize(referenceTrajectory[TrajectoryIndex,:]-referenceTrajectory[TrajectoryIndex-1,:])
            
            # record step
            ReferencePoints[i,:] = currentPoint
        return ReferencePoints

def getShortestDistance(curve_x,curve_y,x,y):
    # Finds the point on a piecewise linear curve that is closest to a given point.
    # Params:
    #     curve_x, curve_y: [vector (n,)] A polygonal chain (a.k.a. piecewise linear curve), 
    #     x,y: [vector  (n,)] The point to be projected onto the curve
    # Returns:
    #     x_min, y_min: The projected point on the curve.
    #     arclength_min: Arc length on the curve between (curve_x(1),curve_y(1)) and (x_min,y_min).
    #     signed_distance_min: Signed distance between (x_min,y_min) and (x,y). Left ~ positive, right ~ negative.

    assert (isinstance(x, int) or isinstance(x, float))
    assert (isinstance(y, int) or isinstance(y, float))
    assert (len(curve_x) == len(curve_y))
    assert (len(curve_x) >= 2)
    arclength_sum = 0

    
    # Guess first point as minimum
    x_min = curve_x[1]
    y_min = curve_y[1]
    arclength_min=0
    signed_distance_min = sqrt((x-curve_x[1])**2 +(y-curve_y[1])**2)
    index_min = 2
    
    for j in range(1,len(curve_x)):
        xp, yp, signed_distance, lambda_para, piecelength =  Projection2D(curve_x[j-1],curve_y[j-1],curve_x[j],curve_y[j],x,y)

        # Projected point is between the end points.
        if (0 < lambda_para or j==1) and (lambda_para < 1 or j==len(curve_x)-1):
            if abs(signed_distance) < abs(signed_distance_min):
                x_min = xp
                y_min = yp
                signed_distance_min = signed_distance
                arclength_min= arclength_sum + lambda_para * piecelength
                index_min = j
        else:
            d_end = sqrt((x-curve_x[j])^2 +(y-curve_y[j])^2)
            if  abs(d_end) < abs(signed_distance_min):
                x_min = curve_x[j]
                y_min = curve_y[j]
                signed_distance_min = np.sign(signed_distance)*d_end
                arclength_min= arclength_sum + piecelength
                index_min = j
        arclength_sum = arclength_sum+piecelength

    return signed_distance_min, arclength_min, x_min, y_min, index_min

def Projection2D(x1,y1,x2,y2,x3,y3):
    # Takes a line and a point, determines the projection (point with shortest distance), distance and 'parameter' of the projection.
    # Params:
    #      x1,y1,x2,y2: Points that make a line.
    #      x3,y3:       The point to be projected.
    # Returns:
    #      xp,yp:       The projected point.
    #      projection_distance: Signed distance of (xp,yp) and (x3,y3)
    #      lambda:      The line 'parameter'. Is zero if (x1,y1)==(xp,yp)
    #                   and one if (x2,y2)==(xp,yp).
    #      line_segment_len: Distance between (x1,y1) and (x2,y2)

    b = sqrt((x2-x1)**2+(y2-y1)**2)
    line_segment_len = b
    if ( b != 0 ) :      
        # normalized direction p1 to p2        
        xn = (x2-x1)/b
        yn = (y2-y1)/b
        
        # vector p1 to p3
        x31 = x3 - x1
        y31 = y3 - y1

        # dot product to project p3 on the line from p1 to p2
        projection_dotproduct = xn * x31 + yn * y31
        
        # cross product to determine the distance from the line to p3
        projection_distance = xn * y31 - yn * x31
        
        # calculate the projected point
        xp = x1 + projection_dotproduct*xn
        yp = y1 + projection_dotproduct*yn

        lambda_para = projection_dotproduct/b   

    else:
       projection_distance = sqrt((x3-x1)**2+(y3-y1)**2)
       lambda_para = 0
       xp = x1
       yp = y1
    
    return xp, yp, projection_distance, lambda_para, line_segment_len
