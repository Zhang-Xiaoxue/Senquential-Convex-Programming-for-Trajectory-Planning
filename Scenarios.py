# %%
import numpy as np
from math import sin,cos,pi,sqrt,ceil,floor
from Model import BicyleModel, DefaultVehicle

# %%
def round_up(value):
    # 替换内置round函数,实现保留2位小数的精确四舍五入
    return round(value + 0.00000001)

# %%
class DefaultObstacle():
    # obstacles are modeled as rotated rectangles that can move with constant speed and direction.
    def __init__(self):
        # obstacle's center position [m]
        self.x = 0
        self.y = 0
            
        self.heading = 0  # obstacle's rotation AND directon of movement [radians]
        self.speed = 0    # [m/s]
        self.length = 2   # obstacle's size measured along its direction of movement [m]
        self.width = 2   # obstacle's size measured at right angels to its direction of movement[m]

class Indices():
    def __init__(self):
        # General
        self.x = 0
        self.y = 1
        self.heading = 2
        self.speed = 3
        
        # Vehicle
        self.acceleration = 4
        
        # Obstacle
        self.length = 4
        self.width = 5

# %%
class Scenario():

    def __init__(self, is_noise):
        # A tick is the shortest timespan considered during simulation.
        # All other time constants must be integral multiples. [s]
        self.tick_length = 0.01
        self.T_end = 20 # Duration of simulation. [s]
        self.delay_x = 0 # Measurement transport delay [s].
        self.delay_u = .03 # Control transport delay [s].
        self.dt   = 0.4 # MPC sample time [s]
        self.Hp   = 10 # Prediction horizon
        self.Hu   = 10 # Control horizon
        self.lateralAccelerationLimit = 9.81/2 # [m/s^2]
        self.mechanicalSteeringLimit = pi/180 * 3 # [radians]
        self.duLim = self.mechanicalSteeringLimit * 2 # Steering limit per timestep [radians]
        self.model = BicyleModel(is_noise)
        self.nVeh = 0 # Number of vehicles

        self.dsafeExtra = 1
        
        
        # The following variables are vectors or matrices,
        # where each row corresponds to a vehicle.
        # For more info see defaultVehicle()
        self.Q = [] # Trajectory deviation weights for MPC
        self.Q_final = [] # Trajectory deviation weight for the final prediction step    
        self.R = [] # Steering rate weights for MPC
        self.Lf = [] # Distance between vehicle center and front axle center [m]
        self.Lr = [] # Distance between vehicle center and rear axle center [m]
        self.Length = [] # Vehicle length (bumper to bumper)[m]
        self.Width = [] # Vehicle width [m]
        self.RVeh = [] # Vehicle radius, distance between center and a corner [m]
        self.x0 = [] # Vehicle state: row=[x y heading speed acceleration]  units:     [m m radians m/s   m/s^2       ]
        self.u0 = [] # Initial steering angle [radians]
        
        # The reference trajectory for each vehicle. [m]
        # referenceTrajectories{v}(i,1) corresponds to the x-coordiante of the
        # i-th point in the piecewise linear reference-curve of vehicle v.
        # Likewise referenceTrajectories{v}(i,2) corresponds to the y-coordiante.
        self.referenceTrajectories = []

        # Obstacle data: Each row corresponds to an obstacle and has the
        # strucure: [x y heading speed length width]
        # For more info see defaultObstacle().
        self.obstacles = []
        
        # Limits for the plot axes [m]. Format:[xmin xmax ymin ymax]
        self.plotLimits = 5*np.array([[-10,10],[-10,10]])

    def addVehicle(self, vehicle):
        self.model.makeInitState(vehicle)
        self.x0.append(self.model.makeInitStateVector)
        self.nVeh = self.nVeh + 1
        self.Q.append(vehicle.Q)
        self.Q_final.append(vehicle.Q_final)
        self.R.append(vehicle.R)
        self.RVeh.append(np.linalg.norm(np.array([vehicle.Length,vehicle.Width]),2)/2)
        self.Lf.append(vehicle.Lf)
        self.Lr.append(vehicle.Lr)
        self.Width.append(vehicle.Width)
        self.Length.append(vehicle.Length)
        self.u0.append(vehicle.u0)
        
        self.referenceTrajectories.append(vehicle.referenceTrajectory)
    
    def addObstacle(self, obstacle):
        self.obstacles.append(np.array([obstacle.x, obstacle.y, obstacle.heading, obstacle.speed, obstacle.length, obstacle.width]).reshape(-1,1))
        # append (6,1) vector

    def get_circle_scenario(self, angles):
        """ angles: list """
        radius = 30
        for angle in angles:
            s = sin(angle)
            c = cos(angle)
            veh = DefaultVehicle()
            veh.labelOffset = np.array([[3,-3]])@np.array([[c,s],[-s,c]])+np.array([[-2,0]])
            veh.x_start = -c*radius
            veh.y_start = -s*radius
            veh.heading = angle
            veh.referenceTrajectory = np.array([[-c*radius,-s*radius],[c*radius,s*radius]])
            self.addVehicle(veh)
        
        self.plotLimits = 1.1*radius*np.array([[-1,1],[-1,1]])
        if (len(angles)==2)  and  (max([abs(sin(angle)) for angle in angles]) < 0.1):
            self.plotLimits[1,:] = np.array([[-6,6]])
    
    def get_frog_scenario(self):
        veh1 = DefaultVehicle()
        veh1.x_start = -18
        veh1.referenceTrajectory = np.array([[-100,0], [100,0]]) 
        self.addVehicle( veh1 )

        for o in range(-2,8+1):
            ob = DefaultObstacle()
            ob.x = 7
            ob.y = 9*o-15
            ob.speed = 2
            ob.heading = pi/2
            ob.length = 4
            ob.width = 2
            self.addObstacle(ob)
            ob.x = 14
            self.addObstacle(ob)
        
        self.obstacles = np.array(self.obstacles)
        self.plotLimits = 35*np.array([[-1,1],[-1,1]])
    
    def get_parallel_scenario(self, nVeh):
        _positions = np.array(range(nVeh))-floor(nVeh/2)
        order = list(range(nVeh))
        _ = order[0:nVeh:2]
        _.reverse()
        order = _ + order[1:nVeh:2]
        positions = np.zeros([nVeh])
        positions[order] = _positions
        
        for i in range(nVeh):
            y = 3*positions[i]
            veh = DefaultVehicle()
            veh.x_start = -37
            veh.y_start = y
            veh.labelOffset = np.array([-6.1-4.5*np.mod(positions[i]-1,2),0])
            veh.referenceTrajectory = np.array([[-30,y],[30,y]]) 
            self.addVehicle(veh)
        
        ob0 = DefaultObstacle()
        ob0.length = 2
        ob0.width = 4
        ob0.x = -15
        ob0.y = 5
        self.addObstacle(ob0)

        ob1 = DefaultObstacle()
        ob1.length = 4
        ob1.width = 2
        ob1.x = -2
        ob1.y = -7
        self.addObstacle(ob1)

        ob2 = DefaultObstacle()
        ob2.length = 4
        ob2.width = 2
        ob2.x = 10
        ob2.y = 5
        self.addObstacle(ob2)

        ob3 = DefaultObstacle()
        ob3.length = 2
        ob3.width = 2
        ob3.x = 20
        ob3.y = -7
        self.addObstacle(ob3)

        if nVeh == 2:
            self.CouplingAdjacencyMatrixPB = np.array([[0,1],[0,0]]) > 0
        elif nVeh > 2:
            self.CouplingAdjacencyMatrixPB = np.diag(range(nVeh-1),2) > 0
            self.CouplingAdjacencyMatrixPB[0,1] = True
        
        self.plotLimits = np.array([[-50,50],[-20,20]])
        self.obstacles = np.array(self.obstacles)

    
    def complete_scenario(self):
        # Calculate tick-rates and round_up time constants to multiples of a tick.    
        self.ticks_per_sim = round_up(self.dt / self.tick_length) # Ticks per simulation step.
        self.dt = self.ticks_per_sim * self.tick_length # Make dt a multiple of the tick.
        self.Nsim = round_up(self.T_end / self.dt) # Number of simulation steps
        self.T_end = self.Nsim * self.dt # Make total time a multiple of dt.
        self.ticks_total = int(round_up(self.T_end / self.tick_length)) # Total number of ticks.    
        self.ticks_delay_x = round_up(self.delay_x / self.tick_length) # Ticks of the measurement delay.
        self.delay_x = self.ticks_delay_x * self.tick_length # Make measurement delay a multiple of the tick.
        self.ticks_delay_u = round_up(self.delay_u / self.tick_length) # Ticks of the control delay.
        self.delay_u = self.ticks_delay_u * self.tick_length # Make control delay a multiple of the tick.
        self.nObst =  len(self.obstacles)

        T = np.linspace(0, self.T_end, int(self.ticks_total+1))
        self.calculate_All_Safety_Distances() 
        
        # Fill in coop matrices if they are missing. Priorities are given based on the vehicle index
        if not hasattr(self,'CooperationCoefficientMatrix'):
            alpha = 1
            self.CooperationCoefficientMatrix = alpha*np.ones([self.nVeh,self.nVeh])+ (1-alpha)*np.eye(self.nVeh)
        if not hasattr(self,'CouplingAdjacencyMatrixCoop'):
            self.CouplingAdjacencyMatrixCoop = (np.triu(np.ones([self.nVeh,self.nVeh]),0) == 0).astype(int)
        if not hasattr(self, 'CouplingAdjacencyMatrixPB'):
            self.CouplingAdjacencyMatrixPB = (np.triu(np.ones([self.nVeh,self.nVeh]),0) == 0).astype(int)

    def calculate_All_Safety_Distances(self):
        self.dsafeVehicles = np.zeros([self.nVeh,self.nVeh])
        self.dsafeObstacles = np.zeros([self.nVeh,self.nObst])
        idx = Indices()

        for v in range(self.nVeh):
            for v2 in range(self.nVeh):        
                max_chord_length = (self.x0[v][idx.speed,:] + self.x0[v2][idx.speed,:])*self.dt
                W1 = self.Width[v]/2
                W2 = self.Width[v2]/2
                L1 = self.Length[v]/2
                L2 = self.Length[v2]/2
                R = sqrt(L1**2 + W1**2) + sqrt(L2**2 + W2**2)
                dsafe = sqrt((max_chord_length/2)**2 + R**2)
                self.dsafeVehicles[v,v2] = dsafe
            
            for o in range(self.nObst):
                max_chord_length = (self.x0[v][idx.speed,:] + self.obstacles[o][idx.speed,:])*self.dt
                W1 = self.Width[v]/2;
                W2 = self.obstacles[o][idx.width,:]/2
                L1 = self.Length[v]/2;
                L2 = self.obstacles[o][idx.length]/2;
                R = sqrt(L1**2 + W1**2) + sqrt(L2**2 + W2**2);
                self.dsafeObstacles[v,o] = sqrt((max_chord_length/2)**2 + R**2)

# %%

