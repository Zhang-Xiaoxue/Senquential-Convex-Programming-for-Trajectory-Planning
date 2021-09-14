# %%
import numpy as np

class MIP_CPLEX():
    def __init__(self):
        self.bigM = 1000
        self.R_Gain = 0.1
        self.polygonalNormApproximationDegree = 6
        self.timelimit = 300 # in seconds
        self.obstAsQCQP = 1

class QCQP():
    def __init__(self):
        self.default_dsafeExtra = 0;

        # conversion between distance tolerance and 
        # constraint tolerance: cons_tol = 2 * d_safe * d_tol
        self.constraintTolerance = 2 * 2.1 * 1e-3 # ~1mm is sufficient

class Config():
    def __init__(self):
        self.MIP_CPLEX = MIP_CPLEX()
        self.QCQP = QCQP()
        # this should be a class. But decide it later
        self.Lagrange_Mosek_R_Gain = 10
            
        