import numpy as np

'''
    Dynamic Movement Primitive Class
    Author: Michail Theofanidis
'''
class DynamicMovementPrimative:

    '''
        a: Gain a of the transformation system
        b: Gain b of the transformation system
        asDeg: Degredation of the canonical system
        ng: Number of Gaussians
        stb: Stablilization term
    '''
    def __init__(self, _a, _b, _as, _ng, _stb):
       
        self.a = _a
        self.b = _b
        self.asDeg = _as
       
        if _ng < 2:
            raise Exception('Number of gaussians must be greater than 1.')
        self.ng = _ng
        self.stb = _stb

    # Create the phase of the system
    def phase(self, time):
         return np.exp((-self.asDeg) * np.linspace(0, 1.0, len(time))).T

    # Generate Gaussian Distributions
    def distributions(self, s):
        raise Exception('Function not implemented yet');
        '''
        increment = (s[0] - s[len(s) - 1]) / (self.ng - 1);
        center = range(s[len(s) - 1], s[0], increment)
        lrCenter = np.fliplr(center)
        d = np.diff(center)
        c /= d[0]
        '''
   
    # Normalize a vector, returns a numpy array with equal distribution between elements
    @staticmethod
    def normalizeVector(v):
        return np.linspace(0, v[len(v) - 1], len(v))

    # Get the time and joint vectors from the demonstration data
    @staticmethod
    def parseDemo(data):
        #return time vector, joint angles vector
        return data[:, 0], data[:, 1:7]

    # Load in time vector and joint positions for a given demonstration
    @staticmethod
    def loadDemo(fileName):
        return np.loadtxt(fileName, dtype=float, delimiter=',', skiprows=1)

    # Generate a Gaussian using its height, center, and the phase
    @staticmethod
    def psiF(height, center, state):
        return np.exp((-height)*(np.power(state-center, 2)))
