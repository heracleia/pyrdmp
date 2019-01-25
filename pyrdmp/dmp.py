'''
    Dynamic Movement Primitive Class
    Author: Michail Theofanidis
'''

import numpy as np

class DynamicMovementPrimative:
    """Create an DMP.
        
        Keyword arguments:
        a -- Gain a of the transformation system
        b -- Gain b of the transformation system
        as_deg -- Degredation of the canonical system
        ng -- Number of Gaussians
        stb -- Stablilization term
    """

    def __init__(self, _a, _b, _as, _ng, _stb):
       
        self.a = _a
        self.b = _b
        self.as_deg = _as
       
        if _ng < 2:
            raise Exception('Number of gaussians must be greater than 1.')
        self.ng = _ng
        self.stb = _stb

    def phase(self, time):
        """Create the phase of the system using the time vector."""

         return np.exp((-self.as_deg) * np.linspace(0, 1.0, len(time))).T

    def distributions(self, s):
        """Generate the gaussian distributions."""
        raise Exception('Function not implemented yet')
   
    @staticmethod
    def normalize_vector(v):
        """Returns a normalized numpy array"""
        return np.linspace(0, v[len(v) - 1], len(v))

    # Get the time and joint vectors from the demonstration data
    @staticmethod
    def parse_demo(data):
        """Return the time vector and the joint angles vector"""
        return data[:, 0], data[:, 1:7]

    @staticmethod
    def load_demo(fileName):
        """Load the data contained in the given file name."""
        return np.loadtxt(fileName, dtype=float, delimiter=',', skiprows=1)

    # Generate a Gaussian using its height, center, and the phase
    @staticmethod
    def psif(height, center, state):
        """Calculate psi (gaussian) based on its height, center, and state"""
        return np.exp((-height)*(np.power(state-center, 2)))
