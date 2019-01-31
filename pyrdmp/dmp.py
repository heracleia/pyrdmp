'''
    Dynamic Movement Primitive Class
    Author: Michail Theofanidis, Joe Cloud, James Brady
'''

import numpy as np


class DynamicMovementPrimitive:

    """Create an DMP.
        
        Keyword arguments:
        a -- Gain a of the transformation system
        b -- Gain b of the transformation system
        as_deg -- Degradation of the canonical system
        ng -- Number of Gaussian
        stb -- Stabilization term
    """

    def __init__(self, _a, _ng, _stb):
       
        self.a = _a
        self.b = self.a/4
        self.as_deg = self.a/3
        self.ng = _ng
        self.stb = _stb

    # Create the phase of the system using the time vector
    def phase(self, time):
        return np.exp((-self.as_deg) * np.linspace(0, 1.0, len(time))).T

    # Generate a gaussian distribution
    def distributions(self, s):
        raise Exception('Function not implemented yet')

    # Function that smooths a trajectory
    # a: NumPy 1-D array containing the data to be smoothed.
    # window: smoothing window size needs, which must be odd number.
    @staticmethod
    def smooth(a, window):
        out0 = np.convolve(a, np.ones(window, dtype=int), 'valid')/window
        r = np.arange(1, window-1, 2)
        start = np.cumsum(a[:window-1])[::2]/r
        stop = (np.cumsum(a[:-window:-1])[::2]/r)[::-1]
        return np.concatenate((start, out0, stop))

    # Returns a normalized numpy array
    @staticmethod
    def normalize_vector(v):
        return np.linspace(0, v[len(v) - 1], len(v))

    # Get the time and joint position vectors from the demonstration data
    @staticmethod
    def parse_demo(data):
        return data[:, 0], data[:, 1:8]

    # Load the data contained in the given file name.
    @staticmethod
    def load_demo(fileName):
        return np.loadtxt(fileName, dtype=float, delimiter=',', skiprows=1)

    # Calculate psi (gaussian) based on its height, center, and state
    @staticmethod
    def psv(height, center, state):
        return np.exp((-height)*(np.power(state-center, 2)))

    # Calculate the velocity given the position and time
    @staticmethod
    def vel(q, t):
        dq = np.zeros(len(q))
        for i in range(0, len(q)-1):
            dq[i+1] = (q[i+1]-q[i])/(t[i+1]-t[i])
        return dq

    #  Add parabolic blends to a trajectory
    @staticmethod
    def blends(q, dq, time, blends):

        tj = np.zeros(len(time))
        window = len(time)//blends-1

        up = 0
        down = window

        for i in range(0, blends):

            if i == blends-1:
                pad = len(time)-down-1
                window = window+pad
                down = down+pad

            q_s = q[up]
            q_f = q[down]

            dq_s = dq[up]
            dq_f = dq[down]

            c = DynamicMovementPrimitive.coefficient(q_s, q_f, dq_s, dq_f, time[window])
            dummy = DynamicMovementPrimitive.trajectory(c, time[0:window+1])

            tj[down-window:down+1] = dummy

            up = down+1
            down = down+window

        return tj

    #  Perform polynomial fitting
    @staticmethod
    def coefficient(q_s, q_f, dq_s, dq_f, t):
        alpha = np.zeros(4)

        t_2nd = np.power(t, 2)
        t_3rd = np.power(t, 3)

        alpha[0] = q_s
        alpha[1] = dq_s
        alpha[2] = np.divide(np.multiply(3, q_f-q_s), t_2nd)-np.divide(np.multiply(2, dq_s), t)-np.divide(dq_f, t)
        alpha[3] = np.divide(np.multiply(-2, q_f-q_s), t_3rd)+np.divide(dq_f+dq_s, t_2nd)

        return alpha

    #  Perform polynomial fitting
    @staticmethod
    def trajectory(alpha, t):
        tj = np.zeros(len(t))

        for i in range(0, len(t)):
            t_2nd = np.power(t[i], 2)
            t_3rd = np.power(t[i], 3)
            tj[i] = alpha[0]+np.multiply(alpha[1], t[i])+np.multiply(alpha[2], t_2nd)+np.multiply(alpha[3], t_3rd)

        return tj






