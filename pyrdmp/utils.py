import numpy as np


# Function that smooths a trajectory
# a: NumPy 1-D array containing the data to be smoothed.
# window: smoothing window size needs, which must be odd number.
def smooth_trajectory(a, window):
    out0 = np.convolve(a, np.ones(window, dtype=int), 'valid')/window
    r = np.arange(1, window-1, 2)
    start = np.cumsum(a[:window-1])[::2]/r
    stop = (np.cumsum(a[:-window:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


# Returns a normalized numpy array
def normalize_vector(v):
    return np.linspace(0, v[len(v) - 1], len(v))


# Get the time and joint position vectors from the demonstration data
def parse_demo(data):
    return data[:, 0], data[:, 1:8]


# Load the data contained in the given file name.
def load_demo(filename):
    return np.loadtxt(filename, dtype=float, delimiter=',', skiprows=1)


# Calculate psi (gaussian) based on its height, center, and state
def psi(height, center, state):
    return np.exp((-height)*(np.power(state-center, 2)))


# Calculate the velocity given the position and time
def vel(q, t):
    dq = np.zeros(len(q))
    for i in range(0, len(q)-1):
        dq[i+1] = (q[i+1]-q[i])/(t[i+1]-t[i])
    return dq


#  Add parabolic blends to a trajectory
def blend_trajectory(q, dq, time, blends):

    tj = np.zeros(len(time))
    window = len(time)//blends-1

    up = 0
    down = window

    for i in range(0, blends):

        if i == blends-1:
            pad = len(time) - down - 1
            window = window+pad
            down = down+pad

        c = coefficient(q[up], q[down], dq[up], dq[down], time[window-1])
        tj[down-window : down+1] = trajectory(c, time[0 : window+1])

        up = down + 1
        down += window

    return tj


#  Perform polynomial fitting
def coefficient(q_s, q_f, dq_s, dq_f, t):
    alpha = np.zeros(4)

    t_2nd = np.power(t, 2)
    t_3rd = np.power(t, 3)

    alpha[0] = q_s
    alpha[1] = dq_s
    alpha[2] = 3*(q_f - q_s)/t_2nd - 2*dq_s/t - dq_f/t
    alpha[3] = -2*(q_f - q_s)/t_3rd + (dq_f+dq_s)/t_2nd

    return alpha


#  Perform polynomial fitting
def trajectory(alpha, t):
    tj = np.zeros(len(t))

    for i in range(0, len(t)):
        t_2nd = np.power(t[i], 2)
        t_3rd = np.power(t[i], 3)
        tj[i] = alpha[0] + alpha[1]*t[i] + alpha[2]*t_2nd + alpha[3]*t_3rd

    return tj

