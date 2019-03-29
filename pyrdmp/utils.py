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


# Rotation matrix 4x4
def rotate(t, p = np):
    Rz = np.array([[ p.cos(t[0]),-p.sin(t[0]),           0, 0],
                   [ p.sin(t[0]), p.cos(t[0]),           0, 0],
                   [           0,           0,           1, 0],
                   [           0,           0,           0, 1]])
    Ry = np.array([[ p.cos(t[1]),           0, p.sin(t[1]), 0],
                   [           0,           1,           0, 0],
                   [-p.sin(t[1]),           0, p.cos(t[1]), 0],
                   [           0,           0,           0, 1]])
    Rx = np.array([[           1,           0,           0, 0],
                   [           0, p.cos(t[2]),-p.sin(t[2]), 0],
                   [           0, p.sin(t[2]), p.cos(t[2]), 0],
                   [           0,           0,           0, 1]])
    return Rz.dot(Ry).dot(Rx)


# Adapted from: http://enesbot.me/tutorial-forward-kinematics-part-ii.html
def draw_frame(ax, cart):
    """Draws a coordinate reference frame from a 4x4 matrix.
    Adds an extra thick line along the Z axis and around the
    origin to simulate the joint in a robot segment"""

    R = rotate(cart[3:])[0:3,0:3] # rotation matrix
    T = cart[:3]   # extract traslation matrix
    p1 = np.matrix([[0.1],[0],[0]]) # get the 3 other points of the frame
    p2 = np.matrix([[0],[0.1],[0]])
    p3 = np.matrix([[0],[0],[0.1]])
    p1 = R*p1   # Rotate the points
    p2 = R*p2
    p3 = R*p3
    x = T.item(0)   # Get the origin's position of the frame
    y = T.item(1)
    z = T.item(2)
    #draw the line from the origin to each of the points
    ax.plot([x,x+p1[0]],[y,y+p1[1]],[z,z+p1[2]], color='red')
    ax.plot([x,x+p2[0]],[y,y+p2[1]],[z,z+p2[2]], color='green')
    ax.plot([x,x+p3[0]],[y,y+p3[1]],[z,z+p3[2]], color='blue')
    l = 0.02 # extra line for origin
    ax.plot([x-p3.item(0)*l,x+p3.item(0)*l],
            [y-p3.item(1)*l,y+p3.item(1)*l],
            [z-p3.item(2)*l,z+p3.item(2)*l],
            linewidth=2, color = "k") 
