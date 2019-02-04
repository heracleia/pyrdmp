#!/usr/bin/env python3

from pyrdmp.dmp import DynamicMovementPrimitive as DMP
from pyrdmp.plots import *
from pyrdmp.utils import *
import numpy as np

# Initialize the DMP class
my_dmp = DMP(20, 20, 0)

# Filtering Parameters
window = 5
blends = 10

# Load the demo data
data = load_demo("data/demo12.txt")

# Obtain the joint position data and the time vector
t, q = parse_demo(data)

# Get the phase from the time vector
s = my_dmp.phase(t)

# Get the Gaussian
psv = my_dmp.distributions(s)

# Normalize the time vector
t = normalize_vector(t)

# Compute velocity and acceleration
dq = np.zeros(q.shape)
ddq = np.zeros(q.shape)

for i in range(0, q.shape[1]):
	q[:, i] = smooth_trajectory(q[:, i], window)
	dq[:, i] = vel(q[:, i], t)
	ddq[:, i] = vel(dq[:, i], t)

# Filter the position velocity and acceleration signals
f_q = np.zeros(q.shape)
f_dq = np.zeros(q.shape)
f_ddq = np.zeros(q.shape)

for i in range(0, q.shape[1]):
	f_q[:, i] = blend_trajectory(q[:, i], dq[:, i], t, blends)
	f_dq[:, i] = vel(f_q[:, i], t)
	f_ddq[:, i] = vel(f_dq[:, i], t)

# Imitation Learning
ftarget = np.zeros(q.shape)
w = np.zeros((my_dmp.ng, q.shape[1]))

print('Imitation start')

for i in range(0, q.shape[1]):
	ftarget[:, i], w[:, i] = my_dmp.imitate(f_q[:, i], f_dq[:, i], f_ddq[:, i], t, s, psv)

print('Imitation done')

# Generate the Learned trajectory
x = np.zeros(q.shape)
dx = np.zeros(q.shape)
ddx = np.zeros(q.shape)

for i in range(0, q.shape[1]):
	ddx[:, i], dx[:, i], x[:, i] = my_dmp.generate(w[:, i], f_q[0, i], f_q[-1, i], t, s, psv)

# Adapt using Reinforcement Learning
x_r = np.zeros(q.shape)
dx_r = np.zeros(q.shape)
ddx_r = np.zeros(q.shape)

goal = np.array([0, np.pi, 0, np.pi, 0, np.pi, np.pi/3])
samples = 10
rate = 0.5

print('Adaptation start')

for i in range(0, q.shape[1]):
	ddx_r[:, i], dx_r[:, i], x_r[:, i] = my_dmp.adapt(w[:, i], x[0, i], goal[i], t, s, psv, samples, rate)

print('Adaptation complete')

# Plot functions
comparison(t, f_q, x, x_r)
show_all()
