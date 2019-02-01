from dmp import DynamicMovementPrimitive as DMP
import numpy as np
import matplotlib.pyplot as plt
from plots import plot as Plot

# Initialize the DMP class
my_dmp = DMP(20, 20, 0)

# Filtering Parameters
window = 5
blends = 10

# Load the demo data
data = DMP.load_demo("demos/demo1.txt")

# Obtain the joint position data and the time vector
t, q = DMP.parse_demo(data)

# Get the phase from the time vector
phase = my_dmp.phase(t)
Plot.phase(phase)

print(q.shape)

# Normalize the time vector
t = DMP.normalize_vector(t)

# Compute velocity and acceleration
dq = np.zeros(q.shape)
ddq = np.zeros(q.shape)

for i in range(0, q.shape[1]):
	q[:, i] = DMP.smooth(q[:, i], window)
	dq[:, i] = DMP.vel(q[:, i], t)
	ddq[:, i] = DMP.vel(dq[:, i], t)

# Filter the position velocity and acceleration signals
f_q = np.zeros(q.shape)
f_dq = np.zeros(q.shape)
f_ddq = np.zeros(q.shape)

for i in range(0, q.shape[1]):
	f_q[:, i] = DMP.blends(q[:, i], dq[:, i], t, blends)
	f_dq[:, i] = DMP.vel(f_q[:, i], t)
	f_ddq[:, i] = DMP.vel(f_dq[:, i], t)

Plot.position(q, f_q)
Plot.velocity(q, t, dq, f_dq)
Plot.acceleration(q, t, ddq, f_ddq)

Plot.show_all()
'''
# Ploting functions

plt.figure(1)
for i in range(0, q.shape[1]):
	plt.subplot(q.shape[1], 1, i)
	plt.plot(q[:, i], 'b')
	plt.plot(f_q[:, i], 'r')

# Velocity
plt.figure(2)
for i in range(0, q.shape[1]):
	plt.subplot(q.shape[1], 1, i)
	plt.plot(t, dq[:, i], 'b')
	plt.plot(t, f_dq[:, i], 'r')

# Acceleration
plt.figure(3)
for i in range(0, q.shape[1]):
	plt.subplot(q.shape[1], 1, i)
	plt.plot(t, ddq[:, i], 'b')
	plt.plot(t, f_ddq[:, i], 'r')
plt.show()
'''
