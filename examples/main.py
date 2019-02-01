from dmp import DynamicMovementPrimitive as DMP
import numpy as np
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
s = my_dmp.phase(t)

# Get the Gaussian
psv = my_dmp.distributions(s)

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


# Plot functions
Plot.position(t, f_q, x)
Plot.show_all()
