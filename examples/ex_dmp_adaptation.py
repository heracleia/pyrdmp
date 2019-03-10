#!/usr/bin/env python3

import argparse
from pyrdmp.dmp import DynamicMovementPrimitive as DMP
import pyrdmp.plots as plot
from pyrdmp.utils import *
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Use Reinforced DMP to adapt to new goals")
    parser.add_argument('-ga', '--gain', type=float, default=20.0,
                        help="Set the gain of the DMP transformation system.")
    parser.add_argument('-ng', '--num-gaussians', type=int, default=20,
                        help="Number of Gaussians")
    parser.add_argument('-sb', '--stabilization', type=bool, default=False,
                        help="Add a stability term at end of trajectory")
    parser.add_argument('-if', '--input-file', type=str, default='data/demo12.txt',
                        help="Input trajectory file")
    parser.add_argument('-of', '--output-file', type=str, default='output.txt',
                        help="Output plot file")
    parser.add_argument('-p', '--show-plots', dest='show_plots', action='store_true',
                        help="Show plots at end of computation")
    parser.add_argument('-np', '--no-plots', dest='show_plots', action='store_false',
                        help="Don't show plots at end of computation")
    parser.add_argument('-w', '--window', type=int, default=5,
                        help="Window size for filtering")
    parser.add_argument('-b', '--blends', type=int, default=10,
                        help="Number of blends for filtering")
    parser.add_argument('-s', '--samples', type=int, default=10,
                        help="Number of paths for exploration")
    parser.add_argument('-r', '--rate', type=float, default=0.5,
                        help="Number of possible paths to keep")
    parser.add_argument('-g', '--goal', nargs='+', type=float, 
                        default=[-2.7, 3.4, 0.6, -0.3, 1.8, -2.7, -1.35],
                        help="New position goal (joint space)")
    parser.set_defaults(show_plots=True)
    arg = parser.parse_args()

    # Initialize the DMP class
    my_dmp = DMP(arg.gain, arg.num_gaussians, arg.stabilization)

    # Load the demo data
    data = load_demo(arg.input_file)

    # Obtain the joint position data and the time vector
    t, q = parse_demo(data)

    # Get the phase from the time vector
    s = my_dmp.phase(t)

    # Get the Gaussian
    psv = my_dmp.distributions(s)

    # Normalize the time vector
    t = normalize_vector(t)

    # Compute velocity and acceleration
    dq, ddq = np.zeros((2, q.shape[0], q.shape[1]))

    for i in range(q.shape[1]):
        q[:, i] = smooth_trajectory(q[:, i], arg.window)
        dq[:, i] = vel(q[:, i], t)
        ddq[:, i] = vel(dq[:, i], t)

    # Filter the position velocity and acceleration signals
    f_q, f_dq, f_ddq = np.zeros((3, q.shape[0], q.shape[1]))

    for i in range(q.shape[1]):
        f_q[:, i] = blend_trajectory(q[:, i], dq[:, i], t, arg.blends)
        f_dq[:, i] = vel(f_q[:, i], t)
        f_ddq[:, i] = vel(f_dq[:, i], t)

    # Imitation Learning
    ftarget = np.zeros(q.shape)
    w = np.zeros((my_dmp.ng, q.shape[1]))

    print('Imitation start')

    for i in range(q.shape[1]):
        ftarget[:, i], w[:, i] = my_dmp.imitate(f_q[:, i], f_dq[:, i], 
                f_ddq[:, i], t, s, psv)

    print('Imitation done')

    # Generate the Learned trajectory
    x, dx, ddx = np.zeros((3, q.shape[0], q.shape[1]))

    for i in range(q.shape[1]):
        ddx[:, i], dx[:, i], x[:, i] = my_dmp.generate(w[:, i], f_q[0, i], 
                f_q[-1, i], t, s, psv)

    # Adapt using Reinforcement Learning
    print('Adaptation start')

    x_r, dx_r, ddx_r = np.zeros((3, q.shape[0], q.shape[1]))
    w_a = np.zeros((my_dmp.ng, q.shape[1]))
    gain = []

     
    for i in range(q.shape[1]):
        ddx_r[:, i], dx_r[:, i], x_r[:, i], w_a[:, i], g = my_dmp.adapt(w[:, i], 
                x[0, i], arg.goal[i], t, s, psv, arg.samples, arg.rate)
        gain.append(g)
    
    print('Adaptation complete')

    # Plot functions
    if arg.show_plots:
        plot.comparison(t, None, x, x_r)
        plot.gaussian(s, psv, w, "Initial Gaussian")
        plot.gaussian(s, psv, w_a, "Altered Gaussian")
        plot.expected_return(gain)
        plot.show_all()


if __name__ == '__main__':
    main()

