import matplotlib.pyplot as plt
from matplotlib import cm, rc


def phase(s):
    figure = plt.figure()
    plt.plot(s)
    figure.suptitle("Phase")
    return figure


def position(t, q, f_q):
    figure = plt.figure()
    figure.suptitle("Position")
    for i in range(q.shape[1]):
        plt.subplot(q.shape[1], 1, i+1)
        plt.plot(t, q[:, i], 'b')
        plt.plot(t, f_q[:, i], 'r')
    return figure


def velocity(t, dq, f_dq):
    figure = plt.figure()
    figure.suptitle("Velocity")
    for i in range(dq.shape[1]):
        plt.subplot(dq.shape[1], 1, i+1)
        plt.plot(t, dq[:, i], 'b')
        plt.plot(t, f_dq[:, i], 'r')
    return figure


def acceleration(t, ddq, f_ddq):
    figure = plt.figure()
    figure.suptitle("Acceleration")
    for i in range(ddq.shape[1]):
        plt.subplot(ddq.shape[1], 1, i+1)
        plt.plot(t, ddq[:, i], 'b')
        plt.plot(t, f_ddq[:, i], 'r')
    return figure


def comparison(t, x, y, z):
    figure = plt.figure()
    figure.suptitle("Position")
    for i in range(x.shape[1]):
        plt.subplot(x.shape[1], 1, i+1)
        plt.plot(t, x[:, i], 'b')
        plt.plot(t, y[:, i], 'r')
        plt.plot(t, z[:, i], 'k')
    return figure


def gaussian(s, psv, w, title):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(w.shape[1]):
        plt.subplot(w.shape[1], 1, i + 1)
        for j in range(psv.shape[0]):
            plt.plot(s, psv[j, :] * w[j, i], color=colors[j % len(colors)])
    return figure


def show_all():
    plt.show()
