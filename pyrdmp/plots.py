import matplotlib.pyplot as plt
from matplotlib import cm, rc


def phase(s, title='Phase', save=True):
    figure = plt.figure()
    plt.plot(s)
    figure.suptitle(title)
    if save: save_fig(figure, title)
    return figure


def position(t, q, f_q, title='Position', save=True):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(q.shape[1]):
        plt.subplot(q.shape[1], 1, i+1)
        plt.plot(t, q[:, i], 'b')
        plt.plot(t, f_q[:, i], 'r')
    if save: save_fig(figure, title)
    return figure


def velocity(t, dq, f_dq, title='Velocity', save=True):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(dq.shape[1]):
        plt.subplot(dq.shape[1], 1, i+1)
        plt.plot(t, dq[:, i], 'b')
        plt.plot(t, f_dq[:, i], 'r')
    if save: save_fig(figure, title)
    return figure


def acceleration(t, ddq, f_ddq, title='Acceleration', save=True):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(ddq.shape[1]):
        plt.subplot(ddq.shape[1], 1, i+1)
        plt.plot(t, ddq[:, i], 'b')
        plt.plot(t, f_ddq[:, i], 'r')
    if save: save_fig(figure, title)
    return figure


def comparison(t, x, y, z, title='Position Comparison', save=True):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(x.shape[1]):
        plt.subplot(x.shape[1], 1, i+1)
        plt.plot(t, x[:, i], 'b')
        plt.plot(t, y[:, i], 'r')
        plt.plot(t, z[:, i], 'k')
    if save: save_fig(figure, title)
    return figure


def gaussian(s, psv, w, title, save=True):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(w.shape[1]):
        plt.subplot(w.shape[1], 1, i + 1)
        for j in range(psv.shape[0]):
            plt.plot(s, psv[j, :] * w[j, i], color=colors[j % len(colors)])
    if save: save_fig(figure, title)
    return figure


def expected_return(gain, title='Expected Return per episode', save=True):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(len(gain)):
        plt.subplot(len(gain), 1, i + 1)
        plt.plot(range(len(gain[i])), gain[i], i + 1)
    if save: save_fig(figure, title)
    return figure


def show_all():
    plt.show()


def save_fig(fig, title):
    fig.savefig(title.lower().replace(' ', '_') + '.png')
