import matplotlib.pyplot as plt
from matplotlib import cm, rc


rc('text', usetex=True)
FONTSIZE=16


def phase(s, title='Phase', directory='', save=True):
    figure = plt.figure()
    plt.plot(s)
    figure.suptitle(title)
    if save: save_fig(figure, title, directory)
    return figure


def position(t, q, f_q, title='Position', directory='', save=True):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(q.shape[1]):
        plt.subplot(q.shape[1], 1, i+1)
        plt.plot(t, q[:, i], 'b')
        plt.plot(t, f_q[:, i], 'r')
        plt.ylabel("$q_%s$" % str(i + 1), fontsize=FONTSIZE)
    if save: save_fig(figure, title, directory)
    return figure


def velocity(t, dq, f_dq, title='Velocity', directory='', save=True):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(dq.shape[1]):
        plt.subplot(dq.shape[1], 1, i+1)
        plt.plot(t, dq[:, i], 'b')
        plt.plot(t, f_dq[:, i], 'r')
        plt.ylabel("$q_%s$" % str(i + 1), fontsize=FONTSIZE)
    if save: save_fig(figure, title, directory)
    return figure


def acceleration(t, ddq, f_ddq, title='Acceleration', directory='', save=True):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(ddq.shape[1]):
        plt.subplot(ddq.shape[1], 1, i+1)
        plt.plot(t, ddq[:, i], 'b')
        plt.plot(t, f_ddq[:, i], 'r')
    if save: save_fig(figure, title, directory)
    return figure


def comparison(t, x=None, y=None, z=None, labels=['NN','DMP','RL'], title='Position Comparison', directory='', save=True):
    figure = plt.figure()
    figure.suptitle(title)
    shp = max([v.shape[1] for v in [x,y,z] if v is not None])
    for i in range(shp):
        plt.subplot(shp, 1, i+1)
        if x is not None: plt.plot(t, x[:, i], 'b', label=labels[0])
        if y is not None: plt.plot(t, y[:, i], 'r', label=labels[1])
        if z is not None: plt.plot(t, z[:, i], 'k', label=labels[2])
        plt.ylabel("${q_%s}^c$" % str(i + 1), fontsize=FONTSIZE)
        if i == 0: plt.legend(loc="upper right")
    figure.align_ylabels()
    plt.xlabel("time (s)", fontsize=FONTSIZE)
    if save: save_fig(figure, title, directory)
    return figure


def gaussian(s, psv, w, title, directory='', save=True):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(w.shape[1]):
        plt.subplot(w.shape[1], 1, i + 1)
        for j in range(psv.shape[0]):
            plt.plot(s, psv[j, :] * w[j, i], color=colors[j % len(colors)])
    if save: save_fig(figure, title, directory)
    return figure


def expected_return(gain, title='Expected Return per Episode', directory='', save=True):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(len(gain)):
        plt.subplot(len(gain), 1, i + 1)
        plt.plot(range(len(gain[i])), gain[i], i + 1)
    if save: save_fig(figure, title, directory)
    return figure


def show_all():
    plt.show()


def save_fig(fig, title, directory):
    fig.savefig(directory + '_' + title.lower().replace(' ', '_') + '.png')
