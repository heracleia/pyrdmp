import matplotlib.pyplot as plt


plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params) 

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

def comparison(t, x, y, z):
    figure = plt.figure()
    figure.suptitle("Position")
    for i in range(x.shape[1]):
        plt.subplot(x.shape[1], 1, i+1)
        plt.plot(t, x[:, i], 'b')
        plt.plot(t, y[:, i], 'r')
        plt.plot(t, z[:, i], 'k')
    return figure

def show_all():
    plt.show()
