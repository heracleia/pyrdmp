import matplotlib.pyplot as plt

class plot:

    def __init__():
        pass

    @staticmethod
    def phase(s):
        figure = plt.figure()
        plt.plot(s)
        figure.suptitle("Phase")
        return figure

    @staticmethod
    def position(t, q, f_q):
        figure = plt.figure()
        figure.suptitle("Position")
        for i in range(q.shape[1]):
            plt.subplot(q.shape[1], 1, i+1)
            plt.plot(t, q[:, i], 'b')
            plt.plot(t, f_q[:, i], 'r')
        return figure

    @staticmethod
    def velocity(t, dq, f_dq):
        figure = plt.figure()
        figure.suptitle("Velocity")
        for i in range(dq.shape[1]):
            plt.subplot(dq.shape[1], 1, i+1)
            plt.plot(t, dq[:, i], 'b')
            plt.plot(t, f_dq[:, i], 'r')
        return figure

    @staticmethod
    def acceleration(t, ddq, f_ddq):
        figure = plt.figure()
        figure.suptitle("Acceleration")
        for i in range(ddq.shape[1]):
            plt.subplot(ddq.shape[1], 1, i+1)
            plt.plot(t, ddq[:, i], 'b')
            plt.plot(t, f_ddq[:, i], 'r')

    @staticmethod
    def comparison(t, x, y, z):
        figure = plt.figure()
        figure.suptitle("Position")
        for i in range(x.shape[1]):
            plt.subplot(x.shape[1], 1, i+1)
            plt.plot(t, x[:, i], 'b')
            plt.plot(t, y[:, i], 'r')
            plt.plot(t, z[:, i], 'k')
        return figure


    @staticmethod
    def show_all():
        plt.show()
