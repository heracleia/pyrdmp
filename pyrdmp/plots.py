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
    def position(q, f_q):
        figure = plt.figure()
        figure.suptitle("Position")
        for i in range(q.shape[1]):
	    plt.subplot(q.shape[1], 1, i)
            plt.plot(q[:, i], 'b')
            plt.plot(f_q[:, i], 'r')
	return figure

    @staticmethod
    def velocity(q, t, dq, f_dq):
        figure = plt.figure()
        figure.suptitle("Velocity")
        for i in range(q.shape[1]):
            plt.subplot(q.shape[1], 1, i)
            plt.plot(t, dq[:, i], 'b')
            plt.plot(t, f_dq[:, i], 'r')
	return figure

    @staticmethod
    def acceleration(q, t, ddq, f_ddq):
        figure = plt.figure()
        figure.suptitle("Acceleration")
        for i in range(q.shape[1]):
            plt.subplot(q.shape[1], 1, i)
            plt.plot(t, ddq[:, i], 'b')
            plt.plot(t, f_ddq[:, i], 'r')

    @staticmethod
    def show_all():
        plt.show()
