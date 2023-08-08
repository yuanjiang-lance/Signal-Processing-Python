import numpy as np


def Differ(y, dx):
    """
    Computing the derivative of a discrete time series y
    --------- Input -----------
    y: time series (e.g. a signal), 1D Numpy array
    dx: sampling time interval of y
    --------- Output ----------
    dy: derivative of y, 1D Numpy array
    """

    L = len(y)
    dy = (y[2:] - y[:-2]) / (2*dx)
    dy = np.insert(dy, 0, (y[1]-y[0])/dx)
    dy = np.insert(dy, len(dy), (y[-1]-y[-2])/dx)

    return dy
