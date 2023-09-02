# -*- coding: utf-8 -*-
"""
@Time       : 2023-09-01
@Author     : Yuan JIANG
@Email      : yuanj5@illinois.edu; lance682@qq.com
@Software   : PyCharm
@Description: Derivative of signal
"""

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

    dy = np.zeros_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (2*dx)
    dy[0] = (y[1] - y[0]) / dx
    dy[-1] = (y[-1] - y[-2]) / dx

    return dy
