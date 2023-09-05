# -*- coding: utf-8 -*-
"""
@Time       : 2023-09-01
@Author     : Yuan JIANG
@Email      : yuanj5@illinois.edu; lance682@qq.com
@Software   : PyCharm
@Description: Time-frequency ridge related functions
"""

import numpy as np
from scipy.sparse import diags, eye, linalg


def DFRE(Spec, delta):
    """
    Dual Fast Ridge Estimation algorithm for time-frequency ridge detection
    Only one ridge curve could be extracted under one execution
    Suitable for time-related signals, e.g. harmonics
    For frequency-related (dispersive) signals, it is recommended to use DFRE_F

    ----------- Parameters -----------
    Spec: Time-frequency Spectrum, 2D numpy array
    delta: searching scope

    ------------ Returns -------------
    index: index of frequency bins corresponding to ridge curve
    """

    Spec = np.abs(Spec)
    M, N = Spec.shape
    index = np.zeros(N, dtype=int)
    fmax, tmax = np.unravel_index(np.argmax(Spec), Spec.shape)
    index[tmax] = fmax

    # Extracting right-side ridge points
    f0 = fmax
    for j in range(min(tmax+1, N), N):
        low = max(0, f0-delta)
        up = min(M, f0+delta+1)
        f0 = np.argmax(Spec[low:up, j]) + low
        index[j] = f0

    # Extracting left-side ridge points
    f1 = fmax
    for j in range(max(-1, tmax-1), -1, -1):
        low = max(0, f1-delta)
        up = min(M, f1+delta+1)
        f1 = np.argmax(Spec[low:up, j]) + low
        index[j] = f1

    return index


def DFRE_F(Spec, delta):
    """
    Dual Fast Ridge Estimation algorithm for time-frequency ridge detection
    Only one ridge curve could be extracted under one execution
    Suitable for frequency-related (dispersive) signals, e.g. Lamb waves
    For time-related signals, it is recommended to use DFRE

    ----------- Parameters -----------
    Spec: Time-frequency Spectrum, 2D numpy array
    delta: searching scope

    ------------ Returns -------------
    index: index of frequency bins corresponding to ridge curve
    """

    Spec = np.abs(Spec)
    M, N = Spec.shape
    index = np.zeros(M, dtype=int)
    fmax, tmax = np.unravel_index(np.argmax(Spec), Spec.shape)
    index[fmax] = tmax

    # Extracting ridge points in higher frequency region
    t0 = tmax
    for j in range(min(fmax+1, M), M):
        low = max(0, t0-delta)
        up = min(M, t0+delta+1)
        t0 = np.argmax(Spec[low:up, j]) + low
        index[j] = t0

    # Extracting ridge points in lower frequency region
    t1 = tmax
    for j in range(max(-1, fmax-1), -1, -1):
        low = max(0, t1-delta)
        up = min(M, t1+delta+1)
        t1 = np.argmax(Spec[low:up, j]) + low
        index[j] = t1

    return index


def IFsmooth(IF, mu):
    """
    Curve smooth for instantaneous frequencies (IFs)

    ------------- Parameters -------------
    IF: instantaneous frequencies, each IF must be listed in one row
    mu: smooth degree controlling parameter, smaller beta results in smoother output IF

    -------------- Returns ---------------
    newIF: smoothed IF
    """

    M, N = IF.shape   # M is the number of IFs (components), N is the length of each IF (component)
    e = np.ones(N)
    e2 = -2 * e
    PHI = diags([e, e2, e], [0, 1, 2], (N-2, N), format='csr')  # 2nd-order difference operator
    PHIdoub = PHI.T.dot(PHI)
    newIF = np.zeros_like(IF)
    for i in range(M):
        A = (2/mu * PHIdoub + eye(N)).tocsc()
        newIF[i, :] = linalg.spsolve(A, IF[i, :].T).T

    return newIF