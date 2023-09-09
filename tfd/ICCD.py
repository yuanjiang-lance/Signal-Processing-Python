# -*- coding: utf-8 -*-
"""
@Time       : 2023-09-05
@Author     : Yuan JIANG
@Email      : yuanj5@illinois.edu; lance682@qq.com
@Software   : PyCharm
@Description: Intrinsic Chirp Component Decomposition (ICCD)

Related documents:

"""

import sys
import numpy as np
from scipy.integrate import cumtrapz
import scipy.sparse as sp


def ICCD(Sig, Fs, iniIF, orderIA, lbd, orderIF=None):
    """

    Intrinsic Chirp Component Decomposition (ICCD)

    --------------- Parameters ---------------
    Sig: measured signal, 1D Numpy array, real/complex signal
    Fs: sampling frequency (Hz)
    iniIF: initial instantaneous frequencies (IFs), each IF lies in one row
           ATTENTION: The length of iniIF and Sig must be equal
    orderIA: the order of Fourier series covering instantaneous amplitude (IA).
             Higher orderIA results in larger bandwidth
    lbd: Tikhonov regularization parameter.
         Higher lambda results in insensitivity of noise
    orderIF: the order of Fourier series for IF fitting.
             If orderIF is omitted, iniIF should be smoothed IFs with high accuracy (e.g. after fitting or low-pass
             filtering). In this case, the output IFest will equal to iniIF.

    ---------------- Returns -----------------
    Sigest: estimated signal modes, each mode lie in one row
    IFest: estimated IFs, each IF lies in one row
    IAest: estimated instantanous amplitudes (IAs), equivalent to the envelope, each IA lies in one row
    """

    # ------------- Initialization ---------------
    fineIF = True if orderIF is None else False
    if len(iniIF.shape) == 1:
        M = 1
        N = iniIF.shape[0]
    else:
        M, N = iniIF.shape
    if len(Sig) != N:
        print('The length of measured signal and initial IF must be equal!')
        sys.exit(1)
    realSig = True if np.isreal(Sig).all() or np.isrealobj(Sig) else False

    t = np.arange(0, N) / Fs

    if fineIF:
        IFest = iniIF.copy()
        phase = 2 * np.pi * cumtrapz(iniIF, t, initial=0)
    else:
        IFest, phase = IFfit_overFourier(iniIF, Fs, orderIF)


    # ------- Constructing Fourier Matrix ----------
    f0 = Fs / (2 * N)
    K = np.zeros([N, 2*orderIA+1])
    for i in range(orderIA+1):
        K[:, i] = np.cos(2 * np.pi * i * f0 * t)
    for i in range(orderIA+1, 2*orderIA+1):
        K[:, i] = np.sin(2 * np.pi * (i-orderIA) * f0 * t)

    if realSig:
        if M == 1:
            Ci = sp.diags([np.cos(phase)], [0], (N, N), format='csr')
            Si = sp.diags([np.sin(phase)], [0], (N, N), format='csr')
            H = sp.hstack([Ci.dot(K), Si.dot(K)])
        else:
            H = np.zeros([N, (2 * orderIA + 1) * 2 * M])
            for i in range(M):
                Ci = sp.diags([np.cos(phase[i, :])], [0], (N, N), format='csr')
                Si = sp.diags([np.sin(phase[i, :])], [0], (N, N), format='csr')
                H[:, i*2*(2*orderIA+1): (2*i+1)*(2*orderIA+1)] = Ci.dot(K)
                H[:, (2*i+1)*(2*orderIA+1): 2*(i+1)*(2*orderIA+1)] = Si.dot(K)
    else:
        if M == 1:
            Ci = sp.diags([np.exp(1j*phase)], [0], (N, N), format='csr')
            H = Ci.dot(K)
        else:
            H = np.zeros([N, (2 * orderIA + 1) * M])
            for i in range(M):
                Ci = sp.diags([np.exp(1j*phase[i, :])], [0], (N, N), format='csr')
                H[:, i*(2*orderIA+1): (i+1)*(2*orderIA+1)] = Ci.dot(K)

    # ----------- Recovering signal modes -------------
    I = sp.eye(H.shape[1])
    y = np.linalg.solve(H.T.conjugate().dot(H) + lbd*I, H.conjugate().T.dot(Sig))

    Sigest = np.zeros([M, N])
    IAest = np.zeros([M, N])

    for i in range(M):
        if realSig:
            Sigest[i, :] = H[:, i*2*(2*orderIA+1): 2*(i+1)*(2*orderIA+1)].dot(
                y[i*2*(2*orderIA+1): 2*(i+1)*(2*orderIA+1)])
            a = K.dot(y[i*2*(2*orderIA+1): (2*i+1)*(2*orderIA+1)])
            b = K.dot(y[(2*i+1)*(2*orderIA+1): 2*(i+1)*(2*orderIA+1)])
            IAest[i, :] = np.sqrt(a**2 + b**2)
        else:
            Sigest[i, :] = H[:, i*(2*orderIA+1): (i+1)*(2*orderIA+1)].dot(
                y[i*(2*orderIA+1): (i+1)*(2*orderIA+1)])
            IAest[i, :] = K.dot(y[i*(2*orderIA+1): (i+1)*(2*orderIA+1)])
    IAest = np.abs(IAest)

    return Sigest, IFest, IAest
