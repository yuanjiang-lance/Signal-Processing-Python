# -*- coding: utf-8 -*-
"""
@Time       : 2023-09-01
@Author     : Yuan JIANG
@Email      : yuanj5@illinois.edu; lance682@qq.com
@Software   : PyCharm
@Description: Adaptive Chirp Mode Decomposition

Related documents:

"""

import sys
import numpy as np
from scipy.integrate import cumtrapz
from scipy.sparse import diags, eye, spdiags, linalg, hstack, vstack

'''
def ACMD(Sig, Fs, iniIF, tao, mu, tol, maxit=300):
    """
    Adaptive Chirp Mode Decomposition (ACMD) without bandwidth adaptation

    ----------- Parameters -----------
    Sig: measured signal, 1D Numpy array
    Fs: sampling frequency (Hz)
    iniIF: initial instantaneous frequency (IF), 1D Numpy array
           The length of iniIF and Sig much be equal
    tao: bandwidth controlling parameter, smaller tao results in narrower bandwidth
    mu: IF smooth degree controlling parameter, smaller mu results in smoother IF
    tol: iteration stopping criterion
    maxit: maximum iteration number to avoid a dead loop, default is 300

    ------------ Returns -------------
    IFest: estimated IF, 1D Numpy array
    Sigest: estimated signal mode, 1D Numpy array
    IAest: estimated instantaneous amplitude (IA), equivalent to the envelope, 1D Numpy array
    """

    if len(Sig) != len(IF):

    N = len(Sig)
'''

def ACMD_adapt(Sig, Fs, iniIF, tao0, mu, tol, maxit=300):
    """
    Adaptive Chirp Mode Decomposition (ACMD) with bandwidth adaptation

    ----------- Parameters -----------
    Sig: measured signal, 1D Numpy array
    Fs: sampling frequency (Hz)
    iniIF: initial instantaneous frequency (IF), 1D Numpy array
           The length of iniIF and Sig much be equal
    tao0: initial bandwidth controlling parameter, smaller tao0 results in narrower bandwidth
    mu: IF smooth degree controlling parameter, smaller mu results in smoother IF
    tol: iteration stopping criterion
    maxit: maximum iteration number to avoid a dead loop, default is 300

    ------------ Returns -------------
    IFest: estimated IF, 1D Numpy array
    Sigest: estimated signal mode, 1D Numpy array
    IAest: estimated instantaneous amplitude (IA), equivalent to the envelope, 1D Numpy array
    taorec: recording of tao (bandwidth controlling parameter) in each iteration, 1D Numpy array
    """

    if len(iniIF.shape) > 1:
        print('Initial IF must be an 1D Numpy array.')
        sys.exit(1)
    if len(Sig) != len(iniIF):
        print('The length of measured signal and initial IF must be equal.')
        sys.exit(1)

    N = len(Sig)    # Signal length (must equal the length of iniIF)
    t = np.arange(N) / Fs
    e = np.ones(N)
    e2 = -2 * e

    D = diags([e, e2, e], [0, 1, 2], (N-2, N), format='csr')    # 2nd-order difference operator, matrix D in the original paper
    Ddoub = D.T.dot(D)
    spzeros = diags([np.zeros(N)], [0], (N-2, N), format='csr')
    PHI = vstack([hstack([D, spzeros]), hstack([spzeros, D])])
    PHIdoub = PHI.T.dot(PHI)    # matrix PHI'*PHI

    IFitset = np.zeros((maxit, N))  # estimated IF in each iteration
    Sigitset = np.zeros((maxit, N)) # estimated signal component in each iteration
    taorec = np.zeros(maxit)

    # ------------ Iteration ------------
    it = 0
    sDif = tol + 1  # sDif is the energy difference between two consecutive iterations
    IF = iniIF.copy()
    tao = tao0

    while sDif > tol and it <= maxit:

        cosm = np.cos(2 * np.pi * cumtrapz(IF, t, initial=0))
        sinm = np.sin(2 * np.pi * cumtrapz(IF, t, initial=0))
        Cm = diags([cosm], [0], (N, N), format='csr')
        Sm = diags([sinm], [0], (N, N), format='csr')
        Km = hstack([Cm, Sm])
        Kmdoub = Km.T.dot(Km)

        # Updating demodulated signals
        ym = linalg.spsolve(1/tao*PHIdoub + Kmdoub, Km.T.dot(Sig))
        Sigm = Km.dot(ym)
        Sigitset[it, :] = Sigm

        # IF refinement
        alpham = ym[:N]
        betam = ym[N:]  # two demodulated quadrature signals
        dalpham = np.gradient(alpham, 1/Fs)
        dbetam = np.gradient(betam, 1/Fs)   # derivative of demodulated signals
        dIF = (betam * dalpham - alpham * dbetam) / (2 * np.pi * (alpham ** 2 + betam ** 2))
        dIF = linalg.spsolve(1/mu*Ddoub + eye(N), dIF)
        IF = IF + dIF
        IFitset[it, :] = IF

        # Bandwidth adaptation
        tao = tao * (Sigm.T.dot(Sigm)) / (Sigm.T.dot(Sig))
        taorec[it] = tao

        # Convergence criterion
        if it > 0:
            sDif = (np.linalg.norm(Sigitset[it, :] - Sigitset[it-1, :]) /
                    np.linalg.norm(Sigitset[it-1, :])) ** 2
        it += 1

    it -= 1     # final iteration
    IFest = IFitset[it, :]  # estimated IF
    Sigest = Sigitset[it, :]    # estimated signal component
    IAest = np.sqrt(alpham ** 2 + betam ** 2)   # estimated IA
    taorec = taorec[:it+1]

    return IFest, Sigest, IAest, taorec
