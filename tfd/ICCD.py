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


def ICCD(Sig, Fs, iniIF, orderIA, lbd, orderIF=None):
    """

    Intrinsic Chirp Component Decomposition (ICCD)

    --------------- Parameters ---------------
    Sig:
    Fs:
    iniIF:
    orderIA:
    lbd:
    orderIF:

    ---------------- Returns -----------------
    Sigest:
    IFest:
    IAest:
    """

    # ------------- Initialization ---------------
    fineIF = True if orderIF is None else False
    if len(Sig) != iniIF.shape[1]:
        print('The length of measured signal and initial IF must be equal!')
        sys.exit(1)
    realSig = True if np.isreal(Sig).all() or np.isrealobj(Sig) else False

    M, N = iniIF.shape
    t = np.arange(0, N) / Fs

    if fineIF:
        IFest = iniIF.copy()
        phase = 2 * np.pi * cumtrapz()