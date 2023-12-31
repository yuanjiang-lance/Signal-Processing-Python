# -*- coding: utf-8 -*-
"""
@Time       : 2023-09-01
@Author     : Yuan JIANG
@Email      : yuanj5@illinois.edu; lance682@qq.com
@Software   : PyCharm
@Description: Short-time Frequency Transform (STFT) and its plot
"""

import math
import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal import hilbert
import matplotlib.pyplot as plt


def mySTFT(Sig, Fs, Nfbin, WinLen):
    """
    Calculating the Short Time Fourier Transform
    ---------- Input -----------
    Sig: the signal to be analyzed, 1D Numpy array
    Fs: sampling frequency (Hz)
    Nfbin: the number of frequency bins
    WinLen: the window length to locate signal in time domain
    ---------- Output ----------
    Spec: the STFT spectrum, 2D Numpy array
    f: frequency bins, 1D Numpy array
    t: time bins, 1D Numpy array
    """
    if np.isreal(Sig).all() or np.isrealobj(Sig):
        Sig = hilbert(Sig)

    SigLen = Sig.shape[0]
    N = 2 * Nfbin - 1

    WinLen = math.ceil(WinLen/2) * 2    # ceiling the window length into a even number
    t = np.linspace(-1, 1, num=WinLen)
    sigma = 0.28
    WinFun = (np.pi * sigma**2)**(-1/4) * np.exp((-t**2) / 2 / (sigma**2))

    Lh = (WinLen - 1) // 2   # half of the window length
    Spec = np.zeros([N, SigLen], dtype=np.complex128)

    for iLoop in range(SigLen):
        tau = np.arange(-min(round(N/2)-1, Lh, iLoop), min(round(N/2)-1, Lh, SigLen-iLoop-1))
        temp = np.floor(iLoop + tau).astype(int)
        temp1 = np.floor(Lh + 1 + tau).astype(int)
        rSig = Sig[temp]

        rSig = rSig * np.conj(WinFun[temp1])
        Spec[:rSig.shape[0], iLoop] = rSig

    Spec = fftshift(fft(Spec, axis=0), axes=0)

    f = np.linspace(-Fs/2, Fs/2, N)
    t = np.arange(SigLen) / Fs
    Spec = Spec[-Nfbin:, :]
    f = f[-Nfbin:]

    return Spec, f, t


def stftshow(t, f, Spec):
    """
    Showing time-frequency representation (TFR), such as STFT
    ----------- Input ----------
    t: time bins, 1D Numpy array
    f: frequency bins, 1D Numpy array
    Spec: time-frequency spectrum, 2D Numpy array
    """
    plt.figure(figsize=(4.14, 3.64))
    parameters = {
        'font.family': 'Times New Roman',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    }
    plt.rcParams.update(parameters)
    plt.imshow(np.abs(Spec),
               extent=(t[0], t[-1]+0.01, 0, f[-1]+1),
               aspect='auto',
               origin='lower',
               cmap='jet')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()


# One simple demo
if __name__ == '__main__':
    Fs = 500
    T = 10
    t = np.arange(0, T, 1/Fs)
    Sig1 = 2 * np.cos(2*np.pi*100*t + 0.25)
    Sig2 = 0.8 * np.cos(2*np.pi*220*t + 0.4)
    Sig = Sig1 + Sig2

    Spec, f, t = mySTFT(Sig, Fs, 1024, 512)
    stftshow(t, f, Spec)