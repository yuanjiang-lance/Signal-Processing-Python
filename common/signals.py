# -*- coding: utf-8 -*-
"""
@Time       : 2023-09-01
@Author     : Yuan JIANG
@Email      : yuanj5@illinois.edu; lance682@qq.com
@Software   : PyCharm
@Description: Signal related functions
"""

import numpy as np
import matplotlib.pyplot as plt

def Segment(Sig, Fs, TStart, TEnd):
    """
    Truncating a segment from a long signal based on time

    ---------- Input -----------
    Sig: original signal, 1D numpy array
    Fs: sampling frequency (Hz)
    TStart: start time of truncation (s)
    TEnd: end time of truncation (s)

    ---------- Output -----------
    Signal segment, 1D numpy array
    """

    t = np.arange(Sig.shape[0]) / Fs
    NStart = np.argmin(np.abs(t - TStart))  # Find index of closest time point to TStart
    N = int(np.ceil((TEnd - TStart) * Fs))  # Calculate the total length of the signal segment
    NEnd = NStart + N

    return Sig[NStart: NEnd]


def sigshow(Sig, Fs):
    """
    Showing one-dimentional signal

    ---------- Input ----------
    Sig: original signal, 1D numpy array
    Fs: sampling frequency (Hz)
    """

    N = Sig.shape[0]
    t = np.arange(N) / Fs

    plt.figure(figsize=(4.15, 2))
    parameters = {
        'font.family': 'Times New Roman',
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    }
    plt.rcParams.update(parameters)
    plt.plot(t, Sig, color='blue', linewidth=0.5)
    plt.xlim(0, t[-1]+0.005)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (AU)')
    plt.tight_layout()
