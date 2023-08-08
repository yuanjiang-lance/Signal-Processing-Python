import math
import numpy as np
from scipy.fft import fft
from scipy.signal import hilbert
import matplotlib.pyplot as plt


def myfft(Sig, Fs):
    """
    Fast Fourier Transform of signals (signal-side spectrum)
    -------- Input -------
    Sig: original signals (real or analytical)
         Type: 1 or 2-dimensional Numpy array
         If the input is a 2-dimensional Numpy array, signals must be listed in different columns
    Fs: sampling frequency (Hz)
        Type: int/float

    -------- Output -------
    f: frequency bins
       Type: 1-dimensional Numpy array
    ffty_Sig: the amplitude of FFT results
              Type: 1 or 2-dimensional Numpy array
    fft_Sig: FFT results
             Type: 1 or 2-dimensional Numpy array
    """

    shape = np.shape(Sig)
    N = shape[0]    # length of signals
    f = np.arange(N) * Fs / N
    f = f[:math.floor(N/2)]

    if np.isreal(Sig).all() or np.isrealobj(Sig):
        Sig = hilbert(Sig, axis=0)

    fft_Sig = fft(Sig, axis=0) / N
    if len(shape) == 1:
        fft_Sig = fft_Sig[:math.floor(N/2)]
    else:
        fft_Sig = fft_Sig[:math.floor(N/2), :]
    ffty_Sig = np.abs(fft_Sig)

    return f, ffty_Sig, fft_Sig


def fftshow(f, ffty_Sig):
    """
    Showing frequency spectrum
    -------- Input ---------
    f: frequency bins
    ffty_Sig: FFT result or its amplitude
    """

    if np.iscomplex(ffty_Sig).any() or np.iscomplexobj(ffty_Sig):
        ffty_Sig = np.abs(ffty_Sig)

    plt.figure(figsize=(4.5, 2.5))
    parameters = {
        'font.family': 'Times New Roman',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    }
    plt.rcParams.update(parameters)
    plt.plot(f, ffty_Sig, color='blue', linewidth=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (g)')
    plt.axis([0, math.ceil(f[-1]), 0, 1.1*max(ffty_Sig)])
    plt.tight_layout()
    plt.show()


# A simple demo
if __name__ == '__main__':
    Fs = 500
    T = 10
    t = np.arange(0, T, 1/Fs)
    Sig1 = 2 * np.cos(2*np.pi*100*t + 0.25)
    Sig2 = 0.8 * np.cos(2*np.pi*220*t + 0.4)
    Sig = Sig1 + Sig2

    f, ffty_Sig, fft_Sig = myfft(Sig, Fs)
    fftshow(f, ffty_Sig)
