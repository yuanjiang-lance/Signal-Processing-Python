# -*- coding: utf-8 -*-
"""
@Time       : 2023-09-02
@Author     : Yuan JIANG
@Email      : yuanj5@illinois.edu; lance682@qq.com
@Software   : PyCharm
@Description: Example 1 for ACMD

Please use plt.show() after all figures are settled.
"""

import numpy as np
import matplotlib.pyplot as plt
from tfd.ACMD import ACMD
from common.signals import sigshow
from common.myfft import myfft, fftshow
from common.mySTFT import mySTFT, stftshow

# --------------- Signal ------------------
Fs = 1000
T = 1
t = np.arange(0, T, 1/Fs)
Sig1 = np.exp(-0.3*t) * np.cos(2*np.pi * (350*t + 1/2/np.pi * np.cos(2*np.pi*25*t)))
IF1 = 350 - 25 * np.sin(50 * np.pi * t)
Sig2 = np.exp(-0.6*t) * np.cos(2*np.pi * (250*t + 1/2/np.pi * np.cos(2*np.pi*20*t)))
IF2 = 250 - 20 * np.sin(40 * np.pi * t)

Sig = Sig1 + Sig2

sigshow(Sig, Fs)
f, fftSpec, _ = myfft(Sig, Fs)
fftshow(f, fftSpec)
Spec, _, _ = mySTFT(Sig, Fs, 512, 32)
stftshow(t, f, Spec)

# ----------- Parameter Setting -------------
tao = 1e-3
mu = 1e-4
tol = 1e-8

# ----------- Component Extraction --------------
iniIF1 = 350 * np.ones(len(Sig))
iniIF2 = 250 * np.ones(len(Sig))
iniIF = np.vstack((iniIF1, iniIF2))

IFest, Sigest, IAest = ACMD(
    Sig, Fs, iniIF, tao, mu, tol)

# ----------- Estimated IF -----------
plt.figure(figsize=(4.15, 3.65))
parameters = {
    'font.family': 'Times New Roman',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
}
plt.rcParams.update(parameters)
plt.plot(t, IF1, t, IF2, color='b', linewidth=2)
plt.plot(t, IFest[0, :], t, IFest[1, :], color='r', linestyle='--', linewidth=2)
plt.xlim(0, T)
plt.ylim(100, Fs/2)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()

# ------------- Reconstructed Modes --------------
plt.figure(figsize=(5, 5))
plt.subplot(2, 1, 1)
plt.plot(t, Sig1, 'k', linewidth=1)
plt.plot(t, Sigest[0, :], 'b--', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('C1')
plt.xlim(0, T)
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.plot(t, Sig2, 'k', linewidth=1)
plt.plot(t, Sigest[1, :], 'b--', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('C2')
plt.xlim(0, T)
plt.tight_layout()

plt.show()