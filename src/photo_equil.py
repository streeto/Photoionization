'''
Created on Apr 11, 2013

@author: andre
'''

from photoion import effCrossSection
from photoion.units import c__AA_s, h__erg_s, Rsun__cm, parsec__cm
import numpy as np
import matplotlib.pyplot as plt

# Load spectrum, convert to frequency
# F_nu * d nu = F_lambda * d lambda
# F_nu = c * F_lambda / lambda^2 
spectrumFile = '../data/k35000-4.dat'
spec = np.genfromtxt(spectrumFile, names=['wl', 'flux']) # [AA, erg/cm^2/s/AA]
_spec = spec[::-1] # invert index to account the minus sign in (d nu / d lambda)
nu = c__AA_s / _spec['wl'] # Hz
F_nu = _spec['flux'] * c__AA_s / _spec['wl'] ** 2 # erg/cm^2/s/Hz

a_nu = effCrossSection('H0', nu, E_units='Hz')

# T = 20000 K
alpha_B = 2.59e-13

r_star = 10.0 * Rsun__cm
r = np.linspace(r_star, 50.0 * parsec__cm, 1000)
N_H = 10.0 * np.ones_like(r) # cm^-3
x = np.zeros_like(r)

for i_r, _r in enumerate(r):
    if i_r == 0.0:
        tau_nu = 0.0
    else:
#         tau_nu = N_H * a_nu * np.trapz((1 - x[:i_r - 1]), r[:i_r - 1])
        tau_nu = a_nu * np.trapz(N_H[:i_r - 1], r[:i_r - 1])
    f_ion = np.trapz(np.pi * F_nu / (h__erg_s * nu) * a_nu * np.exp(-tau_nu), nu)
    if f_ion == 0.0: print i_r
    _x = 2.0 / (1.0 + np.sqrt(1.0 + 4.0 * N_H[i_r] * alpha_B / f_ion))
    x[i_r] = _x

plt.ioff()
plt.figure(1)
plt.clf()
plt.plot(nu, a_nu)
plt.ylabel(r'$a_\nu\ [cm^2]$')
plt.xlabel(r'$\nu\ [Hz]$')

plt.figure(2)
plt.clf()
plt.plot(r / parsec__cm, x)
plt.xlabel(r'$r [pc]$')
plt.ylabel(r'$x [fraction]$')

plt.show()
