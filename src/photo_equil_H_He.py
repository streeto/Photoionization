'''
Created on Apr 11, 2013

@author: andre
'''

from photoion import effCrossSection, ionizationEnergy
from photoion.units import c__AA_s, h__erg_s, Rsun__cm, parsec__cm, h__eV_s
import numpy as np
import matplotlib.pyplot as plt
import sys

# if sys.argv[1] == '35':
#     spectrumFile = '../data/k35000-4.dat'
# elif sys.argv[1] == '50':
#     spectrumFile = '../data/k50000-5.dat'
# else:
#     print 'unknown temperature %s' % sys.argv[1]

spectrumFile = '../data/k35000-4.dat'
    
     
# Load spectrum, convert to frequency
# F_nu * d nu = F_lambda * d lambda
# F_nu = c * F_lambda / lambda^2 
spec = np.genfromtxt(spectrumFile, names=['wl', 'flux']) # [AA, erg/cm^2/s/AA]
_spec = spec[::-1] # invert index to account the minus sign in (d nu / d lambda)
nu = c__AA_s / _spec['wl'] # Hz
F_nu = _spec['flux'] * _spec['wl'] ** 2 / c__AA_s # erg/cm^2/s/Hz

# Recombination cross sections
a_nu_H = effCrossSection('H0', nu, E_units='Hz')
a_nu_He = effCrossSection('He0', nu, E_units='Hz')

# Masks to integrate tau_nu
nu_0_H = ionizationEnergy('H0') / h__eV_s
nu_0_He = ionizationEnergy('He0') / h__eV_s
mask_tau_H = nu >= nu_0_H
mask_tau_He = nu >= nu_0_He

# Recombination coefficients
# T = 10000 K
alpha_B_H = 2.59e-13
alpha_B_He = 2.73e-13
alpha_1_He = 1.59e-13
alpha_A_He = alpha_B_He + alpha_1_He

# Helium bound-bound transitions that ionize Hydrogen.
p = 0.75 + 0.25 * (2.0 / 3.0 + 0.56 / 3.0)

# Geometry of the system.
r_star = 18.0 * Rsun__cm
#r_ini = 1.0 * parsec__cm
r_ini = r_star
r_fin = 3000.0 * parsec__cm
npts = 1000
d_r = (r_fin - r_ini) / npts
r = np.linspace(r_ini, r_fin, npts)

# Helium abundance (N_He / N_H)
Y = 0.1

N_H = 100.0 * np.ones_like(r) # cm^-3
N_He = Y * N_H

x_H = np.zeros_like(N_H)
x_He = np.zeros_like(N_H)

# Recombination of Helium to ground state which
# ionizes Hydrogen is dictated by the effective
# cross section at the Helium threshold energy.
a_24eV_H = effCrossSection('H0', 24.6, E_units='eV')
a_24eV_He = effCrossSection('He0', 24.6, E_units='eV')


# Parameters for solving the equations.
ntries = 1000
epsilon = 1e-10
correction = 0.25


def calc_tau_nu_BAD(i_r):
    tau_nu = np.zeros_like(a_nu_H)
    if i_r == 0.0:
        return tau_nu

    if i_r == 1:
        tau_nu[mask_tau_H] += a_nu_H[mask_tau_H] * N_H[0] * x_H[0] * d_r
        tau_nu[mask_tau_He] += a_nu_He[mask_tau_He] * N_He[0] * x_He[0] * d_r
    else:
        N_H1 = N_H[:i_r] * x_H[:i_r]
        N_He1 = N_He[:i_r] * x_He[:i_r]
        tau_nu[mask_tau_H] += a_nu_H[mask_tau_H] * np.trapz(N_H1, r[:i_r])
        tau_nu[mask_tau_He] += a_nu_He[mask_tau_He] * np.trapz(N_He1, r[:i_r])
    return tau_nu


def calc_tau_nu(i_r):
    tau_nu = np.zeros_like(a_nu_H)
    if i_r == 0.0:
        # First cell, return zero.
        return tau_nu

    if i_r == 1:
        # Second cell, not enough to perform a np.trapz.

        # Optical depth caused by Hydrogen.
        tau_nu[mask_tau_H] += a_nu_H[mask_tau_H] * N_H[0] * (1 - x_H[0]) * d_r
        # Optical depth caused by Helium.
        tau_nu[mask_tau_He] += a_nu_He[mask_tau_He] * N_He[0] * (1 - x_He[0]) * d_r
    else:
        # Density of neutral atoms.
        N_H1 = N_H[:i_r] * (1.0 - x_H[:i_r])
        N_He1 = N_He[:i_r] * (1.0 - x_He[:i_r])
        
        # Optical depth caused by Hydrogen.
        tau_nu[mask_tau_H] += a_nu_H[mask_tau_H] * np.trapz(N_H1, r[:i_r])
        # Optical depth caused by Helium.
        tau_nu[mask_tau_He] += a_nu_He[mask_tau_He] * np.trapz(N_He1, r[:i_r])
    return tau_nu


# value of y to use when x is 1.
y0 = a_24eV_H / (a_24eV_H + Y * a_24eV_He)

def calc_y(_x_H, _x_He):
    # Check to avoid division by zero.
    if ((1.0 - _x_H) > epsilon) and ((1.0 - _x_He) > epsilon):
        y = (1.0 - _x_H) * a_24eV_H / ((1.0 - _x_H) * a_24eV_H + Y * (1.0 - _x_He) * a_24eV_He)
    else:
        y = y0
    return y


tau_nu_r = np.zeros((len(r), len(a_nu_H)))

for i_r, _r in enumerate(r):
    if i_r == 0.0:
        # Initial guess.
        _x_H = 0.9
        _x_He = 0.9
        Ne = N_H[0] * (_x_H + Y * _x_He)

    # Calculate the optical depth.
    tau_nu = calc_tau_nu(i_r)
    
    # Save for latter examination.
    tau_nu_r[i_r] = tau_nu
    
    for j in xrange(ntries):
        y = calc_y(_x_H, _x_He)
        
        # Ionization rate for Helium.
        f_ion_He = (r_star / _r) ** 2 * np.pi * np.trapz(F_nu / (h__erg_s * nu) * a_nu_He * np.exp(-tau_nu), nu)
        
        # Fracion of ionization for Helium.
        _x_He = f_ion_He / (f_ion_He + Ne * (alpha_A_He + (y - 1) * alpha_1_He))

        # Ionization rate for Hydrogen.
        f_ion_H = (r_star / _r) ** 2 * np.pi * np.trapz(F_nu / (h__erg_s * nu) * a_nu_H * np.exp(-tau_nu), nu)

        # Fracion of ionization for Hydrogen.
        _x_H = (f_ion_H + _x_He * Y * Ne * (y * alpha_1_He + p * alpha_B_He)) / (f_ion_H + Ne * alpha_B_H)
        
        # Calculate the Electron density.
        new_Ne = N_H[i_r] * _x_H + N_He[i_r] * _x_He
        
        # Adjust, rinse, repeat.
        dif = new_Ne - Ne
        Ne += dif * correction
        if (np.abs(dif)) < epsilon:
            # Save if converged, use the current values as the initial values in next cell.
            x_H[i_r] = _x_H
            x_He[i_r] = _x_He
            break
#     print _r, _x_H, _x_He, Ne, tau_nu[217], tau_nu[299]  
    if j == (ntries - 1):
        print 'Did not converge!', i_r
        sys.exit()


plt.figure(1)
plt.clf()
plt.plot(nu, a_nu_H)
plt.plot(nu, a_nu_He)
plt.ylabel(r'Total $a_\nu\ [cm^2]$')
plt.xlabel(r'$\nu\ [Hz]$')

plt.figure(2)
plt.clf()
plt.plot(r / parsec__cm, x_H, label='Hydrogen')
plt.plot(r / parsec__cm, x_He, label='Helium')
plt.xlabel(r'$r [pc]$')
plt.ylabel(r'$x [fraction]$')
plt.ylim(-0.05, 1.05)
plt.legend(loc='lower left')
