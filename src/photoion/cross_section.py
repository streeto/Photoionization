'''
Created on Apr 10, 2013

@author: andre
'''

import numpy as np
from os import path
from .units import c__AA_s, h__eV_s, Mb__cm2

__ALL__ = ['effCrossSection', 'ionizationEnergy']

_fieldnames=['Z', 'N', 'E_th', 'E_max', 'E_0', 'sigma_0', 'y_a', 'P', 'y_w', 'y_0', 'y_1']

def _loadFitTable(datafile):
    '''
    Load cross section fit table from
    http://www.pa.uky.edu/~verner/photo.html
    See Verner et al 1996, ApJ
    '''
    
    t = np.genfromtxt(datafile, names=_fieldnames)
    return t


def _getAtomFit(atomSpec, datafile=None):
    '''
    Get cross section fit parameters for atoms specified by atomSpec.
    See Verner et al 1996, ApJ
    
    Parameters
    ----------
    atomSpec : string
        String describing the atom. One of:
            * 'H0' : Neutral Hydrogen
            * 'He0' : Neutral Helium
            * 'He1' : Helium one time ionized

    datafile : string
        Path to the fit datafile. Download from
        http://www.pa.uky.edu/~verner/photo.html
        
    Returns
    -------
    p : dict
        Dictionary containing the fit parameters.
        The keys are described in
        ftp://gradj.pa.uky.edu/dima/photo/photo.txt
    '''
    if datafile is None:
        datafile = path.join(path.dirname(__file__), 'data/photo.dat')
        if not path.exists(datafile):
            raise ValueError('The parameter \'datafile\' must be set, or the ' + \
            'file \'photo.dat\' must be in the same directory as the module.')
        
    if atomSpec == 'H0':
        Z, N = 1, 1
    elif atomSpec == 'He0':
        Z, N = 2, 2
    elif atomSpec == 'He1':
        Z, N = 2, 1
    else:
        raise ValueError('Atom %s is unknown' % atomSpec)

    t = _loadFitTable(datafile)
    values = t[(t['Z'] == Z) & (t['N'] == N)][0]
    return dict(zip(_fieldnames, values))


def ionizationEnergy(atomSpec, datafile=None):
    '''
    '''
    p = _getAtomFit(atomSpec, datafile)
    return p['E_th']
    

def effCrossSection(atomSpec, E, E_units='Hz', datafile=None):
    '''
    Effective cross section for photoionization.
    Equation (1) from Verner et al 1996, ApJ
    
    Parameters
    ----------
    atomSpec : string
        String describing the atom. One of:
            * 'H0' : Neutral Hydrogen
            * 'He0' : Neutral Helium
            * 'He1' : Helium one time ionized
            
    E : array
        Energy of inciding photon.
        
    E_units : array
        Units for E
            * 'eV' : energy, electon-volt
            * 'Hz' : frequency, Hertz
            * 'AA' : wavelength, Angstroms
        
    datafile : string, optional
        Path to the fit datafile. Download from
        http://www.pa.uky.edu/~verner/photo.html
        
    Returns
    -------
    a_nu : array
        Effective cross sections, same shape as ``E``.
    '''
    E = np.atleast_1d(E)
    if E_units == 'Hz':
        _E = h__eV_s * E
    elif E_units == 'eV':
        _E = E
    elif E_units == 'AA':
        _E = h__eV_s * c__AA_s / E
    else:
        raise ValueError('Unknown value for nu_unit.')

    p = _getAtomFit(atomSpec, datafile)

    if (_E > p['E_max']).any():
        print 'Warning: energies above E_max'
    
    F = np.zeros_like(E)
    
    # Only calculate the cross section above the threshold energy.
    m = _E > p['E_th']
    x = _E[m] / p['E_0'] - p['y_0']
    y = np.sqrt(x ** 2 + p['y_1'] ** 2)
    F[m] = ((x - 1) ** 2 + p['y_w'] ** 2) * y ** (0.5 * p['P'] - 5.5) * (1 + np.sqrt(y / p['y_a'])) ** (-p['P'])

    if len(E) == 1:
        F = np.asscalar(F)
            
    return p['sigma_0'] * F * Mb__cm2

