__all__ = ['epsilon_f', 'mu_f', 'properties']

import numpy as np
from scipy.constants import speed_of_light

def eps_1_osc(wn, wlo,wto,eps_inf,gam):
    """ Returns the dielectric function evaluated at frequencies wn, assuming a
    1 oscillator Lorentz model of the dielectric. For materials whose phonon
    frequencies are set to nil returns eps_inf

    Parameters
    ----------

    wn : float or array
        the frequencies to probe

    wlo : float
        longitudinal optic phonon frequency

    wto : float
        transverse optic phonon frequency

    eps_inf : float
        high-frequency dielectric constant

    gamma : float
        damping rate

    Returns
    -------

    eps : float or array
        dielectric constant evaluated at the input frequencies wn

    """
    eps =  eps_inf*(wlo**2 - wn*(wn+1j*gam))/(wto**2 - wn*(wn+1j*gam))
    return eps

def epsilon_f(properties, wn):
    """ Returns the permittivity tensor evaluated at frequencies wn, currently
    assuming a 1 oscillator Lorentz model of the dielectric and that the material
    is orientated so it's c-axis is parallel to the z-direction. 

    Parameters
    ----------

    properties : dict
        contains the material properties to utilise in the calculation

    wn : float or array
        the frequencies to probe

    Returns
    -------

    eps : array
        permittivity tensor evaluated at the input frequencies wn assuming the
        crystal c-axis is parallel to the third dimension

    """
    wto_pe = properties['wto_pe']
    wto_pa = properties['wto_pa']
    wlo_pe = properties['wlo_pa']
    wlo_pa = properties['wlo_pa']
    eps_inf_pe = properties['eps_inf_pe']
    eps_inf_pa = properties['eps_inf_pa']
    gam = properties['gam']
    eps = np.zeros((len(wn),3,3),dtype=complex)
    eps[:,0,0] = eps_1_osc(wn,wlo_pe,wto_pe,eps_inf_pe,gam)
    eps[:,1,1] = eps_1_osc(wn,wlo_pe,wto_pe,eps_inf_pe,gam)
    eps[:,2,2] = eps_1_osc(wn,wlo_pa,wto_pa,eps_inf_pa,gam)
    
    return eps

def mu_f(properties, wn):
    """ Returns the permeability tensor evaluated at frequencies wn, currently
    assuming non-magnetic media.

    Parameters
    ----------

    properties : dict
        contains the material properties to utilise in the calculation

    wn : float or array
        the frequencies to probe

    Returns
    -------

    mu : array
        permeability tensor evaluated at the input frequencies wn assuming the
        crystal c-axis is parallel to the third dimension

    """
    mu = np.zeros((len(wn),3,3),dtype=complex)
    mu[:,0,0] = 1
    mu[:,1,1] = 1
    mu[:,2,2] = 1
    
    return mu  

def properties(mat):
    """ Returns a dict containing the relevant properties of the material specified
    by the input string

    Parameters
    ----------

    mat : string
        material to analyse

    Returns
    -------

    props : dict
        Contains the relevant parameters corresponding to the string mat
    

    References
    ----------

    """

    mats = ['vac', 'SiC3C']
    if mat not in mats:
        raise TypeError(
            "There is no data corresponding to." + mat)
    else:
        props = dict()
        if mat == 'vac':
            props['betac'] = 1/speed_of_light
            props['betal'] = 1/speed_of_light
            props['betat'] = 1/speed_of_light
            props['rho'] = 1
            props['eps_inf_pe'] = 1
            props['eps_inf_pa'] = 1
            props['eps_0_pe'] = 1
            props['eps_0_pa'] = 1
            props['wto_pe'] = 0
            props['wto_pa'] = 0
            props['gam'] = 0
        if mat == 'SiC3C':
            props['betac'] = 4e3/speed_of_light
            props['betal'] = 9e3/speed_of_light
            props['betat'] = 2e3/speed_of_light
            props['rho'] = 3.21
            props['eps_inf_pa'] = 6.52
            props['eps_inf_pe'] = 6.52
            props['eps_0_pa'] = 9.7
            props['eps_0_pe'] = 9.7
            props['wto_pa'] = 797.5
            props['wto_pe'] = 797.5
            props['gam'] = 4
        
    props['wlo_pa'] = props['wto_pa']*np.sqrt(props['eps_0_pa']/props['eps_inf_pa'])
    props['wlo_pe'] = props['wto_pe']*np.sqrt(props['eps_0_pe']/props['eps_inf_pe'])
    return props