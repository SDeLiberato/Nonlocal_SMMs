__all__ = ['Media', 'properties']

import numpy as np
from scipy.constants import speed_of_light


class Media:

    def __init__(self, name, props, wn):
        """ Initiates a material instance with name specified by a string and
        properties by a dict

        Parameters
        ----------

        name : string
            name of the material comprising the layer

        props : dict
            dict containing the material properties

        wn : float or array
            float or array containing the probe frequencies
        """
        self._name = name
        for k, v in props.items():
            setattr(self, '_'+k, v)
        self._wn = wn
        self._eps = self.epsilon_tensor(wn)
        self._mu = self.mu_tensor(wn)
        self._eps_inf = np.diag(
            [self._eps_inf_pe, self._eps_inf_pe, self._eps_inf_pa]
             )
        self._alpha = np.diag(
            [np.sqrt(self._eps_0_pe-self._eps_inf_pe)*self._wto_pe,
             np.sqrt(self._eps_0_pe-self._eps_inf_pe)*self._wto_pe,
             np.sqrt(self._eps_0_pa-self._eps_inf_pa)*self._wto_pa]
            )

    def eps_1_osc(self, wn, orientation='pe'):
        """ Returns the dielectric function evaluated at frequencies wn,
        assuming a 1 oscillator Lorentz model of the dielectric. For materials
        whose phonon frequencies are set to nil returns eps_inf

        Parameters
        ----------

        wn : float or array
            the frequencies to probe

        orientation : string
            either pe or pa, determines whether the function is calculated
            parallel or perpendicular to the crystal axis

        Returns
        -------

        eps : float or array
            dielectric constant evaluated at the input frequencies wn

        """
        if orientation == 'pe':
            wlo, wto, eps_inf, gam = (
                self._wlo_pe, self._wto_pe,
                self._eps_inf_pe, self._gamma
                )
        elif orientation == 'pa':
            wlo, wto, eps_inf, gam = (
                self._wlo_pa, self._wto_pa,
                self._eps_inf_pa, self._gamma
                )

        eps = eps_inf*(wlo**2 - wn*(wn+1j*gam))/(wto**2 - wn*(wn+1j*gam))

        return eps

    def epsilon_tensor(self, wn):
        """ Returns the permittivity tensor evaluated at frequencies wn, currently
        assuming a 1 oscillator Lorentz model of the dielectric and that the
        material is orientated so it's c-axis is parallel to the z-direction.

        Parameters
        ----------

        properties : dict
            contains the material properties to utilise in the calculation

        wn : float or array
            the frequencies to probe

        Returns
        -------

        eps : array
            permittivity tensor evaluated at the input frequencies wn assuming
            the crystal c-axis is parallel to the third dimension

        """

        eps = np.zeros((len(wn), 3, 3), dtype=complex)
        eps[:, 0, 0] = self.eps_1_osc(wn, 'pe')
        eps[:, 1, 1] = self.eps_1_osc(wn, 'pe')
        eps[:, 2, 2] = self.eps_1_osc(wn, 'pa')

        return eps

    def mu_tensor(self, wn):
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
            permeability tensor evaluated at the input frequencies wn assuming
            the crystal c-axis is parallel to the third dimension

        """
        mu = np.zeros((3, 3), dtype=complex)
        mu[0, 0] = 1
        mu[1, 1] = 1
        mu[2, 2] = 1

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
            props['beta_c'] = 1/speed_of_light
            props['beta_l'] = 1/speed_of_light
            props['beta_t'] = 1/speed_of_light
            props['rho'] = 1
            props['eps_inf_pe'] = 1
            props['eps_inf_pa'] = 1
            props['eps_0_pe'] = 1
            props['eps_0_pa'] = 1
            props['wto_pe'] = 0
            props['wto_pa'] = 0
            props['gamma'] = 0
        if mat == 'SiC3C':
            props['beta_l'] = 9e3/speed_of_light
            props['beta_t'] = 2e3/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pa'] = 6.52
            props['eps_inf_pe'] = 6.52
            props['eps_0_pa'] = 9.7
            props['eps_0_pe'] = 9.7
            props['wto_pa'] = 797.5
            props['wto_pe'] = 797.5
            props['gamma'] = 4

    props['wlo_pa'] = (
        props['wto_pa']*np.sqrt(props['eps_0_pa']/props['eps_inf_pa'])
        )
    props['wlo_pe'] = (
        props['wto_pe']*np.sqrt(props['eps_0_pe']/props['eps_inf_pe'])
        )
    return props