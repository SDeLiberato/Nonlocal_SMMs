__all__ = ['prop_mat']

import numpy as np


def prop_mat(wn, eigs, thickness):
    """ Calculates the propagation matrix for a layer of defined thickness
    given that layer's eigenvalues corresponding to the out-of-plane wavevector

    Parameters
    ----------

    wn : 1d array
        a 1d array, containing the probing wavenumbers

    eigs : array
        a 2d array of dimensions (len(wn) x 10) where 10 is the number of
        modes in the problem. This corresponds to the out of plane wavevector
        divided by the wavenumber

    thickness : float
        the thickness of the layer in metres


    Returns
    -------

    propmat : array
        a matrix of size (len(wn)x10x10) containing the propagation factors for
        each eigenmode
    """

    conv = 2*np.pi/0.01  # Wavevector conversion factor
    diag_vec = np.exp(-1j*eigs*wn[:, None]*conv*thickness)
    prop_mat = np.zeros((len(wn), 10, 10), dtype=complex)

    for idx in range(10):
        prop_mat[:, idx, idx] = diag_vec[:, idx]

    return prop_mat
