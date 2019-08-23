__all__ = ['prop_mat', 'inter_mat']

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


def inter_mat(wn, eigs, mat, fields, ang=None, kx=None):
    """ Calculates the interface matrix given that layer's eigenvalues
    corresponding to the out-of-plane wavevector and eigenvectors

    Parameters
    ----------

    wn : 1d array
        a 1d array, containing the probing wavenumbers

    eigs : array
        a 2d array of dimensions (len(wn) x 10) where 10 is the number of
        modes in the problem. This corresponds to the out of plane wavevector
        divided by the wavenumber

    fields : array
        the array of sorted fields


    Returns
    -------

    inter_mat : array
        a matrix of size (len(wn)x10x10) containing the propagation factors for
        each eigenmode
    """

    zeta = np.zeros_like(wn)
    zeta[:] = np.sin(ang*np.pi/180)

    inter_mat = np.zeros((len(wn), 10, 10), dtype=complex)

    # In-plane electric field (x)
    inter_mat[:, 0, :] = fields[:, :, 3]
    # In-plane electric field (y)
    inter_mat[:, 1, :] = fields[:, :, 4]

    # In-plane magnetic field (x)
    inter_mat[:, 2, :] = fields[:, :, 0]
    # In-plane magnetic field (y)
    inter_mat[:, 3, :] = fields[:, :, 1]

    # In-plane polarisation field (x)
    inter_mat[:, 4, :] = fields[:, :, 6]
    # In-plane polarisation field (y)
    inter_mat[:, 5, :] = fields[:, :, 7]
    #  In-plane polarisation field (y)
    inter_mat[:, 6, :] = fields[:, :, 8]

    # In-plane stress tensor (x)
    inter_mat[:, 7, :] = 0.5*mat._beta_c**2*(
        eigs*fields[:, :, 9]+zeta*fields[:, :, 11]
        )

    # In-plane stress tensor (y)
    inter_mat[:, 8, :] = 0.5*mat._beta_c**2*(eigs*fields[:, :, 10])

    # In-plane stress tensor (z)
    inter_mat[:, 9, :] = (
        0.5*mat._beta_l**2*eigs*fields[:, :, 11]
        + (mat._beta_l**2 - 2*mat._beta_t**2)*(zeta*fields[:, :, 9])
    )

    return inter_mat
