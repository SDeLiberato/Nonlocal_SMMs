__all__ = ['layer_dispersion', 'field_gen']

import numpy as np
from itertools import product
from materials.materials import Media, properties


def layer_dispersion(wn, material, ang=None, kx=None):
    """ Calculates the dispersion of the photonic and longitudinal and transverse
    phonon eigenmodes of an anisotropic medium, describable by a diagonal
    permeability and permitivitty

    Parameters
    ----------

    kx : float or 1d array
        contains a range of in-plane wavevectors to probe over

    wn : float or 1d array
        contains a range of frequencies to probe over

    material : string
        material to analyse, corresponds to an entry in ...

    ang : float
        optional, corresponds to the incident angle. useage overrides entry
        of kx

    Returns
    -------

    eigs : array
        Values representing calculated out-of-plane wavevectors of the
        eigenmodes of the layer.

    vecs : array
        Values representing the calculated field components corresponding to
        the eigenmodes of the layer. These are in order Ex, Ey, Xx, Xy, Xz


    References
    ----------

    The formation of the matrix from which the dispersions are calculated is
    detailed at " "

    To-do
    _____

    Currently returning TE and TM polarizations with degenerate frequencies
    but mixed field polarizations. This isn't great, can we sort in some way?
    Maybe look at how Nikolai sifts for polarization?

    """

    # Checks for non-array inputs and converts to numpy arrays
    if kx and type(kx) != np.ndarray:
        kx = np.array(kx)
    if type(wn) != np.ndarray:
        wn = np.array(wn)

    # Checks for an input angle, if present calculates zeta using this else
    # uses the input wavevector array
    if ang:
        kx = np.zeros(1)  # Creates a dummy kx to ease code maintanence below
        arrays = [(wn), (kx)]
        arr = np.array(list(product(*arrays)))
        zeta = np.zeros_like(wn)
        zeta[:] = np.sin(ang*np.pi/180)
    else:
        arrays = [(wn), (kx)]
        arr = np.array(list(product(*arrays)))
        zeta = arr[:, 1]/arr[:, 0]

    # Fetches material properties
    props = properties(material)

    layer = Media(material, props, wn)

    # Creates an array of len(zeta) 5 x 5 identity matrices
    It = np.identity(5)
    It = np.swapaxes(np.broadcast_to(It[..., None], (5, 5, len(zeta))), 0, 2)
    # Creates an array of len(zeta) 5 x 5 null matrices
    Zt = np.zeros_like(It)

    tAt = np.zeros_like(It, dtype=complex)
    tBt = np.zeros_like(It, dtype=complex)

    tAt[:, 0, 0] = layer._eps_inf[0, 0]*(-zeta**2/layer._eps_inf[2, 2]+layer._mu[1, 1])
    tAt[:, 0, 2] = layer._alpha[0, 0]*(-zeta**2/layer._eps_inf[2, 2]+layer._mu[1, 1])

    tAt[:, 1, 1] = layer._mu[0, 0]*(-zeta**2/layer._mu[2, 2]+layer._eps_inf[1, 1])
    tAt[:, 1, 3] = layer._alpha[1, 1]*layer._mu[0, 0]

    tAt[:, 2, 0] = -2*layer._alpha[0, 0]/arr[:, 0]**2/layer._beta_c**2
    tAt[:, 2, 2] = 2*(layer._wto_pe**2
                      - arr[:, 0]*(arr[:, 0]+1j*layer._gamma)
                      - layer._beta_l**2*zeta**2*arr[:, 0]**2
                      )/arr[:, 0]**2/layer._beta_c**2
    tAt[:, 3, 1] = -2*layer._alpha[1, 1]/arr[:, 0]**2/layer._beta_c**2
    tAt[:, 3, 3] = 2*(layer._wto_pe**2
                      - arr[:, 0]*(arr[:, 0]+1j*layer._gamma)
                      - 0.5*layer._beta_c**2*zeta**2*arr[:, 0]**2
                      )/arr[:, 0]**2/layer._beta_c**2

    tAt[:, 4, 4] = ( (layer._wto_pa**2
                       - arr[:, 0]*(arr[:, 0]+1j*layer._gamma)
                       - 0.5*layer._beta_c**2*zeta**2*arr[:, 0]**2
                       )/arr[:, 0]**2/layer._beta_l**2
                    - (layer._alpha[2, 2]**2*layer._mu[1, 1]
                        /arr[:, 0]**2/layer._beta_l**2
                        /(zeta**2 - layer._eps_inf[2, 2]*layer._mu[1, 1])
                        )
                    )

    tBt[:, 0, 4] = -zeta*layer._alpha[2, 2] / layer._eps_inf[2, 2]

    tBt[:, 2, 4] = -2*zeta*(
                            layer._beta_c**2/2
                            + layer._beta_l**2
                            - 2*layer._beta_t**2
                            ) / layer._beta_c**2

    tBt[:, 4, 0] = - layer._alpha[2, 2]*zeta/(layer._beta_l**2*arr[:, 0]**2
                                              *(zeta**2-layer._eps_inf[2, 2]*layer._mu[1, 1])
                                              )

    tBt[:, 4, 2] = -1*zeta*(
                            layer._beta_c**2/2
                            + layer._beta_l**2
                            - 2*layer._beta_t**2
                            ) / layer._beta_l**2

    M3 = -np.concatenate((
        np.hstack((Zt, It)),
        np.hstack((tAt, tBt))
        ), axis=2)

    eigs, vecs = np.linalg.eig(M3)

    # Reshape into a useable array
    eigs = eigs.reshape((len(kx), len(wn), 10), order='F')
    vecs = vecs.reshape((len(kx), len(wn), 10, 10), order='F')

    # Initial sort
    a = -1
    order = eigs.argsort(axis=a)
    m, n, k = eigs.shape
    idx = np.ogrid[:m, :n, :k]
    idx[-1] = order
    eigs_r = eigs[tuple(idx)]

    m, n, k, o = vecs.shape
    idx = np.ogrid[:m, :n, :k, :o]
    order = np.repeat(order[:, :, np.newaxis, :], 10, axis=2)
    idx[-1] = order
    vecs_r = vecs[tuple(idx)]

    eigs_ph, vecs_ph = pol_comp(eigs_r[:, :, 5:7], vecs_r[:, :, :, 5:7])
    eigs_to, vecs_to = pol_comp(eigs_r[:, :, 7:9], vecs_r[:, :, :, 7:9])

    sorted_eigs = np.concatenate((np.concatenate(
        (eigs_ph, eigs_to), axis=2
        ), eigs_r[:, :, -1:]), axis=2
    )

    sorted_vecs = np.concatenate((np.concatenate(
        (vecs_ph, vecs_to), axis=3
        ), vecs_r[:, :, :, -1:]), axis=3
    )

    # Reverse propagation
    eigs_ph_r, vecs_ph_r = pol_comp(eigs_r[:, :, 3:5], vecs_r[:, :, :, 3:5])
    eigs_to_r, vecs_to_r = pol_comp(eigs_r[:, :, 1:3], vecs_r[:, :, :, 1:3])

    sorted_eigs_r = np.concatenate((np.concatenate(
        (eigs_ph_r, eigs_to_r), axis=2
        ), eigs_r[:, :, :1]), axis=2
    )

    sorted_vecs_r = np.concatenate((np.concatenate(
        (vecs_ph_r, vecs_to_r), axis=3
        ), vecs_r[:, :, :, :1]), axis=3
    )

    if ang:
        # return sorted_eigs[0], sorted_vecs[0], sorted_eigs_r[0], sorted_vecs_r[0]
        eigs_out = np.concatenate([sorted_eigs[0], sorted_eigs_r[0]], axis=1)
        vecs_out = np.concatenate([sorted_vecs[0], sorted_vecs_r[0]], axis=2)
        return eigs_out, vecs_out
    else:
        return sorted_eigs, sorted_vecs

def pol_comp(eigs, vecs):
    """ Function to compare the eigenmode polarisation state
    dot dot dot
    """
    fom = (
        np.abs(vecs[:, :, 0])**2
        / (np.abs(vecs[:, :, 0])**2+np.abs(vecs[:, :, 1])**2)
        )

    a = -1
    order = fom.argsort(axis=a)
    m, n, k = eigs.shape
    idx = np.ogrid[:m, :n, :k]
    idx[-1] = order
    eigs = eigs[tuple(idx)]

    m, n, k, o = vecs.shape
    idx = np.ogrid[:m, :n, :k, :o]
    order = np.repeat(order[:, :, np.newaxis, :], 10, axis=2)
    idx[-1] = order
    vecs = vecs[tuple(idx)]

    return eigs, vecs


def field_gen(wn, eigs, vecs, material, ang=None, kx=None):
    """ Takes the eigenvalues and fields calculated for a given layer and finds
    the full vector of fields required to apply the boundary conditions

    Parameters
    ----------

    kx : float or 1d array
        contains a range of in-plane wavevectors to probe over

    wn : float or 1d array
        contains a range of frequencies to probe over

    eigs : 3d array
        contains eigenvalues outputted from layer_dispersion, corresponding to
        the frequencies and wavevectors provided in kx, wn

    vecs: 3d array
        contains the eigenvectors outputted from layer_dispersion corresponding
        to the frequencies and wavevectors provided in kx, wn

    material : string
        material to analyze, corresponds to an entry in ...

    Outputs
    -------

    fields : 3d array
        the full array of fields necessary to solve the boundary matching
        problem for all kx, wn. Fields are outputted in order H, E, P, X with
        Cartesian components for each ordered x, y, z
    """

    # Checks for non-array inputs and converts to numpy arrays
    if kx and type(kx) != np.ndarray:
        kx = np.array(kx)
    if type(wn) != np.ndarray:
        wn = np.array(wn)

    # Checks for equal length inputs, assumed to correspond to a cut through
    # (kx, wn) space
    if ang:
        kx = np.zeros(1)  # Creates a dummy kx to ease code maintanence below
        arrays = [(wn), (kx)]
        arr = np.array(list(product(*arrays)))
        zeta = np.zeros_like(wn)
        zeta[:] = np.sin(ang*np.pi/180)
        vecs = np.expand_dims(vecs, axis=0)
        eigs = np.expand_dims(eigs, axis=0)
        zeta = np.expand_dims(zeta, axis=0)
    else:
        arrays = [(wn), (kx)]
        arr = np.array(list(product(*arrays)))
        zeta = arr[:, 1]/arr[:, 0]
        zeta = np.transpose(zeta.reshape((len(arrays[0]), len(arrays[1]))))
        # Reshape input arrays for calculation
        kx = np.transpose(arr[:, 1].reshape((len(arrays[0]), len(arrays[1]))))
        wn = np.transpose(arr[:, 0].reshape((len(arrays[0]), len(arrays[1]))))
    # Initialize empty output vector
    field_vec = np.zeros(
        (len(arrays[1]), len(arrays[0]), 10, 12), dtype=complex
    )

    # Fetches material properties
    props = properties(material)
    layer = Media(material, props, wn)

    # Fill components which pre-exist in the input
    field_vec[:, :, :, 3] = vecs[:, :,  0, :]
    field_vec[:, :, :, 4] = vecs[:, :, 1, :]
    field_vec[:, :, :, 9] = vecs[:, :, 2, :]
    field_vec[:, :, :, 10] = vecs[:, :, 3, :]
    field_vec[:, :, :, 11] = -vecs[:, :, 4, :]

    # Broadcast zeta to the leading dimensions of the other arrays
    zetaf = np.repeat(zeta[:, :, np.newaxis], 10, axis=2)

    Ex0 = vecs[:, :, 0, :]
    Ey0 = vecs[:, :, 1, :]
    Xx0 = vecs[:, :, 2, :]
    Xy0 = vecs[:, :, 3, :]
    Xz0 = -vecs[:, :, 4, :]

    # Calculates the z-component of the electric field from Eq. in the tex file
    field_vec[:, :, :, 5] = (
        (
            Ex0*eigs*zetaf
            + layer._alpha[2, 2]*layer._mu[1, 1]
            * Xz0
            )
        / (zetaf**2-layer._eps_inf[2, 2]*layer._mu[1, 1])
        )

    # Calculates the x-component of the magnetic field from Eq. in the tex file
    field_vec[:, :, :, 0] = -eigs*Ey0/layer._mu[0, 0]

    # Calculates the y-component of the magnetic field from Eq. in the tex file
    field_vec[:, :, :, 1] = (
        (
            Ex0*eigs*layer._eps_inf[2, 2]
            + layer._alpha[2, 2]*layer._mu[1, 1]
            * Xz0*zetaf
            )
        / (zetaf**2-layer._eps_inf[2, 2]*layer._mu[1, 1])
        )

    # Calculates the z-component of the magnetic field from Eq. in the tex file
    field_vec[:, :, :, 2] = zetaf*Ey0/layer._mu[2, 2]

    # Calculates the x-component of the polarization field from Eq. in the tex
    field_vec[:, :, :, 6] = - (
        (
            Ex0*(
                zetaf**2 + layer._eps_inf[2, 2]*eigs**2
                - layer._eps_inf[2, 2]*layer._mu[1, 1]
                )
            + layer._alpha[2, 2]*layer._mu[1, 1]
            * Xz0*zetaf*eigs
            )
        / (zetaf**2-layer._eps_inf[2, 2]*layer._mu[1, 1])
        )

    # Calculates the y-component of the polarization field from Eq. in the tex
    field_vec[:, :, :, 7] = - (
        1 - eigs**2/layer._mu[0, 0] - zetaf**2/layer._mu[2, 2]
        )*Ey0

    # Calculates the z-component of the polarization field from Eq. in the tex
    field_vec[:, :, :, 8] = (
        (
            Ex0*zetaf*eigs*(layer._eps_inf[2, 2] - 1)
            + layer._alpha[2, 2]*(zetaf**2 - layer._mu[1, 1])
            * Xz0
            )
        / (zetaf**2-layer._eps_inf[2, 2]*layer._mu[1, 1])
        )

    if ang:
        return field_vec[0]
    else:
        return field_vec
