__all__ = ['layer_dispersion', 'field_gen']

import numpy as np
from itertools import product
from materials.materials import Media, properties


def layer_dispersion(kx, wn, material, ang=None):
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

    fields : array
        Values representing the calculated field components corresponding to
        the eigenmodes of the layer


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
    if type(kx) != np.ndarray:
        kx = np.array(kx)
    if type(wn) != np.ndarray:
        wn = np.array(wn)

    # Checks for equal length inputs, assumed to correspond to a cut through
    # (kx, wn) space
    if len(kx) == len(wn):
        arr = np.transpose(np.array([wn, kx]))
    else:
        arrays = [(wn), (kx)]
        arr = np.array(list(product(*arrays)))

    zeta = arr[:, 1]/arr[:, 0]
    # Checks for an input angle, if present recalculates zeta
    if ang:
        zeta[:] = np.sin(ang*np.pi/180)

    # Fetches material properties
    props = properties(material)

    layer = Media(material, props, wn)

    # Creates an array of len(zeta) 5 x 5 identity matrices
    It = np.identity(5)
    It = np.swapaxes(np.broadcast_to(It[..., None], (5, 5, len(zeta))), 0, 2)
    # Creates an array of len(zeta) 5 x 5 null matrices
    Zt = np.zeros_like(It)
    # Creates an array of len(zeta) 5 x 5 null matrices to be filled
    At = np.zeros_like(It, dtype=complex)
    Bt = np.zeros_like(It, dtype=complex)

    At[:, 0, 0] = (
        layer._eps_inf[2, 2]/(zeta**2-layer._eps_inf[2, 2]*layer._mu[1, 1])
        )
    At[:, 1, 1] = (-1/layer._mu[0, 0])
    At[:, 2, 2] = (layer._beta_c**2/2)
    At[:, 3, 3] = (layer._beta_c**2/2)
    At[:, 4, 4] = (layer._beta_l**2)

    Bt[:, 0, 4] = (
        zeta*np.sqrt(4*np.pi)*layer._alpha[2, 2]*layer._mu[1, 1]
        / (zeta**2-layer._eps_inf[2, 2]*layer._mu[1, 1])
        )
    Bt[:, 2, 4] = zeta*(layer._beta_l**2-2*layer._beta_t**2+layer._beta_c**2/2)
    Bt[:, 4, 0] = (
        layer._alpha[2, 2]/np.sqrt(4*np.pi)*zeta
        / (zeta**2-layer._eps_inf[2, 2]*layer._mu[1, 1])/arr[:, 0]**2
        )
    Bt[:, 4, 2] = zeta*(layer._beta_l**2-2*layer._beta_t**2+layer._beta_c**2/2)

    Ct = np.zeros_like(It, dtype=complex)

    Ct[:, 0, 0] = layer._eps_inf[0, 0]
    Ct[:, 0, 2] = np.sqrt(4*np.pi)*layer._alpha[0, 0]

    Ct[:, 1, 1] = layer._eps_inf[1, 1] - zeta**2/layer._mu[2, 2]
    Ct[:, 1, 3] = np.sqrt(4*np.pi)*layer._alpha[2, 2]

    Ct[:, 2, 0] = layer._alpha[0, 0]/np.sqrt(4*np.pi)/arr[:, 0]**2
    Ct[:, 2, 2] = (
        1 + 1j*layer._gamma/arr[:, 0]
        - layer._wto_pe**2/arr[:, 0]**2 + zeta**2*layer._beta_l**2
        )

    Ct[:, 3, 1] = layer._alpha[1, 1]/np.sqrt(4*np.pi)/arr[:, 0]**2
    Ct[:, 3, 3] = (
        1 + 1j*layer._gamma/arr[:, 0]
        - layer._wto_pe**2/arr[:, 0]**2 + zeta**2*layer._beta_c**2/2
        )
    Ct[:, 4, 4] = (
        1 + 1j*layer._gamma/arr[:, 0]
        - layer._wto_pa**2/arr[:, 0]**2 + zeta**2*layer._beta_c**2/2
        + layer._alpha[2, 2]**2*layer._mu[1, 1]/arr[:, 0]**2
        / (zeta**2-layer._eps_inf[2, 2]*layer._mu[1, 1])
        )

    iAt = np.zeros_like(It, dtype=complex)
    iBt = np.zeros_like(It, dtype=complex)

    iAt[:, 0, 0] = (
        (zeta**2-layer._eps_inf[2, 2]*layer._mu[1, 1])/layer._eps_inf[2, 2]
        )
    iAt[:, 1, 1] = -layer._mu[0, 0]
    iAt[:, 2, 2] = 2/layer._beta_c**2
    iAt[:, 3, 3] = 2/layer._beta_c**2
    iAt[:, 4, 4] = 1/layer._beta_l**2

    iBt[:, 0, 4] = -(
        zeta*np.sqrt(4*np.pi)*layer._alpha[2, 2]*layer._mu[1, 1]
        / layer._eps_inf[2, 2]
        )
    iBt[:, 2, 4] = -zeta*(layer._beta_l**2-2*layer._beta_t**2+layer._beta_c**2/2)*(2/layer._beta_c**2/2)
    iBt[:, 4, 0] = -(
        layer._alpha[2, 2]/np.sqrt(4*np.pi)*zeta
        / (zeta**2-layer._eps_inf[2, 2]*layer._mu[1, 1])/arr[:, 0]**2
        )/layer._beta_l**2
    iBt[:, 4, 2] = -zeta*(layer._beta_l**2-2*layer._beta_t**2+layer._beta_c**2/2)*(2/layer._beta_c**2/2)

    M1 = -np.concatenate((
        np.hstack((Zt, It)),
        np.hstack((iAt, iBt))
        ), axis=2)

    M2 = np.concatenate((
        np.hstack((Ct, Zt)),
        np.hstack((Zt, -It))
        ), axis=2)

    Mf = np.matmul(M1, M2)

    eigs, vecs = np.linalg.eig(Mf)

    # # Order the calculated eigenvectors
    # index = np.argsort(eigs)
    # eigs = eigs[np.arange(np.shape(eigs)[0])[:, np.newaxis], index]
    # vecs = vecs[np.arange(np.shape(eigs)[0])[:, np.newaxis], :, index]

    # # Reshape into a useable array
    eigs = eigs.reshape((len(kx), len(wn), 10), order='F')
    vecs = vecs.reshape((len(kx), len(wn), 10, 10), order='F')
    return eigs, vecs


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


def field_gen(kx, wn, eigs, vecs, material):
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
    if type(kx) != np.ndarray:
        kx = np.array(kx)
    if type(wn) != np.ndarray:
        wn = np.array(wn)

    # Checks for equal length inputs, assumed to correspond to a cut through
    # (kx, wn) space
    if len(kx) == len(wn):
        arr = np.transpose(np.array([wn, kx]))
        zeta = arr[:, 1]/arr[:, 0]
    else:
        arrays = [(wn), (kx)]
        arr = np.array(list(product(*arrays)))
        zeta = arr[:, 1]/arr[:, 0]
        zeta = np.transpose(zeta.reshape((len(arrays[0]), len(arrays[1]))))

    # Fetches material properties
    props = properties(material)
    layer = Media(material, props, wn)

    # Reshape input arrays for calculation
    kx = np.transpose(arr[:, 1].reshape((len(arrays[0]), len(arrays[1]))))
    wn = np.transpose(arr[:, 0].reshape((len(arrays[0]), len(arrays[1]))))

    # Initialize empty output vector
    field_vec = np.zeros(
        (len(arrays[1]), len(arrays[0]), 10, 12), dtype=complex
        )

    # Fill components which pre-exist in the input
    field_vec[:, :, :, 3] = vecs[:, :, :, 0]
    field_vec[:, :, :, 4] = vecs[:, :, :, 1]
    field_vec[:, :, :, 9] = vecs[:, :, :, 2]
    field_vec[:, :, :, 10] = vecs[:, :, :, 3]
    field_vec[:, :, :, 11] = vecs[:, :, :, 4]

    # Broadcast zeta to the leading dimensions of the other arrays
    zetaf = np.repeat(zeta[:, :, np.newaxis], 10, axis=2)

    # Calculates the z-component of the electric field from Eq. in the tex file
    field_vec[:, :, :, 5] = (
        (
            vecs[:, :, :, 0]*eigs*zetaf
            + np.sqrt(4*np.pi)*layer._alpha[2, 2]*layer._mu[1, 1]
            * vecs[:, :, :, 4]
            )
        / (zetaf**2-layer._eps_inf[2, 2]*layer._mu[1, 1])
        )

    # Calculates the x-component of the magnetic field from Eq. in the tex file
    field_vec[:, :, :, 0] = -eigs*vecs[:, :, :, 1]/layer._mu[0, 0]

    # Calculates the y-component of the magnetic field from Eq. in the tex file
    field_vec[:, :, :, 1] = - (
        (
            vecs[:, :, :, 0]*eigs*layer._eps_inf[2, 2]
            + np.sqrt(4*np.pi)*layer._alpha[2, 2]*layer._mu[1, 1]
            * vecs[:, :, :, 4]*zetaf
            )
        / (zetaf**2-layer._eps_inf[2, 2]*layer._mu[1, 1])
        )

    # Calculates the z-component of the magnetic field from Eq. in the tex file
    field_vec[:, :, :, 2] = zetaf*vecs[:, :, :, 1]/layer._mu[2, 2]

    # Calculates the x-component of the polarization field from Eq. in the tex
    field_vec[:, :, :, 6] = - 1/4/np.pi*(
        (
            vecs[:, :, :, 0]*(
                zetaf**2 + layer._eps_inf[2, 2]*eigs**2
                - layer._eps_inf[2, 2]*layer._mu[1, 1]
                )
            + np.sqrt(4*np.pi)*layer._alpha[2, 2]*layer._mu[1, 1]
            * vecs[:, :, :, 4]*zetaf*eigs
            )
        / (zetaf**2-layer._eps_inf[2, 2]*layer._mu[1, 1])
        )

    # Calculates the y-component of the polarization field from Eq. in the tex
    field_vec[:, :, :, 7] = - 1/4/np.pi*(
        1 - eigs**2/layer._mu[0, 0] - zetaf**2/layer._mu[2, 2]
        )*field_vec[:, :, :, 1]

    # Calculates the z-component of the polarization field from Eq. in the tex
    field_vec[:, :, :, 8] = 1/4/np.pi*(
        (
            vecs[:, :, :, 0]*zetaf*eigs*(layer._eps_inf[2, 2] - 1)
            + np.sqrt(4*np.pi)*layer._alpha[2, 2]*(zetaf**2 - layer._mu[1, 1])
            * vecs[:, :, :, 4]
            )
        / (zetaf**2-layer._eps_inf[2, 2]*layer._mu[1, 1])
        )

    return field_vec
