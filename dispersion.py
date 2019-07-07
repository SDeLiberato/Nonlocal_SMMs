__all__ = ['layer_dispersion']

import numpy as np
from itertools import product
from materials import epsilon_f, mu_f, properties

def layer_dispersion(kx, wn, mat, ang=None):
    """ Calculates the dispersion of the photonic and longitudinal and transverse
    phonon eigenmodes of an anisotropic medium, describable by a diagonal
    permeability and permitivitty

    Parameters
    ----------

    kx : float or 1d array
        contains a range of in-plane wavevectors to probe over

    wn : float or 1d array
        contains a range of frequencies to probe over

    mat : string
        material to analyse, corresponds to an entry in ...

    ang : float
        optional, corresponds to the incident angle. useage overrides entry
        of kx

    Returns
    -------

    eigs : array
        Values representing calculated out-of-plane wavevectors of the eigenmodes
        of the layer.

    fields : array
        Values representing the calculated field components corresponding to the
        eigenmodes of the layer


    References
    ----------

    The formation of the matrix from which the dispersions are calculated is
    detailed at " "
    """

    nkx = len(kx)
    ### Checks for non-array inputs and converts to numpy arrays
    if type(kx) != np.ndarray:
        kx = np.array(kx)
    if type(wn) != np.ndarray:
        wn = np.array(wn)

    ### Checks for equal length inputs, assumed to correspond to a cut through
    ### (kx, wn) space
    if len(kx)==len(wn):
        arr = np.transpose(np.array([wn,kx]))
    else:
        arrays = [(wn), (kx)]
        arr = np.array(list(product(*arrays)))

    zeta = arr[:,1]/arr[:,0]
    ### Checks for an input angle, if present recalculates zeta
    if ang:
        zeta[:] = np.sin(ang*np.pi/180)
    mu1 = 1


    ### Fetches material properties
    props = properties(mat)
    eps_inf_pe, eps_inf_pa = props['eps_inf_pe'], props['eps_inf_pa']
    eps_0_pe, eps_0_pa = props['eps_0_pe'], props['eps_0_pa']
    wto_pe, wto_pa = props['wto_pe'], props['wto_pa']
    gamma = props['gamma']
    beta_t, beta_l, beta_c = props['beta_t'], props['beta_l'], props['beta_c']

    ## Calculates the permitivitty and permeability tensors
    eps = epsilon_f(props,arr[:,0])
    mu = mu_f(props,arr[:,0])

    ### Creates tensors containing the high-frequency permitivitty
    ### and oscillator strength assuming the crystal c-axis is orientated
    ### along the third dimension
    eps_inf = np.diag(
            [eps_inf_pe, eps_inf_pe, eps_inf_pa]
        )

    alpha = np.diag(
            [np.sqrt(eps_0_pe-eps_inf_pe)*wto_pe,
            np.sqrt(eps_0_pe-eps_inf_pe)*wto_pe,
            np.sqrt(eps_0_pa-eps_inf_pa)*wto_pa]
        )

    ### Creates an array of len(zeta) 5 x 5 identity matrices
    It = np.identity(5)
    It = np.swapaxes(np.broadcast_to(It[...,None], (5,5,len(zeta))), 0, 2)
    ### Creates an array of len(zeta) 5 x 5 null matrices
    Zt = np.zeros_like(It)
    ### Creates an array of len(zeta) 5 x 5 null matrices to be filled
    At = np.zeros_like(It, dtype=complex)
    Bt = np.zeros_like(It, dtype=complex)

    At[:,0,0] = (zeta**2-eps_inf[2,2]*mu[0,0])/eps_inf[2,2]
    At[:,1,1] = -mu[0,0]
    At[:,2,2] = 2/beta_c**2
    At[:,2,2] = -2/beta_c**2
    At[:,3,3] = 2/beta_c**2
    At[:,4,4] = 1/beta_l**2
    At[:,3,3] = -2/beta_c**2
    At[:,4,4] = -1/beta_l**2

    Bt[:,0,4] = - zeta*np.sqrt(4*np.pi)*alpha[2,2]/eps_inf[2,2]
    Bt[:,2,4] = - zeta*(beta_l**2-2*beta_t**2+beta_c**2/2)*2/beta_c**2
    Bt[:,4,0] = - zeta*alpha[2,2]/np.sqrt(4*np.pi)/(zeta**2-eps_inf[2,2]*mu[1,1])/beta_l**2/arr[:,0]**2
    Bt[:,4,0] =  zeta*alpha[2,2]/np.sqrt(4*np.pi)/(zeta**2-eps_inf[2,2]*mu[1,1])/beta_l**2/arr[:,0]**2
    Bt[:,4,2] = - zeta*(beta_l**2-2*beta_t**2+beta_c**2/2)/beta_l**2

    Ct = np.zeros_like(It, dtype=complex)
    Ct[:,0,0] = eps_inf[0,0]
    Ct[:,0,2] = np.sqrt(4*np.pi)*alpha[0,0]
    Ct[:,1,1] = eps_inf[1,1] - zeta**2/mu[2,2]
    Ct[:,1,3] = np.sqrt(4*np.pi)*alpha[1,1]
    Ct[:,2,0] = alpha[0,0]/np.sqrt(4*np.pi)/arr[:,0]**2
    Ct[:,2,2] = 1 + 1j*gamma/arr[:,0] - wto_pe**2/arr[:,0]**2 + zeta**2*beta_l**2
    Ct[:,2,2] = 1 + 1j*gamma/arr[:,0] - wto_pe**2/arr[:,0]**2 - zeta**2*beta_l**2

    Ct[:,3,1] = alpha[1,1]/np.sqrt(4*np.pi)/arr[:,0]**2
    Ct[:,3,3] = 1 + 1j*gamma/arr[:,0] - wto_pe**2/arr[:,0]**2 + zeta**2*beta_c**2/2
    Ct[:,4,4] = 1 + 1j*gamma/arr[:,0] - wto_pa**2/arr[:,0]**2 + zeta**2*beta_c**2/2
    Ct[:,3,3] = 1 + 1j*gamma/arr[:,0] - wto_pe**2/arr[:,0]**2 - zeta**2*beta_c**2/2
    Ct[:,4,4] = 1 + 1j*gamma/arr[:,0] - wto_pa**2/arr[:,0]**2 - zeta**2*beta_c**2/2

    M1 = -np.concatenate((
        np.hstack((Zt, It)),
        np.hstack((At, Bt))
        ), axis=2)
    M2 = np.concatenate((
        np.hstack((Ct, Zt)),
        np.hstack((Zt, -It))
        ), axis=2)
    Mf = np.matmul(M1,M2)

    eigs, fields = np.linalg.eig(Mf)

    ### Order the calculated eigenvectors
    fields[np.arange(np.shape(eigs)[0])[:,np.newaxis], np.argsort(eigs), :]
    eigs = eigs[np.arange(np.shape(eigs)[0])[:,np.newaxis], np.argsort(eigs)]
    ### Reshape into a useable array
    eigs = eigs[:,:].reshape((len(kx), len(wn),10), order='F')
    fields = fields[:,:].reshape((len(kx), len(wn),10, 10), order='F')
    return eigs, fields


