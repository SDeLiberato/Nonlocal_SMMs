"""
Helper functions to calculate the scattering and propagation matrices for a
multilayer stack. These functions are broadly as defined in "Formulation and
comparison of two recursive matrix algorithms for modeling layered diffraction
gratings" with an extension to the nonlocal case.
"""

import numpy as np

def prop_mats(wn, eigs, thickness):
    """ Calculates the propagation matrices for a layer of defined thickness
    given that layer's eigenvalues (out-of-plane wavevectors)

    :param wn: The probe wavenumbers in inverse centimetres
    :type wn: 1D numpy array
    :param eigs: The eigenvalues for each mode at each frequency in wn
    :type eigs: 2D numpy array
    :param thickness: The layer thickness in metres
    :type thickness: float

    :return: a tuple containing the two propagation matrices as defined in
        in "Formulation and comparison of two recursive matrix algorithms
        for modeling layered diffraction gratings"
    :rtype: tuple
    """

    dim = eigs.shape[-1]//2
    conv = 2*np.pi/0.01

    # diag_vecr = np.exp(1j*eigs*wn[:, None]*conv*thickness)
    # diag_vecl = np.exp(-1j*eigs*wn[:, None]*conv*thickness)

    # Fixed the eigenvalues further down instead so can use ^^^
    diag_vecr = np.exp(
        (1j*eigs.real-np.abs(eigs.imag))*wn[:, None]*conv*thickness
        )

    diag_vecl = np.exp(
        (-1j*eigs.real-np.abs(eigs.imag))*wn[:, None]*conv*thickness
        )

    propmat2 = np.zeros((len(wn), 2*dim, 2*dim), dtype=complex)
    propmat1 = np.zeros((len(wn), 2*dim, 2*dim), dtype=complex)

    for idx in range(dim):
        propmat2[:, idx, idx] = diag_vecr[:, idx]
        propmat2[:, idx+dim, idx+dim] = 1
        propmat1[:, idx, idx] = 1
        propmat1[:, idx+dim, idx+dim] = diag_vecl[:, idx+dim]

    return propmat1, propmat2


def s_mat(i0, i1):

    try:
        t = np.einsum('ijk,ikl->ijl', np.linalg.inv(i1), i0)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            t = np.einsum('ijk,ikl->ijl', np.linalg.pinv(i1), i0)

    dim = t.shape[2]//2
    t11 = t[:, :dim, :dim]
    t12 = t[:, :dim, dim:]
    t21 = t[:, dim:, :dim]
    t22 = t[:, dim:, dim:]

    try:
        it22 = np.linalg.inv(t22)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            it22 = np.linalg.pinv(t22)

    s1 = np.zeros_like(t, dtype=complex)
    s1[:, :dim, :dim] = (t11
                         - np.einsum('ijk,ikl->ijl', t12,
                                     np.einsum('ijk,ikl->ijl', it22, t21)
                                     )
                         )
    s1[:, :dim, dim:] = np.einsum('ijk,ikl->ijl', t12, it22)
    s1[:, dim:, :dim] = - np.einsum('ijk,ikl->ijl', it22, t21)
    s1[:, dim:, dim:] = it22
    return s1


def Rdu(wn, eigs, Tdd_o, Rud_o, Rdu_o, Tuu_o, ia, ib, thickness=None):
    # First layer thickness = 0

    dim = Tuu_o.shape[-1]

    if np.all(ia == ib):
        s0 = np.zeros((len(wn), 2*dim, 2*dim))
        s0[:] = np.identity(2*dim)
    else:
        s0 = s_mat(ia, ib)

    if thickness:
        lp, rp = prop_mats(wn, eigs, thickness)

        s = np.einsum('ijk,ikl->ijl', lp,
                      np.einsum('ijk,ikl->ijl', s0, rp)
                      )
    else:
        s = s0

    if np.all(Rdu_o == 0):
        rdut = s0[:, dim:, :dim]
        return rdut
    else:
        rdut = s[:, dim:, :dim]

        idenmat = np.zeros((len(wn), dim, dim))
        idenmat[:] = np.identity(dim)

        try:
            imat = np.linalg.inv(
                idenmat - np.einsum('ijk,ikl->ijl', Rud_o, rdut)
            )
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('err')
                imat = np.linalg.pinv(
                    idenmat - np.einsum('ijk,ikl->ijl', Rud_o, rdut)
                )

        tm1 = np.einsum('ijk,ikl->ijl', Tdd_o, rdut)
        tm2 = np.einsum('ijk,ikl->ijl', imat, Tuu_o)

        return Rdu_o + np.einsum('ijk,ikl->ijl', tm1, tm2)


def Rud(wn, eigs, Tdd_o, Rud_o, Rdu_o, Tuu_o, ia, ib, thickness=None):
    # First layer thickness = 0

    dim = Tuu_o.shape[-1]

    if np.all(ia == ib):
        s0 = np.zeros((len(wn), 2*dim, 2*dim))
        s0[:] = np.identity(2*dim)
    else:
        s0 = s_mat(ia, ib)

    if thickness:
        lp, rp = prop_mats(wn, eigs, thickness)
        s = np.einsum('ijk,ikl->ijl', lp,
                      np.einsum('ijk,ikl->ijl', s0, rp)
                      )
    else:
        s = s0

    if np.all(Rdu_o == 0):
        rudt = s0[:, :dim, dim:]
        return rudt
    else:
        rudt = s[:, :dim, dim:]
        rdut = s[:, dim:, :dim]
        tuut = s[:, :dim, :dim]
        tddt = s[:, dim:, dim:]

        idenmat = np.zeros((len(wn), dim, dim))
        idenmat[:] = np.identity(dim)

        try:
            imat = np.linalg.inv(
                idenmat - np.einsum('ijk,ikl->ijl', rdut, Rud_o)
            )
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print(err)
                imat = np.linalg.pinv(
                    idenmat - np.einsum('ijk,ikl->ijl', rdut, Rud_o)
                )

        tm1 = np.einsum('ijk,ikl->ijl', tuut, Rud_o)
        tm2 = np.einsum('ijk,ikl->ijl', imat, tddt)

        return rudt + np.einsum('ijk,ikl->ijl', tm1, tm2)


def Tuu(wn, eigs, Tdd_o, Rud_o, Rdu_o, Tuu_o, ia, ib, thickness=None):
    # First layer thickness = 0
    dim = Tuu_o.shape[-1]

    if np.all(ia == ib):
        s0 = np.zeros((len(wn), 2*dim, 2*dim))
        s0[:] = np.identity(2*dim)
    else:
        s0 = s_mat(ia, ib)

    if thickness:
        lp, rp = prop_mats(wn, eigs, thickness)

        s = np.einsum('ijk,ikl->ijl', lp,
                      np.einsum('ijk,ikl->ijl', s0, rp)
                      )
    else:
        s = s0

    if np.all(Rdu_o == 0):
        tuut = s0[:, :dim, :dim]
        return tuut
    else:
        rdut = s[:, dim:, :dim]
        tuut = s[:, :dim, :dim]

        idenmat = np.zeros((len(wn), dim, dim))
        idenmat[:] = np.identity(dim)

        try:
            imat = np.linalg.inv(
                idenmat - np.einsum('ijk,ikl->ijl', Rud_o, rdut)
            )
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print(err)
                imat = np.linalg.pinv(
                    idenmat - np.einsum('ijk,ikl->ijl', Rud_o, rdut)
                )

        tm1 = np.einsum('ijk,ikl->ijl', tuut, imat)

        return np.einsum('ijk,ikl->ijl', tm1, Tuu_o)


def Tdd(wn, eigs, Tdd_o, Rud_o, Rdu_o, Tuu_o, ia, ib, thickness=None):
    # First layer thickness = 0
    dim = Tuu_o.shape[-1]

    if np.all(ia == ib):
        s0 = np.zeros((len(wn), 2*dim, 2*dim))
        s0[:] = np.identity(2*dim)
    else:
        s0 = s_mat(ia, ib)

    if thickness:
        lp, rp = prop_mats(wn, eigs, thickness)

        s = np.einsum('ijk,ikl->ijl', lp,
                      np.einsum('ijk,ikl->ijl', s0, rp)
                      )
    else:
        s = s0

    if np.all(Rdu_o == 0):
        tddt = s0[:, dim:, dim:]
        return tddt
    else:
        rdut = s[:, dim:, :dim]
        tddt = s[:, dim:, dim:]

        idenmat = np.zeros((len(wn), dim, dim))
        idenmat[:] = np.identity(dim)

        try:
            imat = np.linalg.inv(
                idenmat - np.einsum('ijk,ikl->ijl', rdut, Rud_o)
            )
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print(err)
                imat = np.linalg.pinv(
                    idenmat - np.einsum('ijk,ikl->ijl', rdut, Rud_o)
                )

        tm1 = np.einsum('ijk,ikl->ijl', Tdd_o, imat)

        return np.einsum('ijk,ikl->ijl', tm1, tddt)