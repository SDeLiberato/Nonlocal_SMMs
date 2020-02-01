"""
Helper functions to calculate the scattering and propagation matrices for a
multilayer stack. These functions are broadly as defined in "Formulation and
comparison of two recursive matrix algorithms for modeling layered diffraction
gratings" with an extension to the nonlocal case.
"""

from typing import Tuple

import numpy as np


def Rdu(
    wn: np.ndarray,
    eigs: np.ndarray,
    Tdd_o: np.ndarray,
    Rud_o: np.ndarray,
    Rdu_o: np.ndarray,
    Tuu_o: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    thickness: float = None,
) -> np.ndarray:
    """ Calculates the interface matrix R_{du} as defined in Eq. ... of the
    paper doi: .

    Parameters
    ----------
    wn: 1D numpy array
        The probe frequencies in inverse centimetres
    eigs: 2D numpy array
        The probe eigenvalues (out-of-plane wavevectors) for all modes
        evaluated at the frequencies wn
    Tdd_o: numpy array
        Old value of the matrix T_dd
    Rud_o: numpy array
        Old value of the matrix R_ud
    Rdu_o: numpy array
        Old value of the matrix R_du
    Tuu_o: numpy array
        Old value of the matrix T_uu
    ia: numpy array
        Interface matrix for layer a, calculated automatically in creation
        of the Media class
    ib: numpy array
        Interface matrix for layer b, calculated automatically in creation
        of the Media class
    thickness: float, optional
        The current layer thickness, defaults to None indicating that the
        layer is the first or last in the stack

    Returns
    -------
    numpy array:
        A numpy array containing the matrix R_du evaluated at all
        frequencies in wn

    """

    # Calculates the matrix dimension, this differs depending on whether
    # the calculation is local or nonlocal
    dim = Tuu_o.shape[-1]

    # Checks whether the interface is physical or just an artifact of multiple
    # entries for the same layer, in the latter case sets the interface matrix
    # to the identity
    if np.all(ia == ib):
        s0 = np.zeros((len(wn), 2 * dim, 2 * dim))
        s0[:] = np.identity(2 * dim)
    else:
        s0 = interfacial_matrix(ia, ib)

    # If the layer has a finite thickness calculates the propagation matrices
    # and calculates their matrix product with the scattering matrix
    if thickness:
        lp, rp = propagation_matrices(wn, eigs, thickness)

        s = np.einsum("ijk,ikl->ijl", lp, np.einsum("ijk,ikl->ijl", s0, rp))
    else:
        s = s0

    # Checks whether it is necessary to do the matrix multiplication, in the
    # first layer Rdu_0 == 0 so we can return directly
    if np.all(Rdu_o == 0):
        rdut = s0[:, dim:, :dim]
        return rdut
    else:
        rdut = s[:, dim:, :dim]

        idenmat = np.zeros((len(wn), dim, dim))
        idenmat[:] = np.identity(dim)

        # Try to invert the matrix and if not calculate the pseudoinverse
        try:
            imat = np.linalg.inv(idenmat - np.einsum("ijk,ikl->ijl", Rud_o, rdut))
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                print("err")
                imat = np.linalg.pinv(idenmat - np.einsum("ijk,ikl->ijl", Rud_o, rdut))

        tm1 = np.einsum("ijk,ikl->ijl", Tdd_o, rdut)
        tm2 = np.einsum("ijk,ikl->ijl", imat, Tuu_o)

        return Rdu_o + np.einsum("ijk,ikl->ijl", tm1, tm2)


def Rud(
    wn: np.ndarray,
    eigs: np.ndarray,
    Tdd_o: np.ndarray,
    Rud_o: np.ndarray,
    Rdu_o: np.ndarray,
    Tuu_o: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    thickness: float = None,
) -> np.ndarray:
    """ Calculates the interface matrix R_{ud} as defined in Eq. ... of the
    paper doi: .

    Parameters
    ----------
    wn: 1D numpy array
        The probe frequencies in inverse centimetres
    eigs: 2D numpy array
        The probe eigenvalues (out-of-plane wavevectors) for all modes
        evaluated at the frequencies wn
    Tdd_o: numpy array
        Old value of the matrix T_dd
    Rud_o: numpy array
        Old value of the matrix R_ud
    Rdu_o: numpy array
        Old value of the matrix R_du
    Tuu_o: numpy array
        Old value of the matrix T_uu
    ia: numpy array
        Interface matrix for layer a, calculated automatically in creation
        of the Media class
    ib: numpy array
        Interface matrix for layer b, calculated automatically in creation
        of the Media class
    thickness: float, optional
        The current layer thickness, defaults to None indicating that the
        layer is the first or last in the stack

    Returns
    -------
    numpy array:
        A numpy array containing the matrix R_du evaluated at all
        frequencies in wn


    """
    # Calculates the matrix dimension, this differs depending on whether
    # the calculation is local or nonlocal
    dim = Tuu_o.shape[-1]

    # Checks whether the interface is physical or just an artifact of multiple
    # entries for the same layer, in the latter case sets the interface matrix
    # to the identity
    if np.all(ia == ib):
        s0 = np.zeros((len(wn), 2 * dim, 2 * dim))
        s0[:] = np.identity(2 * dim)
    else:
        s0 = interfacial_matrix(ia, ib)

    # If the layer has a finite thickness calculates the propagation matrices
    # and calculates their matrix product with the scattering matrix
    if thickness:
        lp, rp = propagation_matrices(wn, eigs, thickness)
        s = np.einsum("ijk,ikl->ijl", lp, np.einsum("ijk,ikl->ijl", s0, rp))
    else:
        s = s0

    # Checks whether it is necessary to do the matrix multiplication, in the
    # first layer Rdu_0 == 0 so we can return directly
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

        # Try to invert the matrix and if not calculate the pseudoinverse
        try:
            imat = np.linalg.inv(idenmat - np.einsum("ijk,ikl->ijl", rdut, Rud_o))
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                print(err)
                imat = np.linalg.pinv(idenmat - np.einsum("ijk,ikl->ijl", rdut, Rud_o))

        tm1 = np.einsum("ijk,ikl->ijl", tuut, Rud_o)
        tm2 = np.einsum("ijk,ikl->ijl", imat, tddt)

        return rudt + np.einsum("ijk,ikl->ijl", tm1, tm2)


def Tuu(
    wn: np.ndarray,
    eigs: np.ndarray,
    Tdd_o: np.ndarray,
    Rud_o: np.ndarray,
    Rdu_o: np.ndarray,
    Tuu_o: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    thickness: float = None,
) -> np.ndarray:
    """ Calculates the interface matrix T_{uu} as defined in Eq. ... of the
    paper doi: .

    Parameters
    ----------
    wn: 1D numpy array
        The probe frequencies in inverse centimetres
    eigs: 2D numpy array
        The probe eigenvalues (out-of-plane wavevectors) for all modes
        evaluated at the frequencies wn
    Tdd_o: numpy array
        Old value of the matrix T_dd
    Rud_o: numpy array
        Old value of the matrix R_ud
    Rdu_o: numpy array
        Old value of the matrix R_du
    Tuu_o: numpy array
        Old value of the matrix T_uu
    ia: numpy array
        Interface matrix for layer a, calculated automatically in creation
        of the Media class
    ib: numpy array
        Interface matrix for layer b, calculated automatically in creation
        of the Media class
    thickness: float, optional
        The current layer thickness, defaults to None indicating that the
        layer is the first or last in the stack

    Returns
    -------
    numpy array:
        A numpy array containing the matrix R_du evaluated at all
        frequencies in wn


    """

    # Calculates the matrix dimension, this differs depending on whether
    # the calculation is local or nonlocal
    dim = Tuu_o.shape[-1]

    # Checks whether the interface is physical or just an artifact of multiple
    # entries for the same layer, in the latter case sets the interface matrix
    # to the identity
    if np.all(ia == ib):
        s0 = np.zeros((len(wn), 2 * dim, 2 * dim))
        s0[:] = np.identity(2 * dim)
    else:
        s0 = interfacial_matrix(ia, ib)

    # If the layer has a finite thickness calculates the propagation matrices
    # and calculates their matrix product with the scattering matrix
    if thickness:
        lp, rp = propagation_matrices(wn, eigs, thickness)

        s = np.einsum("ijk,ikl->ijl", lp, np.einsum("ijk,ikl->ijl", s0, rp))
    else:
        s = s0

    # Checks whether it is necessary to do the matrix multiplication, in the
    # first layer Rdu_0 == 0 so we can return directly
    if np.all(Rdu_o == 0):
        tuut = s0[:, :dim, :dim]
        return tuut
    else:
        rdut = s[:, dim:, :dim]
        tuut = s[:, :dim, :dim]

        idenmat = np.zeros((len(wn), dim, dim))
        idenmat[:] = np.identity(dim)

        # Tries to invert the matrix and if not finds the pseudoinverse
        try:
            imat = np.linalg.inv(idenmat - np.einsum("ijk,ikl->ijl", Rud_o, rdut))
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                print(err)
                imat = np.linalg.pinv(idenmat - np.einsum("ijk,ikl->ijl", Rud_o, rdut))

        tm1 = np.einsum("ijk,ikl->ijl", tuut, imat)

        return np.einsum("ijk,ikl->ijl", tm1, Tuu_o)


def Tdd(
    wn: np.ndarray,
    eigs: np.ndarray,
    Tdd_o: np.ndarray,
    Rud_o: np.ndarray,
    Rdu_o: np.ndarray,
    Tuu_o: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    thickness: float = None,
) -> np.ndarray:
    """ Calculates the interface matrix T_{dd} as defined in Eq. ... of the
    paper doi: .

    Parameters
    ----------
    wn: 1D numpy array
        The probe frequencies in inverse centimetres
    eigs: 2D numpy array
        The probe eigenvalues (out-of-plane wavevectors) for all modes
        evaluated at the frequencies wn
    Tdd_o: numpy array
        Old value of the matrix T_dd
    Rud_o: numpy array
        Old value of the matrix R_ud
    Rdu_o: numpy array
        Old value of the matrix R_du
    Tuu_o: numpy array
        Old value of the matrix T_uu
    ia: numpy array
        Interface matrix for layer a, calculated automatically in creation
        of the Media class
    ib: numpy array
        Interface matrix for layer b, calculated automatically in creation
        of the Media class
    thickness: float, optional
        The current layer thickness, defaults to None indicating that the
        layer is the first or last in the stack

    Returns
    -------
    numpy array:
        A numpy array containing the matrix R_du evaluated at all
        frequencies in wn


    """
    # Calculates the matrix dimension, this differs depending on whether
    # the calculation is local or nonlocal
    dim = Tuu_o.shape[-1]

    # Checks whether the interface is physical or just an artifact of multiple
    # entries for the same layer, in the latter case sets the interface matrix
    # to the identity
    if np.all(ia == ib):
        s0 = np.zeros((len(wn), 2 * dim, 2 * dim))
        s0[:] = np.identity(2 * dim)
    else:
        s0 = interfacial_matrix(ia, ib)

    # If the layer has a finite thickness calculates the propagation matrices
    # and calculates their matrix product with the scattering matrix
    if thickness:
        lp, rp = propagation_matrices(wn, eigs, thickness)

        s = np.einsum("ijk,ikl->ijl", lp, np.einsum("ijk,ikl->ijl", s0, rp))
    else:
        s = s0

    # Checks whether it is necessary to do the matrix multiplication, in the
    # first layer Rdu_0 == 0 so we can return directly
    if np.all(Rdu_o == 0):
        tddt = s0[:, dim:, dim:]
        return tddt
    else:
        rdut = s[:, dim:, :dim]
        tddt = s[:, dim:, dim:]

        idenmat = np.zeros((len(wn), dim, dim))
        idenmat[:] = np.identity(dim)

        # Tries to invert matrix and if not calculates the pseudoinverse
        try:
            imat = np.linalg.inv(idenmat - np.einsum("ijk,ikl->ijl", rdut, Rud_o))
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                print(err)
                imat = np.linalg.pinv(idenmat - np.einsum("ijk,ikl->ijl", rdut, Rud_o))

        tm1 = np.einsum("ijk,ikl->ijl", Tdd_o, imat)

        return np.einsum("ijk,ikl->ijl", tm1, tddt)


def propagation_matrices(
    wn: np.ndarray, eigs: np.ndarray, thickness: float
) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculates the propagation matrices for a layer of defined thickness
    given that layer's eigenvalues (out-of-plane wavevectors). This matrix is
    defined in

    Parameters
    ----------

    wn: 1D numpy array
        The probe wavenumbers in inverse centimetres
    eigs: 2D numpy array
        The eigenvalues for each mode at each frequency in wn
    thickness: float
        The layer thickness in metres

    Returns
    -------

    tuple:
        containing the two propagation matrices as defined in
        in "Formulation and comparison of two recursive matrix algorithms
        for modeling layered diffraction gratings"
    """

    # Calculates the eigenvalue dimensions, this will differ depending on
    # whether a nonlocal or local calculation is performed
    dim = eigs.shape[-1] // 2
    conv = 2 * np.pi / 0.01  # Factor to convert from inverse cm to Hz

    # Calculates the diagonal propagation vectors from the complex eigs
    diag_vecr = np.exp(
        (1j * eigs.real - np.abs(eigs.imag)) * wn[:, None] * conv * thickness
    )

    diag_vecl = np.exp(
        (-1j * eigs.real - np.abs(eigs.imag)) * wn[:, None] * conv * thickness
    )

    # Initialise the two propagation matrices
    propmat1 = np.zeros((len(wn), 2 * dim, 2 * dim), dtype=complex)
    propmat2 = np.zeros((len(wn), 2 * dim, 2 * dim), dtype=complex)

    for idx in range(dim):
        propmat1[:, idx, idx] = 1
        propmat1[:, idx + dim, idx + dim] = diag_vecl[:, idx + dim]
        propmat2[:, idx, idx] = diag_vecr[:, idx]
        propmat2[:, idx + dim, idx + dim] = 1

    return propmat1, propmat2


def interfacial_matrix(
    i0: np.ndarray, i1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculates the interfacial matrix between two layers with
    interface matrices i0 and i1

    Parameters
    ----------

    i0: 2D numpy array
        The interface matrix for layer 0
    i1: 2D numpy array
        The interface matrix for layer 1

    Returns
    -------

    2D numpy array:
        The interfacial matrix between layer 0 and layer 1
    """

    # Tries to directly invert and if not uses the pseudoinverse
    try:
        t = np.einsum("ijk,ikl->ijl", np.linalg.inv(i1), i0)
    except np.linalg.LinAlgError as err:
        if "Singular matrix" in str(err):
            t = np.einsum("ijk,ikl->ijl", np.linalg.pinv(i1), i0)

    # finds the problem dimension, differs in the local and nonlocal cases
    dim = t.shape[2] // 2
    t11 = t[:, :dim, :dim]
    t12 = t[:, :dim, dim:]
    t21 = t[:, dim:, :dim]
    t22 = t[:, dim:, dim:]

    try:
        it22 = np.linalg.inv(t22)
    except np.linalg.LinAlgError as err:
        if "Singular matrix" in str(err):
            it22 = np.linalg.pinv(t22)

    s1 = np.zeros_like(t, dtype=complex)
    # Calculates the 4 blocks comprising s1
    s1[:, :dim, :dim] = t11 - np.einsum(
        "ijk,ikl->ijl", t12, np.einsum("ijk,ikl->ijl", it22, t21)
    )
    s1[:, :dim, dim:] = np.einsum("ijk,ikl->ijl", t12, it22)
    s1[:, dim:, :dim] = -np.einsum("ijk,ikl->ijl", it22, t21)
    s1[:, dim:, dim:] = it22
    return s1
