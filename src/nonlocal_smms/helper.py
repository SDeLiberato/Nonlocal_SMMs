"""Helper functions to calculate the scattering and propagation matrices for a multilayer stack.

These functions are broadly as defined in "Formulation and
comparison of two recursive matrix algorithms for modeling layered diffraction
gratings" with an extension to the nonlocal case.
"""

from typing import Tuple

import numpy as np


def Rdu(
    wavenumber: np.ndarray,
    eigenvalues: np.ndarray,
    Tdd_o: np.ndarray,
    Rud_o: np.ndarray,
    Rdu_o: np.ndarray,
    Tuu_o: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    thickness: float = None,
) -> np.ndarray:
    """Calculates the interface matrix R_{du}.

    This matrix is defined in Eq. ... of the paper doi or in the projects online
    documentation.

    Args:
        wavenumber (np.ndarray): The probe frequencies in inverse centimetres
        eigenvalues (np.ndarray): The probe eigenvalues (out-of-plane wavevectors) for all modes
            evaluated at the frequencies wavenumber
        Tdd_o (np.ndarray): Old value of the matrix T_dd
        Rud_o (np.ndarray): Old value of the matrix R_ud
        Rdu_o (np.ndarray): Old value of the matrix R_du
        Tuu_o (np.ndarray): Old value of the matrix T_uu
        ia (np.ndarray): Interface matrix for layer a, calculated automatically in creation
            of the Media class
        ib (np.ndarray):Interface matrix for layer b, calculated automatically in creation
            of the Media class
        thickness (float, optional): The current layer thickness, defaults to None indicating
            that the layer is the first or last in the stack

    Returns:
        np.ndarray: The matrix R_du evaluated at all frequencies in wavenumber

    """
    # Calculates the matrix dimension, this differs depending on whether
    # the calculation is local or nonlocal
    dim = Tuu_o.shape[-1]

    # Checks whether the interface is physical or just an artifact of multiple
    # entries for the same layer, in the latter case sets the interface matrix
    # to the identity
    if np.all(ia == ib):
        s0 = np.zeros((len(wavenumber), 2 * dim, 2 * dim))
        s0[:] = np.identity(2 * dim)
    else:
        s0 = interfacial_matrix(ia, ib)

    # If the layer has a finite thickness calculates the propagation matrices
    # and calculates their matrix product with the scattering matrix
    if thickness:
        lp, rp = propagation_matrices(wavenumber, eigenvalues, thickness)

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
        print("bell")

        idenmat = np.zeros((len(wavenumber), dim, dim))
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
    wavenumber: np.ndarray,
    eigenvalues: np.ndarray,
    Tdd_o: np.ndarray,
    Rud_o: np.ndarray,
    Rdu_o: np.ndarray,
    Tuu_o: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    thickness: float = None,
) -> np.ndarray:
    """Calculates the interface matrix R_{ud}.

    This matrix is defined in Eq. ... of the paper doi or in the projects online
    documentation .

    Args:
        wavenumber (np.ndarray): The probe frequencies in inverse centimetres
        eigenvalues (np.ndarray): The probe eigenvalues (out-of-plane wavevectors) for all modes
            evaluated at the frequencies wavenumber
        Tdd_o (np.ndarray): Old value of the matrix T_dd
        Rud_o (np.ndarray): Old value of the matrix R_ud
        Rdu_o (np.ndarray): Old value of the matrix R_du
        Tuu_o (np.ndarray): Old value of the matrix T_uu
        ia (np.ndarray): Interface matrix for layer a, calculated automatically in creation
            of the Media class
        ib (np.ndarray):Interface matrix for layer b, calculated automatically in creation
            of the Media class
        thickness (float, optional): The current layer thickness, defaults to None indicating
            that the layer is the first or last in the stack

    Returns:
        np.ndarray: The matrix R_ud evaluated at all frequencies in wavenumber

    """
    # Calculates the matrix dimension, this differs depending on whether
    # the calculation is local or nonlocal
    dim = Tuu_o.shape[-1]

    # Checks whether the interface is physical or just an artifact of multiple
    # entries for the same layer, in the latter case sets the interface matrix
    # to the identity
    if np.all(ia == ib):
        s0 = np.zeros((len(wavenumber), 2 * dim, 2 * dim))
        s0[:] = np.identity(2 * dim)
    else:
        s0 = interfacial_matrix(ia, ib)

    # If the layer has a finite thickness calculates the propagation matrices
    # and calculates their matrix product with the scattering matrix
    if thickness:
        lp, rp = propagation_matrices(wavenumber, eigenvalues, thickness)
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

        idenmat = np.zeros((len(wavenumber), dim, dim))
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
    wavenumber: np.ndarray,
    eigenvalues: np.ndarray,
    Tdd_o: np.ndarray,
    Rud_o: np.ndarray,
    Rdu_o: np.ndarray,
    Tuu_o: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    thickness: float = None,
) -> np.ndarray:
    """Calculates the interface matrix T_{uu}.

    This matrix is defined in Eq. ... of the paper doi or in the projects online
    documentation .

    Args:
        wavenumber (np.ndarray): The probe frequencies in inverse centimetres
        eigenvalues (np.ndarray): The probe eigenvalues (out-of-plane wavevectors) for all modes
            evaluated at the frequencies wavenumber
        Tdd_o (np.ndarray): Old value of the matrix T_dd
        Rud_o (np.ndarray): Old value of the matrix R_ud
        Rdu_o (np.ndarray): Old value of the matrix R_du
        Tuu_o (np.ndarray): Old value of the matrix T_uu
        ia (np.ndarray): Interface matrix for layer a, calculated automatically in creation
            of the Media class
        ib (np.ndarray):Interface matrix for layer b, calculated automatically in creation
            of the Media class
        thickness (float, optional): The current layer thickness, defaults to None indicating
            that the layer is the first or last in the stack

    Returns:
        np.ndarray: The matrix T_uu evaluated at all frequencies in wavenumber

    """
    # Calculates the matrix dimension, this differs depending on whether
    # the calculation is local or nonlocal
    dim = Tuu_o.shape[-1]

    # Checks whether the interface is physical or just an artifact of multiple
    # entries for the same layer, in the latter case sets the interface matrix
    # to the identity
    if np.all(ia == ib):
        s0 = np.zeros((len(wavenumber), 2 * dim, 2 * dim))
        s0[:] = np.identity(2 * dim)
    else:
        s0 = interfacial_matrix(ia, ib)

    # If the layer has a finite thickness calculates the propagation matrices
    # and calculates their matrix product with the scattering matrix
    if thickness:
        lp, rp = propagation_matrices(wavenumber, eigenvalues, thickness)

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

        idenmat = np.zeros((len(wavenumber), dim, dim))
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
    wavenumber: np.ndarray,
    eigenvalues: np.ndarray,
    Tdd_o: np.ndarray,
    Rud_o: np.ndarray,
    Rdu_o: np.ndarray,
    Tuu_o: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    thickness: float = None,
) -> np.ndarray:
    """Calculates the interface matrix T_{dd}.

    This matrix is defined in Eq. ... of the paper doi or in the projects online
    documentation .

    Args:
        wavenumber (np.ndarray): The probe frequencies in inverse centimetres
        eigenvalues (np.ndarray): The probe eigenvalues (out-of-plane wavevectors) for all modes
            evaluated at the frequencies wavenumber
        Tdd_o (np.ndarray): Old value of the matrix T_dd
        Rud_o (np.ndarray): Old value of the matrix R_ud
        Rdu_o (np.ndarray): Old value of the matrix R_du
        Tuu_o (np.ndarray): Old value of the matrix T_uu
        ia (np.ndarray): Interface matrix for layer a, calculated automatically in creation
            of the Media class
        ib (np.ndarray):Interface matrix for layer b, calculated automatically in creation
            of the Media class
        thickness (float, optional): The current layer thickness, defaults to None indicating
            that the layer is the first or last in the stack

    Returns:
        np.ndarray: The matrix T_dd evaluated at all frequencies in wavenumber

    """
    # Calculates the matrix dimension, this differs depending on whether
    # the calculation is local or nonlocal
    dim = Tuu_o.shape[-1]

    # Checks whether the interface is physical or just an artifact of multiple
    # entries for the same layer, in the latter case sets the interface matrix
    # to the identity
    if np.all(ia == ib):
        s0 = np.zeros((len(wavenumber), 2 * dim, 2 * dim))
        s0[:] = np.identity(2 * dim)
    else:
        s0 = interfacial_matrix(ia, ib)

    # If the layer has a finite thickness calculates the propagation matrices
    # and calculates their matrix product with the scattering matrix
    if thickness:
        lp, rp = propagation_matrices(wavenumber, eigenvalues, thickness)

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

        idenmat = np.zeros((len(wavenumber), dim, dim))
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
    wavenumber: np.ndarray, eigenvalues: np.ndarray, thickness: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the propagation matrix.

    For a layer of defined thickness given that layer's eigenvalues
    (out-of-plane wavevectors) this returns the propagation matrix. This matrix is
    defined in ...

    Args:
        wavenumber (np.ndarray): The probe frequencies in inverse centimetres
        eigenvalues (np.ndarray): The probe eigenvalues (out-of-plane wavevectors) for all modes
            evaluated at the frequencies wavenumber
        thickness (float, optional): The current layer thickness, defaults to None indicating
            that the layer is the first or last in the stack

    Returns:
        tuple: containing the two propagation matrices as defined in Formulation and comparison
        of two recursive matrix algorithms for modeling layered diffraction gratings
    """
    # Calculates the eigenvalue dimensions, this will differ depending on
    # whether a nonlocal or local calculation is performed
    dim = eigenvalues.shape[-1] // 2
    conv = 2 * np.pi / 0.01  # Factor to convert from inverse cm to Hz

    # Calculates the diagonal propagation vectors from the complex eigenvalues
    diag_vecr = np.exp(
        (1j * eigenvalues.real - np.abs(eigenvalues.imag))
        * wavenumber[:, None]
        * conv
        * thickness
    )

    diag_vecl = np.exp(
        (-1j * eigenvalues.real - np.abs(eigenvalues.imag))
        * wavenumber[:, None]
        * conv
        * thickness
    )

    # Initialise the two propagation matrices
    propmat1 = np.zeros((len(wavenumber), 2 * dim, 2 * dim), dtype=complex)
    propmat2 = np.zeros((len(wavenumber), 2 * dim, 2 * dim), dtype=complex)

    for idx in range(dim):
        propmat1[:, idx, idx] = 1
        propmat1[:, idx + dim, idx + dim] = diag_vecl[:, idx + dim]
        propmat2[:, idx, idx] = diag_vecr[:, idx]
        propmat2[:, idx + dim, idx + dim] = 1

    return propmat1, propmat2


def interfacial_matrix(ia: np.ndarray, ib: np.ndarray,) -> np.ndarray:
    """Calculates the interfacial matrix.

    For two layers with individual interface matrices ia and ib returns the interface
    matrix

    Args:
        ia (np.ndarray): Interface matrix for layer a, calculated automatically in creation
            of the Media class
        ib (np.ndarray):Interface matrix for layer b, calculated automatically in creation
            of the Media class

    Returns:
        np.ndarray: The interfacial matrix between layer 0 and layer 1
    """
    # Tries to directly invert and if not uses the pseudoinverse
    try:
        t = np.einsum("ijk,ikl->ijl", np.linalg.inv(ib), ia)
    except np.linalg.LinAlgError as err:
        if "Singular matrix" in str(err):
            t = np.einsum("ijk,ikl->ijl", np.linalg.pinv(ib), ia)

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
