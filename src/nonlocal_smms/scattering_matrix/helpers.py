"""Helper functions for the scattering matrix algorithm.

These functions are broadly as defined in "Formulation and
comparison of two recursive matrix algorithms for modeling layered diffraction
gratings".
"""
from typing import Tuple

import numpy as np
import numpy.typing as npt
from nonlocal_smms.constants import SPEED_OF_LIGHT
from nonlocal_smms.constants import TO_METRES


def Rdu(
    frequency: npt.NDArray[np.float64],
    eigenvalues: npt.NDArray[np.complex128],
    Tdd_o: npt.NDArray[np.complex128],
    Rud_o: npt.NDArray[np.complex128],
    Rdu_o: npt.NDArray[np.complex128],
    Tuu_o: npt.NDArray[np.complex128],
    Wm: npt.NDArray[np.complex128],
    Wn: npt.NDArray[np.complex128],
    thickness: npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    """Calculates the interface matrix R_{du}.

    This matrix is the block-matrix defined in Eq. 15a of
    doi: 10.1364/JOSAA.13.001024. This is a 2x2 block of
    the full 4x4 scattering matrix.

    Args:
        frequency (npt.NDArray[np.float64]): Driving frequency in rad / s
        eigenvalues (npt.NDArray[np.complex128]): Sorted Eigenvalues in layer ??
        Tdd_o (npt.NDArray[np.complex128]): Old value of the matrix T_dd
        Rud_o (npt.NDArray[np.complex128]): Old value of the matrix R_ud
        Rdu_o (npt.NDArray[np.complex128]): Old value of the matrix R_du
        Tuu_o (npt.NDArray[np.complex128]): Old value of the matrix T_uu
        Wm (npt.NDArray[np.complex128]): Interface matrix for layer a
        Wn (npt.NDArray[np.complex128]):Interface matrix for layer b
        thickness (npt.NDArray[np.float64]): The current layer thickness

    Returns:
        npt.NDArray[np.complex128]: The matrix R_du evaluated at all frequencies

    """
    V = np.shape(frequency)[0]  # We need to unpack separately to
    W = np.shape(frequency)[2]  # executre without eager
    half_dimension = np.shape(Tdd_o)[-1]
    s0 = interface_matrix(Wm, Wn)

    # If the layer has a finite thickness calculates the propagation matrices
    # and calculates their matrix product with the scattering matrix
    if thickness != 0.0:
        phi_l, phi_r = propagation_matrices(frequency, eigenvalues, thickness)
        s = np.matmul(phi_l, np.matmul(s0, phi_r))
    else:
        s = s0

    # Checks whether it is necessary to do the matrix multiplication, in the
    # first layer Rdu_0 == 0 so we can return directly
    if np.all(Rdu_o == 0):
        rdut = s0[:, :, :, half_dimension:, :half_dimension]
        return rdut
    else:
        rdut = s[:, :, :, half_dimension:, :half_dimension]

        idenmat = np.zeros(
            (V, 1, W, half_dimension, half_dimension), dtype=np.complex128
        )
        idenmat[:, :, :] = np.diag(np.ones(half_dimension, dtype=np.complex128))

        # Try to invert the matrix and if not calculate the pseudoinverse
        try:
            imat = np.linalg.inv(idenmat - np.matmul(Rud_o, rdut))
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                print("err")
                imat = np.linalg.pinv(idenmat - np.matmul(Rud_o, rdut))

        tm1 = np.matmul(Tdd_o, rdut)
        tm2 = np.matmul(imat, Tuu_o)

        return Rdu_o + np.matmul(tm1, tm2)


def Rud(
    frequency: npt.NDArray[np.float64],
    eigenvalues: npt.NDArray[np.complex128],
    Tdd_o: npt.NDArray[np.complex128],
    Rud_o: npt.NDArray[np.complex128],
    Rdu_o: npt.NDArray[np.complex128],
    Tuu_o: npt.NDArray[np.complex128],
    Wm: npt.NDArray[np.complex128],
    Wn: npt.NDArray[np.complex128],
    thickness: npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    """Calculates the interface matrix R_{ud}.

    This matrix is the block-matrix defined in Eq. 15a of
    doi: 10.1364/JOSAA.13.001024. This is a 2x2 block of
    the full 4x4 scattering matrix.

    Args:
        frequency (npt.NDArray[np.float64]): Probe frequencies in rad / s
        eigenvalues (npt.NDArray[np.complex128]): Ordered eigenvalues
        Tdd_o (npt.NDArray[np.complex128]): Old value of the matrix T_dd
        Rud_o (npt.NDArray[np.complex128]): Old value of the matrix R_ud
        Rdu_o (npt.NDArray[np.complex128]): Old value of the matrix R_du
        Tuu_o (npt.NDArray[np.complex128]): Old value of the matrix T_uu
        Wm (npt.NDArray[np.complex128]): Interface matrix for layer m
        Wn (npt.NDArray[np.complex128]):Interface matrix for layer n
        thickness (npt.NDArray[np.float64]): The current layer thickness

    Returns:
        npt.NDArray[np.complex128]: The matrix R_ud evaluated at all frequencies

    """
    V = np.shape(frequency)[0]  # We need to unpack separately to
    W = np.shape(frequency)[2]  # execute without eager
    half_dimension = np.shape(Tdd_o)[-1]
    s0 = interface_matrix(Wm, Wn)

    # If the layer has a finite thickness calculates the propagation matrices
    # and calculates their matrix product with the scattering matrix
    if thickness != 0.0:
        phi_l, phi_r = propagation_matrices(frequency, eigenvalues, thickness)
        s = np.matmul(phi_l, np.matmul(s0, phi_r))
    else:
        s = s0

    # Checks whether it is necessary to do the matrix multiplication, in the
    # first layer Rdu_0 == 0 so we can return directly
    if np.all(Rud_o == 0):
        rudt = s0[:, :, :, :half_dimension, half_dimension:]
        return rudt
    else:
        rudt = s[:, :, :, :half_dimension, half_dimension:]
        rdut = s[:, :, :, half_dimension:, :half_dimension]
        tuut = s[:, :, :, :half_dimension, :half_dimension]
        tddt = s[:, :, :, half_dimension:, half_dimension:]

        idenmat = np.zeros(
            (V, 1, W, half_dimension, half_dimension), dtype=np.complex128
        )
        idenmat[:, :, :] = np.diag(np.ones(half_dimension, dtype=np.complex128))

        # Try to invert the matrix and if not calculate the pseudoinverse
        try:
            imat = np.linalg.inv(idenmat - np.matmul(rdut, Rud_o))
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                print(err)
                imat = np.linalg.pinv(idenmat - np.matmul(rdut, Rud_o))

        tm1 = np.matmul(tuut, Rud_o)
        tm2 = np.matmul(imat, tddt)
        return rudt + np.matmul(tm1, tm2)


def Tuu(
    frequency: npt.NDArray[np.float64],
    eigenvalues: npt.NDArray[np.complex128],
    Tdd_o: npt.NDArray[np.complex128],
    Rud_o: npt.NDArray[np.complex128],
    Rdu_o: npt.NDArray[np.complex128],
    Tuu_o: npt.NDArray[np.complex128],
    Wm: npt.NDArray[np.complex128],
    Wn: npt.NDArray[np.complex128],
    thickness: npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    """Calculates the interface matrix T_{uu}.

    This matrix is the block-matrix defined in Eq. 15a of
    doi: 10.1364/JOSAA.13.001024. This is a 2x2 block of
    the full 4x4 scattering matrix.

    Args:
        frequency (npt.NDArray[np.float64]): Frequencies in rad / s
        eigenvalues (npt.NDArray[np.complex128]): Sorted eigenvalues at frequency
        Tdd_o (npt.NDArray[np.complex128]): Old value of the matrix T_dd
        Rud_o (npt.NDArray[np.complex128]): Old value of the matrix R_ud
        Rdu_o (npt.NDArray[np.complex128]): Old value of the matrix R_du
        Tuu_o (npt.NDArray[np.complex128]): Old value of the matrix T_uu
        Wm (npt.NDArray[np.complex128]): Interface matrix for layer m
        Wn (npt.NDArray[np.complex128]):Interface matrix for layer n
        thickness (npt.NDArray[np.float64]): The current layer thickness

    Returns:
        npt.NDArray[np.complex128]: The matrix T_uu evaluated at all frequencies

    """
    V = np.shape(frequency)[0]  # We need to unpack separately to
    W = np.shape(frequency)[2]  # executre without eager
    half_dimension = np.shape(Tdd_o)[-1]
    s0 = interface_matrix(Wm, Wn)

    # If the layer has a finite thickness calculates the propagation matrices
    # and calculates their matrix product with the scattering matrix
    if thickness != 0.0:
        phi_l, phi_r = propagation_matrices(frequency, eigenvalues, thickness)
        s = np.matmul(phi_l, np.matmul(s0, phi_r))
    else:
        s = s0

    # Checks whether it is necessary to do the matrix multiplication, in the
    # first layer Rdu_0 == 0 so we can return directly
    if np.all(Rud_o == 0):
        tuut = s[:, :, :, :half_dimension, :half_dimension]
        return np.matmul(tuut, Tuu_o)
    else:
        rdut = s[:, :, :, half_dimension:, :half_dimension]
        tuut = s[:, :, :, :half_dimension, :half_dimension]


        idenmat = np.zeros(
            (V, 1, W, half_dimension, half_dimension), dtype=np.complex128
        )
        idenmat[:, :, :] = np.diag(np.ones(half_dimension, dtype=np.complex128))

        # Tries to invert the matrix and if not finds the pseudoinverse
        try:
            imat = np.linalg.inv(idenmat - np.matmul(Rud_o, rdut))
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                print(err)
                imat = np.linalg.pinv(idenmat - np.matmul(Rud_o, rdut))

        tm1 = np.matmul(tuut, imat)

        return np.matmul(tm1, Tuu_o)


def Tdd(
    frequency: npt.NDArray[np.float],
    eigenvalues: npt.NDArray[np.complex128],
    Tdd_o: npt.NDArray[np.complex128],
    Rud_o: npt.NDArray[np.complex128],
    Rdu_o: npt.NDArray[np.complex128],
    Tuu_o: npt.NDArray[np.complex128],
    Wm: npt.NDArray[np.complex128],
    Wn: npt.NDArray[np.complex128],
    thickness: npt.NDArray[np.float],
) -> npt.NDArray[np.complex128]:
    """Calculates the interface matrix T_{dd}.

    This matrix is the block-matrix defined in Eq. 15a of
    doi: 10.1364/JOSAA.13.001024. This is a 2x2 block of
    the full 4x4 scattering matrix.

    Args:
        frequency (npt.NDArray[np.float64]): Frequencies in rad/s
        eigenvalues (npt.NDArray[np.complex128]): Sorted eigenvalues at frequency
        Tdd_o (npt.NDArray[np.complex128]): Old value of the matrix T_dd
        Rud_o (npt.NDArray[np.complex128]): Old value of the matrix R_ud
        Rdu_o (npt.NDArray[np.complex128]): Old value of the matrix R_du
        Tuu_o (npt.NDArray[np.complex128]): Old value of the matrix T_uu
        Wm (npt.NDArray[np.complex128]): Interface matrix for layer m
        Wn (npt.NDArray[np.complex128]): Interface matrix for layer n
        thickness (npt.NDArray[np.float64]): The current layer thickness

    Returns:
        npt.NDArray[np.complex128]: The matrix T_dd evaluated at all frequencies

    """
    V = np.shape(frequency)[0]  # We need to unpack separately to
    W = np.shape(frequency)[2]  # executre without eager
    half_dimension = np.shape(Tdd_o)[-1]
    s0 = interface_matrix(Wm, Wn)

    # If the layer has a finite thickness calculates the propagation matrices
    # and calculates their matrix product with the scattering matrix
    if thickness != 0.0:
        phi_l, phi_r = propagation_matrices(frequency, eigenvalues, thickness)
        s = np.matmul(phi_l, np.matmul(s0, phi_r))
    else:
        s = s0

    # Checks whether it is necessary to do the matrix multiplication, in the
    # first layer Rdu_0 == 0 so we can return directly
    if np.all(Rud_o == 0):
        tddt = s[:, :, :, half_dimension:, half_dimension:]
        return np.matmul(Tdd_o, tddt)
    else:
        rdut = s[:, :, :, half_dimension:, :half_dimension]
        tddt = s[:, :, :, half_dimension:, half_dimension:]

        idenmat = np.zeros(
            (V, 1, W, half_dimension, half_dimension), dtype=np.complex128
        )
        idenmat[:, :, :] = np.diag(np.ones(half_dimension, dtype=np.complex128))

        # Tries to invert matrix and if not calculates the pseudoinverse
        try:
            imat = np.linalg.inv(idenmat - np.matmul(rdut, Rud_o))
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                print(err)
                imat = np.linalg.pinv(idenmat - np.matmul(rdut, Rud_o))

        tm1 = np.matmul(Tdd_o, imat)

        return np.matmul(tm1, tddt)


def propagation_matrices(
    frequency: npt.NDArray[np.float64],
    eigenvalues: npt.NDArray[np.complex128],
    thickness: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculates the propagation matrix.

    For a layer of defined thickness given that layer's eigenvalues
    (out-of-plane wavevectors) this returns the propagation matrix. These
    matrices are defined as the left and right components in Eq. 13a
    of doi 10.1364/JOSAA.13.001024

    Args:
        frequency (npt.ndarray[np.float64]): Probe frequencies in rad / s
        eigenvalues (npt.ndarray[np.complex128]): Sorted eigenvalues for all modes
            evaluated at frequency
        thickness (npt.NDArray[np.float64]): Current layer thickness in nm

    Returns:
        Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
            containing the two propagation matrices as defined in "Formulation
            and comparison of two recursive matrix algorithms for modeling layered
            diffraction gratings"
    """
    V = np.shape(frequency)[0]  # We need to unpack separately to
    W = np.shape(frequency)[2]  # executre without eager
    dimension = np.shape(eigenvalues)[-1]
    half_dimension = dimension // 2

    # Calculates the diagonal propagation vectors from the complex eigenvalues
    # to calculate the stack needs to be in the first axis, so we swap
    tp_eigenvalues = np.transpose(eigenvalues, axes=[3, 0, 1, 2])
    diag_vec_upward = np.exp(
        1j
        * tp_eigenvalues[:half_dimension, :, :, :]
        * frequency
        / SPEED_OF_LIGHT
        * (thickness * TO_METRES)
    )
    diag_vec_downward = np.exp(
        - 1j
        * tp_eigenvalues[half_dimension:, :, :, :]  # ???
        * frequency
        / SPEED_OF_LIGHT
        * (thickness * TO_METRES)
    )
    # and then swap back ...
    diag_vec_upward = np.transpose(diag_vec_upward, axes=[1, 2, 3, 0])
    diag_vec_downward = np.transpose(diag_vec_downward, axes=[1, 2, 3, 0])

    propmatl_upper = np.zeros((V, 1, W, half_dimension, dimension), dtype=np.complex128)
    propmatl_upper[:, :, :] = np.concatenate(
        [
            np.diag(np.ones(half_dimension), dtype=np.complex128),
            np.zeros((half_dimension, half_dimension), dtype=np.complex128)
        ],
        axis=1
    )

    propmatl_21 = np.zeros(
        (V, 1, W, half_dimension, half_dimension), dtype=np.complex128
    )
    propmatl_22 = np.zeros(
        (V, 1, W, half_dimension, half_dimension), dtype=np.complex128
    )

    # fill the last diagonal block with the vector
    # in tensorflow we have set_diag, is there a np way to do the same?
    for idx in range(V * W):
        propmatl_22[idx // W, 0, idx // V] = np.diag(
            diag_vec_downward[idx // W, 0, idx // V]
        )

    propmatl_lower = np.concatenate([propmatl_21, propmatl_22], axis=4)  # 2)
    propmatl = np.concatenate([propmatl_upper, propmatl_lower], axis=3)  # 1)

    propmatr_upper = np.zeros((V, 1, W, 2, 4), dtype=np.complex128)
    # fill the last diagonal block with the vector
    # in tensorflow we have set_diag, is there a np way to do the same?
    for idx in range(V * W):
        propmatr_upper[idx // W, 0, idx // V] = np.concatenate(
            [
                np.diag(diag_vec_downward[idx // W, 0, idx // V]),
                np.zeros((2, 2), dtype=np.complex128)
            ],
            axis=4
        )

    propmatr_21 = np.zeros((V, 1, W, 2, 2), dtype=np.complex128)
    propmatr_22 = np.zeros((V, 1, W, 2, 2), dtype=np.complex128)
    propmatr_22[:, :, :] = np.diag(np.ones(2, dtype=np.complex128))

    propmatr_lower = np.concatenate([propmatr_21, propmatr_22], axis=4)  # 2)
    propmatr = np.concatenate([propmatr_upper, propmatr_lower], axis=3)  # 1)

    return propmatl, propmatr


def interface_matrix(
    Wm: npt.NDArray[np.complex128],
    Wn: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Calculates the interfacial matrix.

    For two layers with individual interface matrices Wm and Wn returns the interface
    matrix

    Args:
        Wm (npt.NDArray[np.complex128]): Interface matrix for layer m
        Wn (npt.NDArray[np.complex128]):Interface matrix for layer n

    Returns:
        npt.NDArray[np.complex128]: The interfacial matrix between layer m and layer n
    """
    # Tries to directly invert and if not uses the pseudoinverse

    try:
        t = np.matmul(np.linalg.inv(Wn), Wm)
    except np.linalg.LinAlgError as err:
        if "Singular matrix" in str(err):
            t = np.matmul(np.linalg.pinv(Wn), Wm)

    half_dimension = np.shape(t)[-1]

    t11 = t[:, :, :, :half_dimension, :half_dimension]
    t12 = t[:, :, :, :half_dimension, half_dimension:]
    t21 = t[:, :, :, half_dimension:, :half_dimension]
    t22 = t[:, :, :, half_dimension:, half_dimension:]

    try:
        inv_t22 = np.linalg.inv(t22)
    except np.linalg.LinAlgError as err:
        if "Singular matrix" in str(err):
            inv_t22 = np.linalg.pinv(t22)

    # Calculates the 4 blocks comprising s1
    s11 = t11 - np.matmul(t12, np.matmul(inv_t22, t21))
    s12 = np.matmul(t12, inv_t22)
    s21 = -np.matmul(inv_t22, t21)
    s22 = inv_t22

    s_upper = np.concatenate([s11, s12], axis=4)
    s_lower = np.concatenate([s21, s22], axis=4)

    return np.concatenate([s_upper, s_lower], axis=3)
