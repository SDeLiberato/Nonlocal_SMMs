"""Extracts observables from the scattering matrix method algorithm."""
from typing import Tuple

import numpy as np
import numpy.typing as npt
from nonlocal_smms.types.structures import Heterostructure
from nonlocal_smms.scattering_matrix.algorithm import scattering_matrix_method


def absorption(
    angles: npt.NDArray[np.float64],
    frequencies: npt.NDArray[np.float64],
    heterostructure: Heterostructure,
) -> npt.NDArray[np.float64]:
    """Calculate the absorption in Heterostructure.

    Args:
        angles (npt.NDArray[np.float64]): Incident angle in radiams
        frequencies (npt.NDArray[np.float64]): Frequencies to optimise over
        heterostructure (Heterostructure): Heterostructure to simulate

    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Absorption at
            frequency for (TM, TE) polarisation
    """
    # assume both are 1D tensors
    # Frequencies is (N_batch, 1)
    W = np.shape(frequencies)[0]  # How many frequencies (W), how many angles (X)?
    X = np.shape(angles)[0]  # How many frequencies (W), how many angles (X)?

    frequencies = np.expand_dims(frequencies, 2)
    angles = np.expand_dims(angles, 0)
    angles = np.expand_dims(angles, 0)

    frequencies = np.repeat(frequencies, X, axis=2)
    angles = np.repeat(angles, W, axis=0)

    _, Rdu, _, Tuu = scattering_matrix_method(angles, frequencies, heterostructure)
    r_tm = Rdu[:, :, :, 0, 0]
    r_te = Rdu[:, :, :, 1, 1]
    t_tm = Tuu[:, :, :, 0, 0]
    t_te = Tuu[:, :, :, 1, 1]

    reflectance_te = (np.abs(r_te)) ** 2
    reflectance_tm = (np.abs(r_tm)) ** 2

    # Snell's law assuming isotropic superstrate and substrate
    angles_substrate = np.asin(
        np.real(heterostructure.superstrate.refractive_index(frequencies))
        / np.real(heterostructure.substrate.refractive_index(frequencies))
        * np.sin(angles)
    )

    power_factor = (
        np.cos(angles_substrate)
        / np.cos(angles)
        * np.real(heterostructure.substrate.refractive_index(frequencies))
        / np.real(heterostructure.superstrate.refractive_index(frequencies))
    )

    transmittance_te = (np.abs(t_te)) ** 2 * power_factor

    transmittance_tm = (
        (np.abs(t_tm)) ** 2
        * power_factor
        / (np.cos(angles_substrate) / np.cos(angles)) ** 2
    )  # Why is this the case??

    absorption_te = (1 - reflectance_te - transmittance_te) * 100
    absorption_tm = (1 - reflectance_tm - transmittance_tm) * 100
    return absorption_tm, absorption_te


def reflectance(
    angles: npt.NDArray[np.float64],
    frequencies: npt.NDArray[np.float64],
    heterostructure: Heterostructure,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the reflectance from Heterostructure.

    Args:
        angles (npt.NDArray[np.float64]): Incident angle in radiams
        frequencies (npt.NDArray[np.float64]): Frequencies to optimise over
        heterostructure (Heterostructure): Heterostructure to simulate

    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Reflectance at
            frequencies for (TM, TE) polarisation
    """
    # assume both are 1D tensors
    W = np.shape(frequencies)[0]  # How many frequencies (W), how many angles (X)?
    X = np.shape(angles)[0]  # How many frequencies (W), how many angles (X)?

    frequencies = np.expand_dims(frequencies, 2)
    angles = np.expand_dims(angles, 0)
    angles = np.expand_dims(angles, 0)

    frequencies = np.repeat(frequencies, X, axis=2)
    angles = np.repeat(angles, W, axis=0)

    _, Rdu, *_ = scattering_matrix_method(angles, frequencies, heterostructure)
    r_tm = Rdu[:, :, :, 0, 0]
    r_te = Rdu[:, :, :, 1, 1]

    return (np.abs(r_tm)) ** 2 * 100, (np.abs(r_te)) ** 2 * 100


def transmission(
    angles: npt.NDArray[np.float64],
    frequencies: npt.NDArray[np.float64],
    heterostructure: Heterostructure,
) -> npt.NDArray[np.float64]:
    """Calculate the absorption in Heterostructure.

    Args:
        angles (npt.NDArray[np.float64]): Incident angle in radiams
        frequencies (npt.NDArray[np.float64]): Frequencies to optimise over
        heterostructure (Heterostructure): Heterostructure to simulate

    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Transmission at
            frequency for (TM, TE) polarisation
    """
    # assume both are 1D tensors
    W = np.shape(frequencies)[0]  # How many frequencies (W), how many angles (X)?
    X = np.shape(angles)[0]  # How many frequencies (W), how many angles (X)?

    frequencies = np.expand_dims(frequencies, 2)
    angles = np.expand_dims(angles, 0)
    angles = np.expand_dims(angles, 0)

    frequencies = np.repeat(frequencies, X, axis=2)
    angles = np.repeat(angles, W, axis=0)

    Rud, Rdu, Tdd, Tuu = scattering_matrix_method(angles, frequencies, heterostructure)
    t_tm = Tuu[:, :, :, 0, 0]
    t_te = Tuu[:, :, :, 1, 1]

    # Snell's law assuming isotropic superstrate and substrate
    angles_substrate = np.asin(
        np.real(heterostructure.superstrate.refractive_index(frequencies))
        / np.real(heterostructure.substrate.refractive_index(frequencies))
        * np.sin(angles)
    )

    power_factor = (
        np.cos(angles_substrate)
        / np.cos(angles)
        * np.real(heterostructure.substrate.refractive_index(frequencies))
        / np.real(heterostructure.superstrate.refractive_index(frequencies))
    )

    transmittance_te = (np.abs(t_te)) ** 2 * power_factor

    transmittance_tm = (
        (np.abs(t_tm)) ** 2
        * power_factor
        / (np.cos(angles_substrate) / np.cos(angles)) ** 2
    )  # Why is this the case??

    return transmittance_tm * 100.0, transmittance_te * 100.0
