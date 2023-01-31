"""Core scattering matrix algorithm."""
import numpy as np
import numpy.typing as npt
from nonlocal_smms.types.calculations import Locality
from nonlocal_smms.types.structures import Heterostructure
from nonlocal_smms.scattering_matrix.helpers import Rdu
from nonlocal_smms.scattering_matrix.helpers import Rud
from nonlocal_smms.scattering_matrix.helpers import Tdd
from nonlocal_smms.scattering_matrix.helpers import Tuu


def scattering_matrix_method(
    angles: npt.NDArray[np.float64],
    frequencies: npt.NDArray[np.float64],
    heterostructure: Heterostructure,
    flavour: Locality
    ) -> npt.NDArray[np.complex128]:
    """Carry out the scattering matrix calculation.

    Args:
        angles (npt.NDArray[np.float64]): Incident angle sin radians
        frequencies (npt.NDArray[np.float64]): Array of frequencies
        heterostructure (Heterostructure): Heterostructure to simulate
        flavour (Locality): Which calculation to run

    Returns:
        npt.NDArray[np.complex128]: Output transfer matrix for full heterostructure.
    """
    V = np.shape(frequencies)[0]
    W = np.shape(frequencies)[2]

    zeta = np.sin(angles) * heterostructure.superstrate.refractive_index(
        frequencies
    )

    match flavour:
        case Locality.Local:
            dimension = 4
        case Locality.Nonlocal:
            dimension = 10

    half_dimension = dimension // 2


    S_matrix = np.zeros((V, 1, W, dimension, dimension), dtype=np.complex128)
    S_matrix[:, :, :] = np.diag(np.ones(dimension), dtype=np.complex128)

    # Assign the layer by layer 2x2 block scattering matrices
    # In the top layer these are just defined from the pre-initialised S-matrix
    Rud1 = S_matrix[:, :, :, :half_dimension, half_dimension:]
    Tdd1 = S_matrix[:, :, :, half_dimension:, half_dimension:]
    Rdu1 = S_matrix[:, :, :, half_dimension:, :half_dimension]
    Tuu1 = S_matrix[:, :, :, :half_dimension, :half_dimension]

    scattering_matrix_iterator = heterostructure.scattering_matrix_iterator(flavour)

    for (thickness_m, eigenvalues, interface_matrices) in scattering_matrix_iterator:
        Rud0, Rdu0, Tdd0, Tuu0 = (
            Rud1,
            Rdu1,
            Tdd1,
            Tuu1,
        )  # Update the scattering block matrices

        eigenvalues_m = eigenvalues[0](frequencies, zeta)
        eigenvalues_n = eigenvalues[1](frequencies, zeta)
        interface_matrix_m = interface_matrices[0](frequencies, zeta, eigenvalues_m)
        interface_matrix_n = interface_matrices[1](frequencies, zeta, eigenvalues_n)

        Rud1 = Rud(
            frequencies,
            eigenvalues_m,
            Tdd0,
            Rud0,
            Rdu0,
            Tuu0,
            interface_matrix_m,
            interface_matrix_n,
            thickness_m,
        )
        Tdd1 = Tdd(
            frequencies,
            eigenvalues_m,
            Tdd0,
            Rud0,
            Rdu0,
            Tuu0,
            interface_matrix_m,
            interface_matrix_n,
            thickness_m,
        )
        Rdu1 = Rdu(
            frequencies,
            eigenvalues_m,
            Tdd0,
            Rud0,
            Rdu0,
            Tuu0,
            interface_matrix_m,
            interface_matrix_n,
            thickness_m,
        )
        Tuu1 = Tuu(
            frequencies,
            eigenvalues_m,
            Tdd0,
            Rud0,
            Rdu0,
            Tuu0,
            interface_matrix_m,
            interface_matrix_n,
            thickness_m,
        )

    return Rud1, Rdu1, Tdd1, Tuu1
