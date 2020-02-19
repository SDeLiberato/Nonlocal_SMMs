"""Core scattering matrix function."""

__all__ = ["scattering_matrix"]

import os
from itertools import product
from typing import Tuple

from tinydb import TinyDB, Query
import numpy as np
from nonlocal_smms.materials import Media
from nonlocal_smms.helper import Rdu, Rud, Tdd, Tuu


# flake8: noqa: C901
def scattering_matrix(
    wavenumber: np.ndarray,
    heterostructure: list,
    orientation: str = "c-cut",
    angle: float = None,
    wavevector: np.ndarray = None,
    locality: str = "nonlocal",
    parameters: dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the scattering matrix as presented in the paper arXiv.

    Args:
        wavenumber (np.ndarray): Input wavenumbers in inverse centimetres
        heterostructure (list): Each element is length 2 list whose first element
            contains a string corresponding to a material name and whose second
            element is the layer thickness
        orientation (str): The considered crystal orientation, either 'a-cut' or 'c-cut'
        angle (float, optional): Incident angle in degrees, defaults to None.
        wavevector (np.ndarray, optional): Incident wavevectors in inverse centimetre,
            defaults to None.
        locality (str, optional): Whether to do a local calculation or not, defaults
            to 'nonlocal' meaning a nonlocal calculation is carried out
        parameters (dict, optional): Parameters to overwrite the default values in the
            material library, defaults to none

    Returns:
        tuple: First (second) elements correspond to TE (TM) polarised reflectance.

    Raises:
        ValueError: If neither an angle or wavevector is passed
        ValueError: If both an angle and wavevector is passed

    """

    if angle is None and wavevector is None:
        raise ValueError("Either an angle or wavevector array must be passed")
    elif angle is not None and wavevector is not None:
        raise ValueError("Only an angle or wavevector array must be passed")

    # Create a list of the unique materials comprising heterostructure
    mats = list(set([row[0] for row in heterostructure]))
    material_data = dict()
    if angle:
        angler = np.deg2rad(angle)  # Convert the incident anglele to degrees

    """ Iterates through the heterostructurestructure materials, adding the corresponding
    properties to material_data
    """
    material_db = TinyDB(os.path.dirname(os.path.abspath(__file__)) + "/materials.json")
    query = Query()
    for mat in mats:
        try:
            properties = material_db.search(query.material == mat)
            if properties == []:
                raise NameError
            properties = properties[0]
        except NameError:
            print("The material " + mat + " is not in the material database")
            raise

        """ Checks for custom parameters, passed in the parameters dict, and if present
        updates the material properties accordingly
        """
        if parameters:
            if mat in parameters:
                for key, val in parameters[mat].items():
                    properties[key] = val
        # Call the Media class with the appropriate in-plane argument
        if angle:
            material_data[mat] = Media(
                mat, properties, wavenumber, angle=angle, orientation=orientation
            )
        elif wavevector is not None:
            material_data[mat] = Media(
                mat,
                properties,
                wavenumber,
                wavevector=wavevector,
                orientation=orientation,
            )

    # Initialises the problem by Eq. 48 in the local and nonlocal cases
    if locality == "local":
        S = np.zeros((3, len(wavenumber), 4, 4))
        S[0, :] = np.diag(np.ones(4))
    else:
        S = np.zeros((3, len(wavenumber), 10, 10))
        S[0, :] = np.diag(np.ones(10))

    dim = S.shape[2] // 2  # Calculates the dimension of the 4 block matrices

    if angle:
        zeta = np.zeros_like(wavenumber)
        zeta[:] = np.sin(angle)
        wavenumber_p = wavenumber
    else:
        arrays = [(wavenumber), (wavevector)]
        arr = np.array(list(product(*arrays)))
        zeta = arr[:, 1] / arr[:, 0]
        wavenumber_p = arr[:, 0] / np.ones_like(arr[:, 1])

    Rud0 = np.zeros((len(zeta), dim, dim), dtype=complex)
    Tdd0 = np.zeros((len(zeta), dim, dim), dtype=complex)
    Tuu0 = np.zeros((len(zeta), dim, dim), dtype=complex)
    Rdu0 = np.zeros((len(zeta), dim, dim), dtype=complex)

    Rud1 = np.zeros((len(zeta), dim, dim), dtype=complex)
    Tdd1 = np.zeros((len(zeta), dim, dim), dtype=complex)
    Tuu1 = np.zeros((len(zeta), dim, dim), dtype=complex)
    Rdu1 = np.zeros((len(zeta), dim, dim), dtype=complex)

    # Iterate through the heterostructure object
    for idx, layer in enumerate(heterostructure[:-1]):
        """ In the first layer idx=0 the problem is initialised by Eq. 48.
        In other layers, idx>0, the problem is seeded by the result
        in layer idx-1 """
        if idx == 0:
            Rud0 = S[0, :, :dim, dim:]
            Tdd0 = S[0, :, dim:, dim:]
            Rdu0 = S[0, :, dim:, :dim]
            Tuu0 = S[0, :, :dim:, :dim]
        else:
            Rud0, Rdu0, Tdd0, Tuu0 = Rud1, Rdu1, Tdd1, Tuu1

        mat = layer[0]  # Gets the material name in the current layer
        thickness = layer[1]  # Gets the current layer thickness

        """ Look up the material properties in the material_data dict,
        and assign the eigenvalues and interface matrix to eigsm
        and imatm respectively
        """
        materialm = material_data[mat]
        eigsm = materialm.eigs
        imatm = materialm.imat

        """ Look up the material properites in the material_data dict for
        layer n = m + 1, and assign the layers interface matrix to imatn
        """
        materialn = material_data[heterostructure[idx + 1][0]]
        imatn = materialn.imat
        if not angle:
            eigsm = np.reshape(eigsm, (len(zeta), 10), order="F")
            imatm = np.reshape(imatm, (len(zeta), 10, 10), order="F")
            imatn = np.reshape(imatn, (len(zeta), 10, 10), order="F")

        # If doing a local calculation, discard the nonlocal modes
        if locality == "local":
            eigsm = np.concatenate([eigsm[:, 0:2], eigsm[:, 5:7]], axis=1)
            imatm = np.concatenate([imatm[:, :4, :2], imatm[:, :4, 5:7]], axis=2)
            imatn = np.concatenate([imatn[:, :4, :2], imatn[:, :4, 5:7]], axis=2)

        # Calculate the scattering matrices
        Rud1 = Rud(wavenumber_p, eigsm, Tdd0, Rud0, Rdu0, Tuu0, imatm, imatn, thickness)
        Tdd1 = Tdd(wavenumber_p, eigsm, Tdd0, Rud0, Rdu0, Tuu0, imatm, imatn, thickness)
        Rdu1 = Rdu(wavenumber_p, eigsm, Tdd0, Rud0, Rdu0, Tuu0, imatm, imatn, thickness)
        Tuu1 = Tuu(wavenumber_p, eigsm, Tdd0, Rud0, Rdu0, Tuu0, imatm, imatn, thickness)

    output = Rdu1[:, 0, 0], Rdu1[:, 1, 1]
    if angle:
        return output
    else:
        outer = list()
        for o in output:
            temp = np.reshape(o, (len(wavevector), len(wavenumber)), order="F")
            outer.append(temp)

        return outer[0], outer[1]
