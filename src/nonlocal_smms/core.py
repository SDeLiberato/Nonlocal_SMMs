__all__ = ["scattering_matrix"]

import os

from tinydb import TinyDB, Query
import numpy as np
from nonlocal_smms.materials import Media
from nonlocal_smms.helper import Rdu, Rud, Tdd, Tuu

# flake8: noqa: C901
def scattering_matrix(
    wn,
    hetero,
    orientation="c-cut",
    ang=None,
    kx=None,
    loc=None,
    params=None,
    return_inter=False,
):
    """ Calculates the scattering matrix as presented in the paper arXiv

    Parameters
    ----------
    wn: 1D numpy array
        Input wavenumbers in inverse centimetres
    hetero: list
        Ordered list. Each element is length 2 list whose first element
        contains a string corresponding to a material name and whose second
        element is the layer thickness
    orientation: string
        The considered crystal orientation, either a-cut or c-cut
    ang: float, optional
        Incident angle in degrees, defaults to None. Either this or kx must
        be passed to the function
    kx: 1D numpy array, optional
        Incident wavevectors in inverse centimetre, defaults to None. Either
        this or ang must be passed to the function
    loc: bool, optional
        Whether to do a local calculation or not, defaults to None meaning a
        nonlocal calculation is carried out
    params: dict, optional
        Parameters to overwrite the default values in the material library,
        efaults to none
    return_inter: bool, optional
        Whether to return intermediate results, useful for trobuleshooting.

    Returns
    -------

    tuple:
        First (second) elements correspond to TE (TM) polarised reflectance.
    """

    if ang is None and kx is None:
        raise Exception("Either an angle ang or wavevector array kx must be passed")
    # Create a list of the unique materials comprising hetero
    mats = list(set([row[0] for row in hetero]))
    material_data = dict()
    if ang:
        angr = np.deg2rad(ang)  # Convert the incident angle to degrees

    """ Iterates through the heterostructure materials, adding the corresponding
    properties to material_data
    """
    material_db = TinyDB(os.path.dirname(os.path.abspath(__file__)) + "/materials.json")
    query = Query()
    for mat in mats:
        try:
            props = material_db.search(query.material == mat)
            if props == []:
                raise NameError
            props = props[0]
        except NameError:
            print("The material " + mat + " is not in the material database")
            raise

        """ Checks for custom parameters, passed in the params dict, and if present
        updates the material properties accordingly
        """
        if params:
            if mat in params:
                for key, val in params[mat].items():
                    props[key] = val
        # Call the Media class with the appropriate in-plane argument
        if ang:
            material_data[mat] = Media(mat, props, wn, ang=angr, orient=orientation)
        elif kx:
            material_data[mat] = Media(mat, props, wn, kx=kx, orient=orientation)

    # Initialises the problem by Eq. 48 in the local and nonlocal cases
    if loc:
        S = np.zeros((3, len(wn), 4, 4))
        S[0, :] = np.diag(np.ones(4))
    else:
        S = np.zeros((3, len(wn), 10, 10))
        S[0, :] = np.diag(np.ones(10))

    dim = S.shape[2] // 2  # Calculates the dimension of the 4 block matrices

    # creates placeholders for the intermediate objects
    if return_inter:
        tlen = S.shape[1]
        nlay = len(hetero)
        Rud_ret = np.zeros((nlay, tlen, dim, dim), dtype=complex)
        Tdd_ret = np.zeros((nlay, tlen, dim, dim), dtype=complex)
        Tuu_ret = np.zeros((nlay, tlen, dim, dim), dtype=complex)
        Rdu_ret = np.zeros((nlay, tlen, dim, dim), dtype=complex)

        Rud0 = np.zeros((nlay, tlen, dim, dim), dtype=complex)
        Tdd0 = np.zeros((nlay, tlen, dim, dim), dtype=complex)
        Tuu0 = np.zeros((nlay, tlen, dim, dim), dtype=complex)
        Rdu0 = np.zeros((nlay, tlen, dim, dim), dtype=complex)

    # Iterate through the hetero object
    for idx, layer in enumerate(hetero[:-1]):
        """ In the first layer idx=0 the problem is initialised by Eq. 48.
        In other layers, idx>0, the problem is seeded by the result
        in layer idx-1 """
        if idx == 0:
            Rud0 = S[0, :, :dim, dim:]
            Tdd0 = S[0, :, dim:, dim:]
            Rdu0 = S[0, :, dim:, :dim]
            Tuu0 = S[0, :, :dim:, :dim]
            if return_inter:
                Rud_ret[0, :, :, :] = np.copy(Rud0)
                Tdd_ret[0, :, :, :] = np.copy(Tdd0)
                Tuu_ret[0, :, :, :] = np.copy(Tuu0)
                Rdu_ret[0, :, :, :] = np.copy(Rdu0)
        else:
            Rud0, Rdu0, Tdd0, Tuu0 = Rud1, Rdu1, Tdd1, Tuu1

        mat = layer[0]  # Gets the material name in the current layer
        thickness = layer[1]  # Gets the current layer thickness

        """ Look up the material properties in the material_data dict,
        and assign the eigenvalues and interface matrix to eigsm
        and imatm respectively
        """
        materialm = material_data[mat]
        eigsm = materialm._eigs
        imatm = materialm._imat

        """ Look up the material properites in the material_data dict for
        layer n = m + 1, and assign the layers interface matrix to imatn
        """
        materialn = material_data[hetero[idx + 1][0]]
        imatn = materialn._imat

        # If doing a local calculation, discard the nonlocal modes
        if loc:
            eigsm = np.concatenate([eigsm[:, 0:2], eigsm[:, 5:7]], axis=1)
            imatm = np.concatenate([imatm[:, :4, :2], imatm[:, :4, 5:7]], axis=2)
            imatn = np.concatenate([imatn[:, :4, :2], imatn[:, :4, 5:7]], axis=2)

        # Calculate the scattering matrices
        Rud1 = Rud(wn, eigsm, Tdd0, Rud0, Rdu0, Tuu0, imatm, imatn, thickness)
        Tdd1 = Tdd(wn, eigsm, Tdd0, Rud0, Rdu0, Tuu0, imatm, imatn, thickness)
        Rdu1 = Rdu(wn, eigsm, Tdd0, Rud0, Rdu0, Tuu0, imatm, imatn, thickness)
        Tuu1 = Tuu(wn, eigsm, Tdd0, Rud0, Rdu0, Tuu0, imatm, imatn, thickness)

        if return_inter:
            Rud_ret[idx + 1, :, :, :] = np.copy(Rud1)
            Tdd_ret[idx + 1, :, :, :] = np.copy(Tdd1)
            Tuu_ret[idx + 1, :, :, :] = np.copy(Tuu1)
            Rdu_ret[idx + 1, :, :, :] = np.copy(Rdu1)

    if return_inter:
        Aout = [Rud_ret, Tdd_ret, Tuu_ret, Rdu_ret]
        return (
            Rdu1[:, 0, 0],
            Rdu1[:, 1, 1],
            Aout,
        )  # , Rud_ret, Rdu_ret, Tuu_ret, Tdd_ret
    else:
        output = Rdu1[:, 0, 0], Rdu1[:, 1, 1]
        return output
