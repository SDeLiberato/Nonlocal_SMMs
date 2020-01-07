__all__ = ['scattering_matrix']

import numpy as np
from source.materials import Media, properties
from source.helper import *

def scattering_matrix(
            wn, hetero, ang=None, kx=None, loc=None,
            params=None, return_inter=False
        ):
    """ Calculates the scattering matrix as presented in the paper doi ...

    :param wn: Input wavenumbers in inverse centimetres
    :type wn: 1D numpy array
    :param hetero: Ordered list of material names and thicknesses comprising
        the heterostructure
    :type hetero: List
    :param ang: Incident angle in degrees, defaults to None
    :type ang: Float, optional
    :param kx: Incident wavevectors in inverse centimetre, defaults to None
    :type kx: 1D numpy array
    :param params: Parameters to overwrite the default values in the material
        library, defaults to none
    :type params: Dict, optional
    :param return_inter: ??
    :type return_inter: ??

    :return: Returns a tuple, whose first (second) elements correspond to TE
        (TM) polarised reflectance
    :rtype: Tuple
    """

    # Create a list of the unique materials in hetero
    mats = list(set([row[0] for row in hetero]))
    material_data = dict()  # Initialises a blank dict for the material data
    if ang:
        angr = np.deg2rad(ang)  # Convert the incident angle to egrees

    """ Iterates through the heterostructure materials, adding the corresponding
    properties to material_data
    """
    for mat in mats:
        props = properties(mat)
        """ Checks for custom parameters, passed in the params dict, and if present
        updates the material properties accordingly
        """
        if params:
            if mat in params:
                for key, val in params[mat].items():
                    props[key] = val
        if ang:
            material_data[mat] = Media(mat, props, wn, ang=angr)
        elif kx:
            material_data[mat] = Media(mat, props, wn, kx=kx)

    # Initialises the problem by Eq. 48 in the local and nonlocal cases
    if loc:
        S = np.zeros((3, len(wn), 4, 4))
        S[0, :] = np.diag(np.ones(4))
    else:
        S = np.zeros((3, len(wn), 10, 10))
        S[0, :] = np.diag(np.ones(10))

    dim = S.shape[2]//2  #Calculates the dimension of the 4 block matrices

    if return_inter:
        tlen = S.shape[1]
        nlay = len(hetero)
        Rud_ret = np.zeros((nlay, tlen, dim, dim), dtype=complex)
        Tdd_ret = np.zeros((nlay, tlen, dim, dim), dtype=complex)
        Tuu_ret = np.zeros((nlay, tlen, dim, dim), dtype=complex)
        Rdu_ret = np.zeros((nlay, tlen, dim, dim), dtype=complex)

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
        materialn = material_data[hetero[idx+1][0]]
        imatn = materialn._imat

        if loc:
            eigsm = np.concatenate([eigsm[:, 0:2], eigsm[:, 5:7]], axis=1)
            imatm = np.concatenate([imatm[:, :4, :2],
                                    imatm[:, :4, 5:7]], axis=2)
            imatn = np.concatenate([imatn[:, :4, :2],
                                    imatn[:, :4, 5:7]], axis=2)

        Rud1 = Rud(wn, eigsm, Tdd0, Rud0, Rdu0, Tuu0, imatm, imatn, thickness)
        Tdd1 = Tdd(wn, eigsm, Tdd0, Rud0, Rdu0, Tuu0, imatm, imatn, thickness)
        Rdu1 = Rdu(wn, eigsm, Tdd0, Rud0, Rdu0, Tuu0, imatm, imatn, thickness)
        Tuu1 = Tuu(wn, eigsm, Tdd0, Rud0, Rdu0, Tuu0, imatm, imatn, thickness)

        if return_inter:
            Rud_ret[idx+1, :, :, :] = np.copy(Rud1)
            Tdd_ret[idx+1, :, :, :] = np.copy(Tdd1)
            Tuu_ret[idx+1, :, :, :] = np.copy(Tuu1)
            Rdu_ret[idx+1, :, :, :] = np.copy(Rdu1)

    if return_inter:
        Aout = [Rud_ret, Tdd_ret, Tuu_ret, Rdu_ret]
        return Rdu1[:, 0, 0], Rdu1[:, 1, 1], Aout#, Rud_ret, Rdu_ret, Tuu_ret, Tdd_ret
    else:
        return Rdu1[:, 0, 0], Rdu1[:, 1, 1]
