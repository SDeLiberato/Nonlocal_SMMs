__all__ = ['Media', 'properties']

import numpy as np
from scipy.constants import speed_of_light
from itertools import product


class Media:

    def __init__(self, name, props, wn, ang=None, kx=None):
        """ Initiates a material instance with name specified by a string and
        properties by a dict

        Parameters
        ----------

        name : string
            name of the material comprising the layer

        props : dict
            dict containing the material properties

        wn : float or array
            float or array containing the probe frequencies
        """
        self._name = name
        for k, v in props.items():
            setattr(self, '_'+k, v)
        self._wn = wn
        self._eps = self.epsilon_tensor(wn)
        self._mu = self.mu_tensor(wn)
        self._eps_inf = np.diag(
            [self._eps_inf_pe, self._eps_inf_pe, self._eps_inf_pa]
             )
        self._alpha = np.diag(
            [np.sqrt(self._eps_0_pe-self._eps_inf_pe)*self._wto_pe,
             np.sqrt(self._eps_0_pe-self._eps_inf_pe)*self._wto_pe,
             np.sqrt(self._eps_0_pa-self._eps_inf_pa)*self._wto_pa]
            )

        if ang:
            self._ang = ang

            self._eigs, self._vecs = self.layer_dispersion(wn, ang=ang)

            self._fields = self.field_gen(wn, ang=ang)
            self._imat = self.inter_mat(wn, ang=ang)
        elif kx:
            self._kx = kx
            self._eigs, self._vecs = self.layer_dispersion(wn, kx=kx)

            self._fields = self.field_gen(wn, kx=kx)
            self._imat = self.inter_mat(wn, kx=kx)
            pass

    def eps_1_osc(self, wn, orientation='pe'):
        """ Returns the dielectric function evaluated at frequencies wn,
        assuming a 1 oscillator Lorentz model of the dielectric. For materials
        whose phonon frequencies are set to nil returns eps_inf

        Parameters
        ----------

        wn : float or array
            the frequencies to probe

        orientation : string
            either pe or pa, determines whether the function is calculated
            parallel or perpendicular to the crystal axis

        Returns
        -------

        eps : float or array
            dielectric constant evaluated at the input frequencies wn

        """
        if orientation == 'pe':
            wlo, wto, eps_inf, gam = (
                self._wlo_pe, self._wto_pe,
                self._eps_inf_pe, self._gamma
                )
        elif orientation == 'pa':
            wlo, wto, eps_inf, gam = (
                self._wlo_pa, self._wto_pa,
                self._eps_inf_pa, self._gamma
                )

        eps = eps_inf*(wlo**2 - wn*(wn+1j*gam))/(wto**2 - wn*(wn+1j*gam))

        return eps

    def epsilon_tensor(self, wn):
        """ Returns the permittivity tensor evaluated at frequencies wn, currently
        assuming a 1 oscillator Lorentz model of the dielectric and that the
        material is orientated so it's c-axis is parallel to the z-direction.

        Parameters
        ----------

        properties : dict
            contains the material properties to utilise in the calculation

        wn : float or array
            the frequencies to probe

        Returns
        -------

        eps : array
            permittivity tensor evaluated at the input frequencies wn assuming
            the crystal c-axis is parallel to the third dimension

        """

        eps = np.zeros((len(wn), 3, 3), dtype=complex)
        eps[:, 0, 0] = self.eps_1_osc(wn, 'pe')
        eps[:, 1, 1] = self.eps_1_osc(wn, 'pe')
        eps[:, 2, 2] = self.eps_1_osc(wn, 'pa')

        return eps

    def mu_tensor(self, wn):
        """ Returns the permeability tensor evaluated at frequencies wn, currently
        assuming non-magnetic media.

        Parameters
        ----------

        properties : dict
            contains the material properties to utilise in the calculation

        wn : float or array
            the frequencies to probe

        Returns
        -------

        mu : array
            permeability tensor evaluated at the input frequencies wn assuming
            the crystal c-axis is parallel to the third dimension

        """
        mu = np.zeros((3, 3), dtype=complex)
        mu[0, 0] = 1
        mu[1, 1] = 1
        mu[2, 2] = 1

        return mu

    def layer_dispersion(self, wn, ang=None, kx=None):
        """ Calculates the dispersion of the photonic and longitudinal and transverse
        phonon eigenmodes of an anisotropic medium, describable by a diagonal
        permeability and permitivitty

        Parameters
        ----------

        kx : float or 1d array
            contains a range of in-plane wavevectors to probe over

        wn : float or 1d array
            contains a range of frequencies to probe over

        material : string
            material to analyse, corresponds to an entry in ...

        ang : float
            optional, corresponds to the incident angle. useage overrides entry
            of kx

        Returns
        -------

        eigs : array
            Values representing calculated out-of-plane wavevectors of the
            eigenmodes of the layer.

        vecs : array
            Values representing the calculated field components corresponding to
            the eigenmodes of the layer. These are in order Ex, Ey, Xx, Xy, Xz


        References
        ----------

        The formation of the matrix from which the dispersions are calculated is
        detailed at " "

        To-do
        _____

        Currently returning TE and TM polarizations with degenerate frequencies
        but mixed field polarizations. This isn't great, can we sort in some way?
        Maybe look at how Nikolai sifts for polarization?

        """

        # Checks for non-array inputs and converts to numpy arrays
        if kx and type(kx) != np.ndarray:
            kx = np.array([kx])
        if type(wn) != np.ndarray:
            wn = np.array(wn)

        # Checks for an input angle, if present calculates zeta using this else
        # uses the input wavevector array
        if ang:
            kx = np.zeros(1)  # Creates a dummy kx to ease code maintanence below
            arrays = [(wn), (kx)]
            arr = np.array(list(product(*arrays)))
            zeta = np.zeros_like(wn)
            zeta[:] = np.sin(ang)
        else:
            arrays = [(wn), (kx)]
            arr = np.array(list(product(*arrays)))
            zeta = arr[:, 1]/arr[:, 0]

        if self._beta_l == 0:
            # Creates an array of len(zeta) 5 x 5 identity matrices
            It = np.identity(2)
            It = np.swapaxes(np.broadcast_to(It[..., None], (2, 2, len(zeta))), 0, 2)
            # Creates an array of len(zeta) 5 x 5 null matrices

            tAit = np.zeros_like(It, dtype=complex)
            tAit[:, 0, 0] = (zeta**2/self._eps_inf[2, 2]-self._mu[1, 1])
            tAit[:, 1, 1] = -self._mu[0, 0]

            tCt = np.zeros_like(It, dtype=complex)
            tCt[:, 0, 0] = self._eps_inf[0, 0]
            tCt[:, 1, 1] = self._eps_inf[1, 1] - zeta**2/self._mu[2, 2]

            fm = np.matmul(tAit, tCt)

            sq_eigs, vecs = np.linalg.eig(-fm)
            eigs = np.sqrt(sq_eigs)

            eigs = eigs.reshape((len(kx), len(wn), 2), order='F')
            vecs = vecs.reshape((len(kx), len(wn), 2, 2), order='F')

            eigs_ph, vecs_ph = self.pol_comp(eigs[:, :, :], vecs[:, :, :, :])

            ts0, ts1, ts2 = eigs_ph.shape
            zm = np.zeros((ts0, ts1, ts2+1))

            sorted_eigs = np.concatenate((eigs_ph, zm), axis=2)
            sorted_eigs_r = np.concatenate((-eigs_ph, zm), axis=2)

            ts0, ts1, ts2, ts3 = vecs_ph.shape
            zm1 = np.zeros((ts0, ts1, ts2+6, ts3))
            zm2 = np.zeros((ts0, ts1, ts2+8, ts3+1))

            sorted_vecs = np.concatenate(
                (np.concatenate((vecs_ph, zm1), axis=2), zm2),
                axis=3
            )
            sorted_vecs_r = np.concatenate(
                (np.concatenate((vecs_ph, zm1), axis=2), zm2),
                axis=3
            )

        else:

            # Creates an array of len(zeta) 5 x 5 identity matrices
            It = np.identity(5)
            It = np.swapaxes(np.broadcast_to(It[..., None], (5, 5, len(zeta))), 0, 2)
            # Creates an array of len(zeta) 5 x 5 null matrices
            Zt = np.zeros_like(It)

            tAt = np.zeros_like(It, dtype=complex)
            tBt = np.zeros_like(It, dtype=complex)

            tAt[:, 0, 0] = self._eps_inf[0, 0]*(-zeta**2/self._eps_inf[2, 2]+self._mu[1, 1])
            tAt[:, 0, 2] = self._alpha[0, 0]*(-zeta**2/self._eps_inf[2, 2]+self._mu[1, 1])

            tAt[:, 1, 1] = self._mu[0, 0]*(-zeta**2/self._mu[2, 2]+self._eps_inf[1, 1])
            tAt[:, 1, 3] = self._alpha[1, 1]*self._mu[0, 0]

            tAt[:, 2, 0] = -2*self._alpha[0, 0]/arr[:, 0]**2/self._beta_c**2
            tAt[:, 2, 2] = 2*(self._wto_pe**2
                              - arr[:, 0]*(arr[:, 0]+1j*self._gamma)
                              - self._beta_l**2*zeta**2*arr[:, 0]**2
                              )/arr[:, 0]**2/self._beta_c**2
            tAt[:, 3, 1] = -2*self._alpha[1, 1]/arr[:, 0]**2/self._beta_c**2
            tAt[:, 3, 3] = 2*(self._wto_pe**2
                              - arr[:, 0]*(arr[:, 0]+1j*self._gamma)
                              - 0.5*self._beta_c**2*zeta**2*arr[:, 0]**2
                              )/arr[:, 0]**2/self._beta_c**2

            tAt[:, 4, 4] = ( (self._wto_pa**2
                               - arr[:, 0]*(arr[:, 0]+1j*self._gamma)
                               - 0.5*self._beta_c**2*zeta**2*arr[:, 0]**2
                               )/arr[:, 0]**2/self._beta_l**2
                            - (self._alpha[2, 2]**2*self._mu[1, 1]
                                /arr[:, 0]**2/self._beta_l**2
                                /(zeta**2 - self._eps_inf[2, 2]*self._mu[1, 1])
                                )
                            )

            tBt[:, 0, 4] = -zeta*self._alpha[2, 2] / self._eps_inf[2, 2]

            tBt[:, 2, 4] = -2*zeta*(
                                    self._beta_c**2/2
                                    + self._beta_l**2
                                    - 2*self._beta_t**2
                                    ) / self._beta_c**2

            tBt[:, 4, 0] = - self._alpha[2, 2]*zeta/(self._beta_l**2*arr[:, 0]**2
                                                      *(zeta**2-self._eps_inf[2, 2]*self._mu[1, 1])
                                                      )

            tBt[:, 4, 2] = -1*zeta*(
                                    self._beta_c**2/2
                                    + self._beta_l**2
                                    - 2*self._beta_t**2
                                    ) / self._beta_l**2

            M3 = -np.concatenate((
                np.hstack((Zt, It)),
                np.hstack((tAt, tBt))
                ), axis=2)

            eigs, vecs = np.linalg.eig(M3)

            # Reshape into a useable array
            eigs = eigs.reshape((len(kx), len(wn), 10), order='F')
            vecs = vecs.reshape((len(kx), len(wn), 10, 10), order='F')

            # Initial sort for photon modes
            a = -1
            order = eigs.argsort(axis=a)
            m, n, k = eigs.shape
            idx = np.ogrid[:m, :n, :k]
            idx[-1] = order
            eigs_r = eigs[tuple(idx)]

            m, n, k, o = vecs.shape
            idx = np.ogrid[:m, :n, :k, :o]
            order = np.repeat(order[:, :, np.newaxis, :], 10, axis=2)
            idx[-1] = order
            vecs_r = vecs[tuple(idx)]

            eigs_photon = np.concatenate(
                (eigs_r[:, :, 3:5], eigs_r[:, :, 5:7]), axis=2
                )
            vecs_photon = np.concatenate(
                (vecs_r[:, :, :, 3:5], vecs_r[:, :, :, 5:7]), axis=3
                )

            # Initial sort for photon modes
            a = -1
            order = np.lexsort(
                (eigs_photon.imag, -np.sign(eigs_photon.imag)),
                axis=a
                )
            m, n, k = eigs_photon.shape
            idx = np.ogrid[:m, :n, :k]
            idx[-1] = order
            eigs_photon_s = eigs_photon[tuple(idx)]

            m, n, k, o = vecs_photon.shape
            idx = np.ogrid[:m, :n, :k, :o]
            order = np.repeat(order[:, :, np.newaxis, :], 10, axis=2)
            idx[-1] = order
            vecs_photon_s = vecs_photon[tuple(idx)]

            eigs_ph, vecs_ph = self.pol_comp(
                eigs_photon_s[:, :, 0:2], vecs_photon_s[:, :, :, 0:2]
                )
            eigs_ph_r, vecs_ph_r = self.pol_comp(
                eigs_photon_s[:, :, 2:], vecs_photon_s[:, :, :, 2:]
                )

            # New Sort, more robust for phonon modes
            eigs_phonon = np.concatenate(
                (eigs_r[:, :, :3], eigs_r[:, :, 7:]), axis=2
                )
            vecs_phonon = np.concatenate(
                (vecs_r[:, :, :, :3], vecs_r[:, :, :, 7:]), axis=3
                )

            a = -1
            order = np.lexsort(
                (eigs_phonon.real, eigs_phonon.imag, -np.sign(eigs_phonon.real)),
                axis=a
                )
            m, n, k = eigs_phonon.shape
            idx = np.ogrid[:m, :n, :k]
            idx[-1] = order
            eigs_r_pho = eigs_phonon[tuple(idx)]

            m, n, k, o = vecs_phonon.shape
            idx = np.ogrid[:m, :n, :k, :o]
            order = np.repeat(order[:, :, np.newaxis, :], 10, axis=2)
            idx[-1] = order
            vecs_r_pho = vecs_phonon[tuple(idx)]

            eigs_to, vecs_to = self.pol_comp(
                eigs_r_pho[:, :, 4:], vecs_r_pho[:, :, :, 4:]
                )

            sorted_eigs = np.concatenate((np.concatenate(
                (eigs_ph, eigs_to), axis=2
                ), eigs_r_pho[:, :, 3:4]), axis=2
            )

            sorted_vecs = np.concatenate((np.concatenate(
                (vecs_ph, vecs_to), axis=3
                ), vecs_r_pho[:, :, :, 3:4]), axis=3
            )

            # Reverse propagation
            eigs_to_r, vecs_to_r = self.pol_comp(
                eigs_r_pho[:, :, 0:2], vecs_r_pho[:, :, :, 0:2]
                )

            sorted_eigs_r = np.concatenate((np.concatenate(
                (eigs_ph_r, eigs_to_r), axis=2
                ), eigs_r_pho[:, :, 2:3]), axis=2
            )

            sorted_vecs_r = np.concatenate((np.concatenate(
                (vecs_ph_r, vecs_to_r), axis=3
                ), vecs_r_pho[:, :, :, 2:3]), axis=3
            )

        if ang:
            eigs_out = np.concatenate([sorted_eigs[0], sorted_eigs_r[0]], axis=1)
            vecs_out = np.concatenate([sorted_vecs[0], sorted_vecs_r[0]], axis=2)
            return eigs_out, vecs_out
        else:
            eigs_out = np.concatenate([sorted_eigs[0], sorted_eigs_r[0]], axis=1)
            vecs_out = np.concatenate([sorted_vecs[0], sorted_vecs_r[0]], axis=2)
            return eigs_out, vecs_out

    def pol_comp(self, eigs, vecs):
        """ Function to compare the eigenmode polarisation state
        dot dot dot
        """

        ldx = vecs.shape[-2]

        fom = (
            np.abs(vecs[:, :, 0])**2
            / (np.abs(vecs[:, :, 0])**2+np.abs(vecs[:, :, 1])**2)
            )

        a = -1
        order = fom.argsort(axis=a)
        m, n, k = eigs.shape
        idx = np.ogrid[:m, :n, :k]
        idx[-1] = order
        eigs = eigs[tuple(idx)]

        m, n, k, o = vecs.shape
        idx = np.ogrid[:m, :n, :k, :o]
        order = np.repeat(order[:, :, np.newaxis, :], ldx, axis=2)
        idx[-1] = order
        vecs = vecs[tuple(idx)]

        return eigs, vecs

    def field_gen(self, wn, ang=None, kx=None):
        """ Takes the eigenvalues and fields calculated for a given layer and finds
        the full vector of fields required to apply the boundary conditions. The
        fields are normalised to the electric field parallel to the interface, for
        TM polarised modes this means they are normalised with respect to the
        x-component while TE polarised modes they are normalised with respect
        to the y-comp


        Parameters
        ----------

        kx : float or 1d array
            contains a range of in-plane wavevectors to probe over

        wn : float or 1d array
            contains a range of frequencies to probe over

        eigs : 3d array
            contains eigenvalues outputted from layer_dispersion, corresponding to
            the frequencies and wavevectors provided in kx, wn

        vecs: 3d array
            contains the eigenvectors outputted from layer_dispersion corresponding
            to the frequencies and wavevectors provided in kx, wn

        material : string
            material to analyze, corresponds to an entry in ...

        Outputs
        -------

        fields : 3d array
            the full array of fields necessary to solve the boundary matching
            problem for all kx, wn. Fields are outputted in order H, E, P, X with
            Cartesian components for each ordered x, y, z and are normalised as
            described above
        """

        # Checks for non-array inputs and converts to numpy arrays
        if kx and type(kx) != np.ndarray:
            kx = np.array([kx])
        if type(wn) != np.ndarray:
            wn = np.array(wn)

        if self._name == 'vac':
            vecs = self._vecs[:, :5, :]
        else:
            vecs = self._vecs[:, 5:, :]

        eigs = self._eigs
        # Checks for equal length inputs, assumed to correspond to a cut through
        # (kx, wn) space
        if ang:
            kx = np.zeros(1)  # Creates a dummy kx to ease code maintanence below
            arrays = [(wn), (kx)]
            arr = np.array(list(product(*arrays)))
            zeta = np.zeros_like(wn)
            zeta[:] = np.sin(ang)
            vecs = np.expand_dims(vecs, axis=0)
            eigs = np.expand_dims(eigs, axis=0)
            zeta = np.expand_dims(zeta, axis=0)
        else:
            arrays = [(wn), (kx)]
            arr = np.array(list(product(*arrays)))
            zeta = arr[:, 1]/arr[:, 0]
            # zeta = np.transpose(zeta.reshape((len(arrays[0]), len(arrays[1]))))
            # Reshape input arrays for calculation
            # kx = np.transpose(arr[:, 1].reshape((len(arrays[0]), len(arrays[1]))))
            # wn = np.transpose(arr[:, 0].reshape((len(arrays[0]), len(arrays[1]))))
            vecs = np.expand_dims(vecs, axis=0)
            eigs = np.expand_dims(eigs, axis=0)
            zeta = np.expand_dims(zeta, axis=0)
        # Initialize empty output vector
        field_vec = np.zeros(
            (len(arrays[1]), len(arrays[0]), 10, 12), dtype=complex
        )

        # Fill components which pre-exist in the input
        field_vec[:, :, :, 3] = vecs[:, :, 0, :]
        field_vec[:, :, :, 4] = vecs[:, :, 1, :]

        Ex0 = vecs[:, :, 0, :]
        Ey0 = vecs[:, :, 1, :]

        if self._beta_l == 1.55:

            zetaf = np.repeat(zeta[:, :, np.newaxis], 2, axis=2)

            field_vec[:, :, :, 0] = -eigs*Ey0/self._mu[0, 0]
            field_vec[:, :, :, 1] = Ex0*eigs*self._eps_inf[2, 2]/(zetaf**2-self._eps_inf[2, 2]*self._mu[1, 1])
            field_vec[:, :, :, 2] = zetaf*Ey0/self._mu[2, 2]
            # Calculates the z-component of the electric field from Eq. in the tex file
            field_vec[:, :, :, 5] = Ex0*eigs*zetaf/(zetaf**2-self._eps_inf[2, 2]*self._mu[1, 1])

        else:
            field_vec[:, :, :, 9] = vecs[:, :, 2, :]
            field_vec[:, :, :, 10] = vecs[:, :, 3, :]
            field_vec[:, :, :, 11] = -vecs[:, :, 4, :]

            # Broadcast zeta to the leading dimensions of the other arrays
            zetaf = np.repeat(zeta[:, :, np.newaxis], 10, axis=2)

            Xx0 = vecs[:, :, 2, :]
            Xy0 = vecs[:, :, 3, :]
            Xz0 = -vecs[:, :, 4, :]

            # Calculates the z-component of the electric field from Eq. in the tex file
            field_vec[:, :, :, 5] = (
                (
                    Ex0*eigs*zetaf
                    + self._alpha[2, 2]*self._mu[1, 1]
                    * Xz0
                    )
                / (zetaf**2-self._eps_inf[2, 2]*self._mu[1, 1])
                )

            # Calculates the x-component of the magnetic field from Eq. in the tex file
            field_vec[:, :, :, 0] = -eigs*Ey0/self._mu[0, 0]

            # Calculates the y-component of the magnetic field from Eq. in the tex file
            field_vec[:, :, :, 1] = (
                (
                    Ex0*eigs*self._eps_inf[2, 2]
                    + self._alpha[2, 2]*self._mu[1, 1]
                    * Xz0*zetaf
                    )
                / (zetaf**2-self._eps_inf[2, 2]*self._mu[1, 1])
                )

            # Calculates the z-component of the magnetic field from Eq. in the tex file
            field_vec[:, :, :, 2] = zetaf*Ey0/self._mu[2, 2]

            # Calculates the x-component of the polarization field from Eq. in the tex
            field_vec[:, :, :, 6] = - (
                (
                    Ex0*(
                        zetaf**2 + self._eps_inf[2, 2]*eigs**2
                        - self._eps_inf[2, 2]*self._mu[1, 1]
                        )
                    + self._alpha[2, 2]*self._mu[1, 1]
                    * Xz0*zetaf*eigs
                    )
                / (zetaf**2-self._eps_inf[2, 2]*self._mu[1, 1])
                )

            # Calculates the y-component of the polarization field from Eq. in the tex
            field_vec[:, :, :, 7] = - (
                1 - eigs**2/self._mu[0, 0] - zetaf**2/self._mu[2, 2]
                )*Ey0

            # Calculates the z-component of the polarization field from Eq. in the tex
            field_vec[:, :, :, 8] = (
                (
                    Ex0*zetaf*eigs*(self._eps_inf[2, 2] - 1)
                    + self._alpha[2, 2]*(zetaf**2 - self._mu[1, 1])
                    * Xz0
                    )
                / (zetaf**2-self._eps_inf[2, 2]*self._mu[1, 1])
                )

        # Compares the value of the x and y electric Fields
        ivec = (
            np.sum(np.sum(np.abs(field_vec[:, :, :, 3]), axis=1), axis=0)
            > np.sum(np.sum(np.abs(field_vec[:, :, :, 4]), axis=1), axis=0)
            )

        field_vec_o = np.zeros_like(field_vec)

        for idx, iv in enumerate(ivec):
            if iv:
                if np.any(field_vec[:, :, idx, 3, None] == 0):
                    field_vec_o[:, :, idx, :] = field_vec[:, :, idx, :]
                else:
                    field_vec_o[:, :, idx, :] = (
                      field_vec[:, :, idx, :]/field_vec[:, :, idx, 3, None]
                    )
            else:
                if np.any(field_vec[:, :, idx, 4, None] == 0):
                    field_vec_o[:, :, idx, :] = field_vec[:, :, idx, :]
                else:
                    field_vec_o[:, :, idx, :] = (
                        field_vec[:, :, idx, :]/field_vec[:, :, idx, 4, None]
                    )

        if ang:
            return field_vec_o[0]
        else:
            return field_vec_o[0]

    def inter_mat(self, wn, ang=None, kx=None):
        """ Calculates the interface matrix given that layer's eigenvalues
        corresponding to the out-of-plane wavevector and eigenvectors

        Parameters
        ----------

        wn : 1d array
            a 1d array, containing the probing wavenumbers

        eigs : array
            a 2d array of dimensions (len(wn) x 10) where 10 is the number of
            modes in the problem. This corresponds to the out of plane wavevector
            divided by the wavenumber

        fields : array
            the array of sorted fields


        Returns
        -------

        inter_mat : array
            a matrix of size (len(wn)x10x10) containing the propagation factors for
            each eigenmode
        """
        # Checks for non-array inputs and converts to numpy arrays
        if kx and type(kx) != np.ndarray:
            kx = np.array([kx])
        if type(wn) != np.ndarray:
            wn = np.array(wn)

        # Checks for an input angle, if present calculates zeta using this else
        # uses the input wavevector array
        if ang:
            zeta = np.zeros_like(wn)
            zeta[:] = np.sin(ang)
        else:
            arrays = [(wn), (kx)]
            arr = np.array(list(product(*arrays)))
            zeta = arr[:, 1]/arr[:, 0]

        inter_mat = np.zeros((len(wn), 10, 10), dtype=complex)

        # In-plane electric field (x)
        inter_mat[:, 0, :] = self._fields[:, :, 3]
        # In-plane electric field (y)
        inter_mat[:, 1, :] = self._fields[:, :, 4]

        # In-plane magnetic field (x)
        inter_mat[:, 2, :] = self._fields[:, :, 0]
        # In-plane magnetic field (y)
        inter_mat[:, 3, :] = self._fields[:, :, 1]

        # In-plane ionic displacement field (x)
        inter_mat[:, 4, :] = self._fields[:, :, 9]
        # In-plane ionic displacement ield (y)
        inter_mat[:, 5, :] = self._fields[:, :, 10]
        #  In-plane ionic displacement field (y)
        inter_mat[:, 6, :] = self._fields[:, :, 11]

        # In-plane stress tensor (x)
        inter_mat[:, 7, :] = 0.5*self._beta_c**2*(
            self._eigs*self._fields[:, :, 9]+zeta[:, None]*self._fields[:, :, 11]
            )

        # In-plane stress tensor (y)
        inter_mat[:, 8, :] = 0.5*self._beta_c**2*(self._eigs*self._fields[:, :, 10])

        # In-plane stress tensor (z)
        inter_mat[:, 9, :] = (
            +0.5*self._beta_l**2*self._eigs*self._fields[:, :, 11]
            + (self._beta_l**2 - 2*self._beta_t**2)*(zeta[:, None]*self._fields[:, :, 9])
        )

        return inter_mat


def properties(mat):
    """ Returns a dict containing the relevant properties of the material specified
    by the input string

    Parameters
    ----------

    mat : string
        material to analyse

    Returns
    -------

    props : dict
        Contains the relevant parameters corresponding to the string mat


    References
    ----------

    """

    mats = [
    'vac', 'SiC3C', 'SiC4H', 'GaN', 'AlN', 'hBN', 'SiO2', 'SiC4Hsub','AlNiso', 'GaNiso','AlNisob', 'GaNisob','AlNisoa', 'GaNisoa',
    'AlNanisoa', 'GaNanisoa']
    if mat not in mats:
        raise TypeError(
            "There is no data corresponding to." + mat)
    else:
        props = dict()
        if mat == 'vac':
            props['beta_c'] = 0
            props['beta_l'] = 0
            props['beta_t'] = 0
            props['rho'] = 1
            props['eps_inf_pe'] = 1
            props['eps_inf_pa'] = 1
            props['eps_0_pe'] = 1
            props['eps_0_pa'] = 1
            props['wto_pe'] = 0
            props['wto_pa'] = 0
            props['gamma'] = 0
        if mat == 'SiC3C':
            props['beta_l'] = 1e4/speed_of_light
            props['beta_t'] = 1e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pa'] = 6.52
            props['eps_inf_pe'] = 6.52
            props['eps_0_pa'] = 9.7
            props['eps_0_pe'] = 9.7
            props['wto_pa'] = 797.4
            props['wto_pe'] = 797.5
            props['gamma'] = 4
        if mat == 'SiC4H':
            props['beta_l'] = 2e4/speed_of_light
            props['beta_t'] = 1e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pa'] = 6.56
            props['eps_inf_pe'] = 6.78
            props['eps_0_pa'] = 9.780
            props['eps_0_pe'] = 10.34
            props['wto_pa'] = 796.6
            props['wto_pe'] = 783.6
            props['gamma'] = 4
        if mat == 'SiO2':
            props['beta_l'] = 0.0002e4/speed_of_light
            props['beta_t'] = 0.0001e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pa'] = 6.56
            props['eps_inf_pe'] = 6.56
            props['eps_0_pa'] = 6.561
            props['eps_0_pe'] = 6.561
            props['wto_pa'] = 10
            props['wto_pe'] = 10
            props['gamma'] = 0.1
        if mat == 'SiC4Hsub':
            props['beta_l'] = 2e4/speed_of_light
            props['beta_t'] = 1e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pe'] = 6.56
            props['eps_inf_pa'] = 6.78
            props['eps_0_pe'] = 9.780
            props['eps_0_pa'] = 10.34
            props['wto_pe'] = 796.6#-1.5
            props['wto_pa'] = 783.6#-1.5
            props['gamma'] = 4
        if mat == 'GaN':
            props['beta_l'] = 1e4/speed_of_light
            props['beta_t'] = 1e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pa'] = 5.469
            props['eps_inf_pe'] = 5.416
            props['eps_0_pa'] = 10.4
            props['eps_0_pe'] = 9.5
            props['wto_pa'] = 533
            props['wto_pe'] = 561
            props['gamma'] = 4
        if mat == 'AlN':
            props['beta_l'] = 1.75e4/speed_of_light
            props['beta_t'] = 2e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pe'] = 4.86  # non set
            props['eps_inf_pa'] = 4.84  # non set
            props['eps_0_pa'] = 10.238
            props['eps_0_pe'] = 8.97
            props['wto_pa'] = 614
            props['wto_pe'] = 673
            props['gamma'] = 4
        if mat == 'hBN':
            props['beta_l'] = 1.75e4/speed_of_light
            props['beta_t'] = 2e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pe'] = 4.86
            props['eps_inf_pa'] = 2.85
            props['eps_0_pa'] = 3.34
            props['eps_0_pe'] = 6.72
            props['wto_pa'] = 780
            props['wto_pe'] = 1370
            props['gamma'] = 4
        if mat == 'AlNiso':
            props['beta_l'] = 0.9e4/speed_of_light
            props['beta_t'] = 1.0e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pe'] = 4.86  # non set
            props['eps_inf_pa'] = 4.86  # non set
            props['eps_0_pa'] = 10.238
            props['eps_0_pe'] = 10.238
            props['wto_pa'] = 610.5
            props['wto_pe'] = 611
            props['gamma'] = 6.4
        if mat == 'GaNiso':
            props['beta_l'] = 1e4/speed_of_light
            props['beta_t'] = 1.2e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pa'] = 5.416
            props['eps_inf_pe'] = 5.416
            props['eps_0_pa'] = 9.5
            props['eps_0_pe'] = 9.5
            props['wto_pa'] = 560.5
            props['wto_pe'] = 561
            props['gamma'] = 4
            props['gammaph'] = 2
        if mat == 'AlNisob':
            sc = 2.525
            props['beta_l'] = sc*0.9e4/speed_of_light
            props['beta_t'] = sc*1.0e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pe'] = 4.86  # non set
            props['eps_inf_pa'] = 4.86  # non set
            props['eps_0_pa'] = 10.238
            props['eps_0_pe'] = 10.238
            props['wto_pa'] = 610.5
            props['wto_pe'] = 611
            props['gamma'] = 2
            props['gammaph'] = 2
        if mat == 'GaNisob':
            sc = 2.525
            props['beta_l'] = sc*1e4/speed_of_light
            props['beta_t'] = sc*1.2e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)

            props['rho'] = 3.21
            props['eps_inf_pa'] = 5.416
            props['eps_inf_pe'] = 5.416
            props['eps_0_pa'] = 9.5
            props['eps_0_pe'] = 9.5
            props['wto_pa'] = 560.5
            props['wto_pe'] = 561
            props['gamma'] = 11*4
            # props['gamma'] = 1*4
        if mat == 'AlNisoa':
            # sc = 1
            sc = 0.585

            props['beta_l'] = sc*0.9e4/speed_of_light
            props['beta_t'] = sc*1.0e4/speed_of_light

            props['beta_l'] = 0.65e4/speed_of_light
            props['beta_t'] = 1.0e1/speed_of_light

            # props['beta_t'] = 1.0e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pe'] = 4.86  # non set
            props['eps_inf_pa'] = 4.86  # non set
            props['eps_0_pa'] = 10.238
            props['eps_0_pe'] = 10.238
            props['wto_pa'] = 611.48
            props['wto_pe'] = 611.52
            props['gamma'] = 10/4
        if mat == 'AlNanisoa':
            # sc = 1
            sc = 0.585

            props['beta_l'] = sc*0.9e4/speed_of_light
            props['beta_t'] = sc*1.0e4/speed_of_light

            props['beta_l'] = 0.9e4/speed_of_light
            props['beta_t'] = 1.0e1/speed_of_light

            props['beta_c'] = np.sqrt(2*props['beta_t']**2)
            props['rho'] = 3.21
            props['eps_inf_pe'] = 4.86  # non set
            props['eps_inf_pa'] = 4.84  # non set
            props['eps_0_pa'] = 10.238
            props['eps_0_pe'] = 8.97
            props['wto_pa'] = 611
            props['wto_pe'] = 673
            props['gamma'] = 6
        if mat == 'GaNisoa':
            sc = 1


            props['beta_l'] = sc*0.9e4/speed_of_light
            props['beta_t'] = sc*1.0e4/speed_of_light

            props['beta_l'] = 0.65e4/speed_of_light
            props['beta_t'] = 1.0e1/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)

            props['rho'] = 3.21
            props['eps_inf_pa'] = 5.416
            props['eps_inf_pe'] = 5.416
            props['eps_0_pa'] = 9.5
            props['eps_0_pe'] = 9.5
            props['wto_pa'] = 560.5
            props['wto_pe'] = 561
            props['gamma'] = 6
        if mat == 'GaNanisoa':
            sc = 1


            props['beta_l'] = sc*0.9e4/speed_of_light
            props['beta_t'] = sc*1.0e4/speed_of_light
            props['beta_c'] = np.sqrt(2*props['beta_t']**2)

            props['rho'] = 3.21
            props['eps_inf_pa'] = 5.469
            props['eps_inf_pe'] = 5.416
            props['eps_0_pa'] = 10.4
            props['eps_0_pe'] = 9.5
            props['wto_pa'] = 533
            props['wto_pe'] = 561
            props['gamma'] = 4

    props['wlo_pa'] = (
        props['wto_pa']*np.sqrt(props['eps_0_pa']/props['eps_inf_pa'])
        )
    props['wlo_pe'] = (
        props['wto_pe']*np.sqrt(props['eps_0_pe']/props['eps_inf_pe'])
        )
    return props