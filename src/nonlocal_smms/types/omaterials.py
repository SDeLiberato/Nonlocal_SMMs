"""Contains the media class, which holds the properties of a given material in bulk"""


__all__ = ["Media"]

from typing import Any, Tuple

from itertools import product
import numpy as np

speed_of_light = 299792458


class Media:
    """A class to describe the optical properties of a bulk medium."""

    material: str
    wavenumber: np.ndarray
    angle: float
    wavevector: np.ndarray
    orientation: np.ndarray
    eps: np.ndarray
    mu: np.ndarray
    alpha: np.ndarray
    eigs: np.ndarray
    vecs: np.ndarray
    fields: np.ndarray
    imat: np.ndarray
    beta_c: float
    beta_t: float
    beta_l: float
    rho: float
    eps_inf_pe: float
    eps_inf_pa: float
    eps_0_pe: float
    eps_0_pa: float
    gamma: float
    wto_pe: float
    wto_pa: float
    wlo_pe: float
    wlo_pa: float
    wlo_ax_pe: float
    wto_ax_pe: float
    gamma_ax_pe: float

    def __init__(
        self,
        material: str,
        properties: dict,
        wavenumber: np.ndarray,
        angle: float = None,
        wavevector: np.ndarray = None,
        orientation: str = "c-cut",
    ):
        """Initiates a material instance.

        Args:
            material (str): The name of the material comprising the layer
            properties (dict): Containing the material properties
            wavenumber (np.ndarray): The probe frequencies in inverse centimetres
            angle (float, optional): The incident angle in degrees
            wavevector (np.ndarray, optional): The incident in-plane wavevector in
                inverse centimetres. Either an angle or an incident wavevector must
                be passed.
            orientation (str): The crystal orientation, accepts either 'a-cut' or 'c-cut'

        """
        self.orientation = orientation
        for k, v in properties.items():
            setattr(self, k, v)
        self.wavenumber = wavenumber
        self.eps = self.epsilon_tensor(wavenumber)
        self.mu = self.mu_tensor(wavenumber)

        if self.orientation == "c-cut":
            self.eps_inf = np.diag([self.eps_inf_pe, self.eps_inf_pe, self.eps_inf_pa])
            self.alpha = np.diag(
                [
                    np.sqrt(self.eps_0_pe - self.eps_inf_pe) * self.wto_pe,
                    np.sqrt(self.eps_0_pe - self.eps_inf_pe) * self.wto_pe,
                    np.sqrt(self.eps_0_pa - self.eps_inf_pa) * self.wto_pa,
                ]
            )
        elif self.orientation == "a-cut":
            self.eps_inf = np.diag([self.eps_inf_pa, self.eps_inf_pe, self.eps_inf_pe])
            self.alpha = np.diag(
                [
                    np.sqrt(self.eps_0_pa - self.eps_inf_pa) * self.wto_pa,
                    np.sqrt(self.eps_0_pe - self.eps_inf_pe) * self.wto_pe,
                    np.sqrt(self.eps_0_pe - self.eps_inf_pe) * self.wto_pe,
                ]
            )

        if angle:
            self.angle = angle

            self.eigs, self.vecs = self.layer_dispersion(wavenumber, angle=angle)

            self.fields = self.field_generator(wavenumber, angle=angle)
            self.imat = self.interface_matrix(wavenumber, angle=angle)
        elif wavevector is not None:
            self.wavevector = wavevector
            self.eigs, self.vecs = self.layer_dispersion(
                wavenumber, wavevector=wavevector
            )

            self.fields = self.field_generator(wavenumber, wavevector=wavevector)
            self.imat = self.interface_matrix(wavenumber, wavevector=wavevector)

    def epsilon_1_oscillator(
        self, wavenumber: np.ndarray, orientation: str = "pe"
    ) -> np.ndarray:
        """Returns the scalar dielectric function.

        Evaluated at frequencies wavenumber assuming a 1 oscillator Lorentz model of
        the dielectric. For materials whose phonon frequencies are set to nil returns
        eps_inf

        Args:
            wavenumber (np.ndarray): The probe frequencies in inverse centimetres
            orientation (str): The crystal orientation, accepts either 'a-cut' or 'c-cut'

        Returns:
            np.ndarray: The dielectric constant evaluated at input frequencies wavenumber

        """
        if orientation == "pe":
            wlo, wto, eps_inf, gam = (
                self.wlo_pe,
                self.wto_pe,
                self.eps_inf_pe,
                self.gamma,
            )
        elif orientation == "pa":
            wlo, wto, eps_inf, gam = (
                self.wlo_pa,
                self.wto_pa,
                self.eps_inf_pa,
                self.gamma,
            )
        elif orientation == "pe_ax":
            wlo, wto, eps_inf, gam = (
                self.wlo_ax_pe,
                self.wto_ax_pe,
                self.eps_inf_pe,
                self.gamma_ax_pe,
            )

        eps = (
            eps_inf
            * (wlo**2 - wavenumber * (wavenumber + 1j * gam))
            / (wto**2 - wavenumber * (wavenumber + 1j * gam))
        )

        return eps

    def epsilon_tensor(self, wavenumber: np.ndarray) -> np.ndarray:
        """Returns the permittivity tensor evaluated at frequencies wavenumber.

        This assumes a 1 oscillator Lorentz model of the dielectric and that the
        material is orientated so it's c-axis is parallel or perpendicular to the
        growth direction. The tensor is assumed to be diagonal

        Args:
            wavenumber (np.ndarray): The probe frequencies in inverse centimetres

        Returns:
            np.ndarray: The dielectric tensor evaluated at input frequencies wavenumber

        """

        eps = np.zeros((len(wavenumber), 3, 3), dtype=complex)

        if self.orientation == "c-cut":
            eps[:, 0, 0] = self.epsilon_1_oscillator(wavenumber, "pe")
            eps[:, 1, 1] = self.epsilon_1_oscillator(wavenumber, "pe")
            eps[:, 2, 2] = self.epsilon_1_oscillator(wavenumber, "pa")
            if hasattr(self, "_wlo_ax_pe"):
                eps[:, 0, 0] += self.epsilon_1_oscillator(wavenumber, "pe_ax")
                eps[:, 1, 1] += self.epsilon_1_oscillator(wavenumber, "pe_ax")
        elif self.orientation == "a-cut":
            eps[:, 0, 0] = self.epsilon_1_oscillator(wavenumber, "pa")
            eps[:, 1, 1] = self.epsilon_1_oscillator(wavenumber, "pe")
            eps[:, 2, 2] = self.epsilon_1_oscillator(wavenumber, "pe")
            if hasattr(self, "_wlo_ax_pe"):
                eps[:, 1, 1] += self.epsilon_1_oscillator(wavenumber, "pe_ax")
                eps[:, 2, 2] += self.epsilon_1_oscillator(wavenumber, "pe_ax")

        return eps

    def mu_tensor(self, wavenumber: np.ndarray) -> np.ndarray:
        """Returns the permeability tensor evaluated at frequencies wavenumber.

        We assume magnetically inactive media so this is just the identity matrix

        Args:
            wavenumber (np.ndarray): The probe frequencies in inverse centimetres

        Returns:
            np.ndarray: The permeability tensor for non magnetic media

        """
        mu = np.zeros((3, 3), dtype=complex)
        mu[0, 0] = 1
        mu[1, 1] = 1
        mu[2, 2] = 1

        return mu

    def layer_dispersion(
        self,
        wavenumber: np.ndarray,
        angle: float = None,
        wavevector: np.ndarray = np.ones(1),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the dispersion of the system modes.

        We solve for the photonic and the longitudinal and transverse
        phonon eigenmodes of an anisotropic medium, described by a diagonal
        permeability and permitivitty

        Args:
            wavenumber (np.ndarray): The probe frequencies in inverse centimetres
            angle (float, optional): The incident angle in degrees, optional defaults to None
            wavevector (np.ndarray, optional): The probe in-plane wavevectors in inverse
                centimetres. This defaults to 1

        Returns:
            tuple: The first element is the calculated out-of-plane wavevectors of the
                layers eigenmodes. The second element is those modes eigenvectors, holding
                the field components in orde Ex, Ey, Xx, Xy, Xz

        """
        # Checks for non-array inputs and converts to numpy arrays
        if (wavevector is not None) and type(wavevector) != np.ndarray:
            wavevector = np.array([wavevector])
        if type(wavenumber) != np.ndarray:
            wavenumber = np.array(wavenumber)

        # Checks for an input angle, if present calculates zeta using this else
        # uses the input wavevector array
        if angle:
            wavevector = np.zeros(
                1
            )  # Creates a dummy wavevector to ease code maintanence below
            arrays = [(wavenumber), (wavevector)]
            arr = np.array(list(product(*arrays)))
            zeta = np.zeros_like(wavenumber)
            zeta[:] = np.sin(angle)
        else:
            arrays = [(wavenumber), (wavevector)]
            arr = np.array(list(product(*arrays)))
            zeta = arr[:, 1] / arr[:, 0]

        if self.beta_l == 0:
            # Creates an array of len(zeta) 5 x 5 identity matrices
            It = np.identity(2)
            It = np.swapaxes(np.broadcast_to(It[..., None], (2, 2, len(zeta))), 0, 2)
            # Creates an array of len(zeta) 5 x 5 null matrices
            tAit = np.zeros_like(It, dtype=complex)
            tAit[:, 0, 0] = zeta**2 / self.eps_inf[2, 2] - self.mu[1, 1]
            tAit[:, 1, 1] = -self.mu[0, 0]
            tCt = np.zeros_like(It, dtype=complex)
            tCt[:, 0, 0] = self.eps_inf[0, 0]
            tCt[:, 1, 1] = self.eps_inf[1, 1] - zeta**2 / self.mu[2, 2]

            fm = np.matmul(tAit, tCt)

            sq_eigs, vecs = np.linalg.eig(-fm)
            eigs = np.sqrt(sq_eigs + 0j)

            eigs = eigs.reshape((len(wavevector), len(wavenumber), 2), order="F")
            vecs = vecs.reshape((len(wavevector), len(wavenumber), 2, 2), order="F")

            eigs_ph, vecs_ph = self.polarisation_comparison(
                eigs[:, :, :], vecs[:, :, :, :]
            )

            ts0, ts1, ts2 = eigs_ph.shape
            zm = np.zeros((ts0, ts1, ts2 + 1))

            sorted_eigs = np.concatenate((eigs_ph, zm), axis=2)
            sorted_eigs_r = np.concatenate((-eigs_ph, zm), axis=2)

            ts0, ts1, ts2, ts3 = vecs_ph.shape
            zm1 = np.zeros((ts0, ts1, ts2 + 6, ts3))
            zm2 = np.zeros((ts0, ts1, ts2 + 8, ts3 + 1))

            sorted_vecs = np.concatenate(
                (np.concatenate((vecs_ph, zm1), axis=2), zm2), axis=3
            )
            sorted_vecs_r = np.concatenate(
                (np.concatenate((vecs_ph, zm1), axis=2), zm2), axis=3
            )

        else:

            # Creates an array of len(zeta) 5 x 5 identity matrices
            It = np.identity(5)
            It = np.swapaxes(np.broadcast_to(It[..., None], (5, 5, len(zeta))), 0, 2)
            # Creates an array of len(zeta) 5 x 5 null matrices
            Zt = np.zeros_like(It)

            tAt = np.zeros_like(It, dtype=complex)
            tBt = np.zeros_like(It, dtype=complex)

            tAt[:, 0, 0] = self.eps_inf[0, 0] * (
                -(zeta**2) / self.eps_inf[2, 2] + self.mu[1, 1]
            )
            tAt[:, 0, 2] = self.alpha[0, 0] * (
                -(zeta**2) / self.eps_inf[2, 2] + self.mu[1, 1]
            )

            tAt[:, 1, 1] = self.mu[0, 0] * (
                -(zeta**2) / self.mu[2, 2] + self.eps_inf[1, 1]
            )
            tAt[:, 1, 3] = self.alpha[1, 1] * self.mu[0, 0]

            tAt[:, 2, 0] = -2 * self.alpha[0, 0] / arr[:, 0] ** 2 / self.beta_c**2
            tAt[:, 2, 2] = (
                2
                * (
                    self.wto_pe**2
                    - arr[:, 0] * (arr[:, 0] + 1j * self.gamma)
                    - self.beta_l**2 * zeta**2 * arr[:, 0] ** 2
                )
                / arr[:, 0] ** 2
                / self.beta_c**2
            )
            tAt[:, 3, 1] = -2 * self.alpha[1, 1] / arr[:, 0] ** 2 / self.beta_c**2
            tAt[:, 3, 3] = (
                2
                * (
                    self.wto_pe**2
                    - arr[:, 0] * (arr[:, 0] + 1j * self.gamma)
                    - 0.5 * self.beta_c**2 * zeta**2 * arr[:, 0] ** 2
                )
                / arr[:, 0] ** 2
                / self.beta_c**2
            )

            tAt[:, 4, 4] = (
                self.wto_pa**2
                - arr[:, 0] * (arr[:, 0] + 1j * self.gamma)
                - 0.5 * self.beta_c**2 * zeta**2 * arr[:, 0] ** 2
            ) / arr[:, 0] ** 2 / self.beta_l**2 - (
                self.alpha[2, 2] ** 2
                * self.mu[1, 1]
                / arr[:, 0] ** 2
                / self.beta_l**2
                / (zeta**2 - self.eps_inf[2, 2] * self.mu[1, 1])
            )

            tBt[:, 0, 4] = -zeta * self.alpha[2, 2] / self.eps_inf[2, 2]

            tBt[:, 2, 4] = (
                -2
                * zeta
                * (self.beta_c**2 / 2 + self.beta_l**2 - 2 * self.beta_t**2)
                / self.beta_c**2
            )

            tBt[:, 4, 0] = (
                -self.alpha[2, 2]
                * zeta
                / (
                    self.beta_l**2
                    * arr[:, 0] ** 2
                    * (zeta**2 - self.eps_inf[2, 2] * self.mu[1, 1])
                )
            )

            tBt[:, 4, 2] = (
                -1
                * zeta
                * (self.beta_c**2 / 2 + self.beta_l**2 - 2 * self.beta_t**2)
                / self.beta_l**2
            )

            M3 = -np.concatenate((np.hstack((Zt, It)), np.hstack((tAt, tBt))), axis=2)

            eigs, vecs = np.linalg.eig(M3)

            # Reshape into a useable array
            eigs = eigs.reshape((len(wavevector), len(wavenumber), 10), order="F")
            vecs = vecs.reshape((len(wavevector), len(wavenumber), 10, 10), order="F")

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

            eigs_photon = np.concatenate((eigs_r[:, :, 3:5], eigs_r[:, :, 5:7]), axis=2)
            vecs_photon = np.concatenate(
                (vecs_r[:, :, :, 3:5], vecs_r[:, :, :, 5:7]), axis=3
            )

            # Initial sort for photon modes
            a = -1
            order = np.lexsort((eigs_photon.imag, -np.sign(eigs_photon.imag)), axis=a)
            m, n, k = eigs_photon.shape
            idx = np.ogrid[:m, :n, :k]
            idx[-1] = order
            eigs_photon_s = eigs_photon[tuple(idx)]

            m, n, k, o = vecs_photon.shape
            idx = np.ogrid[:m, :n, :k, :o]
            order = np.repeat(order[:, :, np.newaxis, :], 10, axis=2)
            idx[-1] = order
            vecs_photon_s = vecs_photon[tuple(idx)]

            eigs_ph, vecs_ph = self.polarisation_comparison(
                eigs_photon_s[:, :, 0:2], vecs_photon_s[:, :, :, 0:2]
            )
            eigs_ph_r, vecs_ph_r = self.polarisation_comparison(
                eigs_photon_s[:, :, 2:], vecs_photon_s[:, :, :, 2:]
            )

            # New Sort, more robust for phonon modes
            eigs_phonon = np.concatenate((eigs_r[:, :, :3], eigs_r[:, :, 7:]), axis=2)
            vecs_phonon = np.concatenate(
                (vecs_r[:, :, :, :3], vecs_r[:, :, :, 7:]), axis=3
            )

            a = -1
            order = np.lexsort(
                (eigs_phonon.real, eigs_phonon.imag, -np.sign(eigs_phonon.real)), axis=a
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

            eigs_to, vecs_to = self.polarisation_comparison(
                eigs_r_pho[:, :, 4:], vecs_r_pho[:, :, :, 4:]
            )

            sorted_eigs = np.concatenate(
                (np.concatenate((eigs_ph, eigs_to), axis=2), eigs_r_pho[:, :, 3:4]),
                axis=2,
            )

            sorted_vecs = np.concatenate(
                (np.concatenate((vecs_ph, vecs_to), axis=3), vecs_r_pho[:, :, :, 3:4]),
                axis=3,
            )

            # Reverse propagation
            eigs_to_r, vecs_to_r = self.polarisation_comparison(
                eigs_r_pho[:, :, 0:2], vecs_r_pho[:, :, :, 0:2]
            )

            sorted_eigs_r = np.concatenate(
                (np.concatenate((eigs_ph_r, eigs_to_r), axis=2), eigs_r_pho[:, :, 2:3]),
                axis=2,
            )

            sorted_vecs_r = np.concatenate(
                (
                    np.concatenate((vecs_ph_r, vecs_to_r), axis=3),
                    vecs_r_pho[:, :, :, 2:3],
                ),
                axis=3,
            )

        if angle:
            eigs_out = np.concatenate([sorted_eigs[0], sorted_eigs_r[0]], axis=1)
            vecs_out = np.concatenate([sorted_vecs[0], sorted_vecs_r[0]], axis=2)
            return eigs_out, vecs_out
        else:
            eigs_out = np.concatenate([sorted_eigs, sorted_eigs_r], axis=2)
            vecs_out = np.concatenate([sorted_vecs, sorted_vecs_r], axis=3)

            return eigs_out, vecs_out

    def polarisation_comparison(
        self, eigs: np.ndarray, vecs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Function to compare the eigenmode polarisation states.

        Args:
            eigs (np.ndarray): The layer eigenvalues
            vecs (np.ndarray)): The corresponding eigenvectors

        Returns:
            tuple: The first element is the calculated out-of-plane wavevectors of the
                layers eigenmodes. The second element is those modes eigenvectors, holding
                the field components in orde Ex, Ey, Xx, Xy, Xz. The eigenmodes have been
                sorted according to their polarisation state.

        """
        ldx = vecs.shape[-2]

        fom = np.abs(vecs[:, :, 0]) ** 2 / (
            np.abs(vecs[:, :, 0]) ** 2 + np.abs(vecs[:, :, 1]) ** 2
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

    # flake8: noqa: C901
    def field_generator(
        self, wavenumber: np.ndarray, angle: float = None, wavevector: np.ndarray = None
    ) -> np.ndarray:
        """Calculates the full vector of fields in the layer.

        Takes the eigenvalues and fields calculated for a given layer and finds
        the full vector of fields required to apply the boundary conditions. The
        fields are normalised to the electric field parallel to the interface, for
        TM polarised modes this means they are normalised with respect to the
        x-component while TE polarised modes they are normalised with respect
        to the y-comp

        Args:
            wavenumber (np.ndarray): The probe frequencies in inverse centimetres
            angle (float, optional): The incident angle in degrees, optional defaults to None
            wavevector (np.ndarray, optional): The probe in-plane wavevectors in inverse
                centimetres. This defaults to 1

        Returns:
            np.ndarray: the full array of fields necessary to solve the boundary matching
                problem for all wavevector, wavenumber. Fields are outputted in order
                H, E, P, X with Cartesian components for each ordered x, y, z and are
                normalised as described above

        """
        # Checks for non-array inputs and converts to numpy arrays
        if (wavevector is not None) and type(wavevector) != np.ndarray:
            wavevector = np.array([wavevector])
        if type(wavenumber) != np.ndarray:
            wavenumber = np.array(wavenumber)

        if self.material == "vacuum":
            if angle:
                vecs = self.vecs[:, :5, :]
            else:
                vecs = self.vecs[:, :, :5, :]
        else:
            if angle:
                vecs = self.vecs[:, 5:, :]
            else:
                vecs = self.vecs[:, :, 5:, :]

        eigs = self.eigs
        # Checks for equal length inputs, assumed to correspond to a cut through
        # (wavevector, wavenumber) space
        if angle:
            wavevector = np.zeros(
                1
            )  # Creates a dummy wavevector to ease code maintanence below
            arrays = [(wavenumber), (wavevector)]
            arr = np.array(list(product(*arrays)))
            zeta = np.zeros_like(wavenumber)
            zeta[:] = np.sin(angle)
            vecs = np.expand_dims(vecs, axis=0)
            eigs = np.expand_dims(eigs, axis=0)
            zeta = np.expand_dims(zeta, axis=0)
        else:
            arrays = [(wavenumber), (wavevector)]
            arr = np.array(list(product(*arrays)))
            zeta = arr[:, 1] / arr[:, 0]
            # zeta = np.transpose(zeta.reshape((len(arrays[0]), len(arrays[1]))))
            # Reshape input arrays for calculation
            # wavevector = np.transpose(arr[:, 1].reshape((len(arrays[0]), len(arrays[1]))))
            # wavenumber = np.transpose(arr[:, 0].reshape((len(arrays[0]), len(arrays[1]))))
            # vecs = np.expand_dims(vecs, axis=0)
            # eigs = np.expand_dims(eigs, axis=0)
            # zeta = np.expand_dims(zeta, axis=0)

        # Initialize empty output vector
        field_vec = np.zeros((len(arrays[1]), len(arrays[0]), 10, 12), dtype=complex)

        # Fill components which pre-exist in the input
        field_vec[:, :, :, 3] = vecs[:, :, 0, :]
        field_vec[:, :, :, 4] = vecs[:, :, 1, :]

        Ex0 = vecs[:, :, 0, :]
        Ey0 = vecs[:, :, 1, :]

        if self.material == "vacuum":

            # Broadcast zeta to the leading dimensions of the other arrays
            if angle:
                zetaf = np.repeat(zeta[:, :, np.newaxis], 10, axis=2)
            else:
                zetaf = np.reshape(zeta, (len(wavevector), len(wavenumber)), order="F")
                zetaf = np.repeat(zetaf[:, :, np.newaxis], 10, axis=2)

            field_vec[:, :, :, 0] = -eigs * Ey0 / self.mu[0, 0]
            field_vec[:, :, :, 1] = (
                Ex0
                * eigs
                * self.eps_inf[2, 2]
                / (zetaf**2 - self.eps_inf[2, 2] * self.mu[1, 1])
            )
            field_vec[:, :, :, 2] = zetaf * Ey0 / self.mu[2, 2]
            # Calculates the z-component of the electric field from Eq. in the tex file
            field_vec[:, :, :, 5] = (
                Ex0 * eigs * zetaf / (zetaf**2 - self.eps_inf[2, 2] * self.mu[1, 1])
            )

        else:
            field_vec[:, :, :, 9] = vecs[:, :, 2, :]
            field_vec[:, :, :, 10] = vecs[:, :, 3, :]
            field_vec[:, :, :, 11] = -vecs[:, :, 4, :]

            # Broadcast zeta to the leading dimensions of the other arrays
            if angle:
                zetaf = np.repeat(zeta[:, :, np.newaxis], 10, axis=2)
            else:
                zetaf = np.reshape(zeta, (len(wavevector), len(wavenumber)), order="F")
                zetaf = np.repeat(zetaf[:, :, np.newaxis], 10, axis=2)

            # Xx0 = vecs[:, :, 2, :]
            # Xy0 = vecs[:, :, 3, :]
            Xz0 = -vecs[:, :, 4, :]

            # Calculates the z-component of the electric field from Eq. in the tex file
            field_vec[:, :, :, 5] = (
                Ex0 * eigs * zetaf + self.alpha[2, 2] * self.mu[1, 1] * Xz0
            ) / (zetaf**2 - self.eps_inf[2, 2] * self.mu[1, 1])

            # Calculates the x-component of the magnetic field from Eq. in the tex file
            field_vec[:, :, :, 0] = -eigs * Ey0 / self.mu[0, 0]

            # Calculates the y-component of the magnetic field from Eq. in the tex file
            field_vec[:, :, :, 1] = (
                Ex0 * eigs * self.eps_inf[2, 2]
                + self.alpha[2, 2] * self.mu[1, 1] * Xz0 * zetaf
            ) / (zetaf**2 - self.eps_inf[2, 2] * self.mu[1, 1])

            # Calculates the z-component of the magnetic field from Eq. in the tex file
            field_vec[:, :, :, 2] = zetaf * Ey0 / self.mu[2, 2]

            # Calculates the x-component of the polarization field from Eq. in the tex
            field_vec[:, :, :, 6] = -(
                (
                    Ex0
                    * (
                        zetaf**2
                        + self.eps_inf[2, 2] * eigs**2
                        - self.eps_inf[2, 2] * self.mu[1, 1]
                    )
                    + self.alpha[2, 2] * self.mu[1, 1] * Xz0 * zetaf * eigs
                )
                / (zetaf**2 - self.eps_inf[2, 2] * self.mu[1, 1])
            )

            # Calculates the y-component of the polarization field from Eq. in the tex
            field_vec[:, :, :, 7] = (
                -(1 - eigs**2 / self.mu[0, 0] - zetaf**2 / self.mu[2, 2]) * Ey0
            )

            # Calculates the z-component of the polarization field from Eq. in the tex
            field_vec[:, :, :, 8] = (
                Ex0 * zetaf * eigs * (self.eps_inf[2, 2] - 1)
                + self.alpha[2, 2] * (zetaf**2 - self.mu[1, 1]) * Xz0
            ) / (zetaf**2 - self.eps_inf[2, 2] * self.mu[1, 1])

        # Compares the value of the x and y electric Fields
        ivec = np.sum(np.sum(np.abs(field_vec[:, :, :, 3]), axis=1), axis=0) > np.sum(
            np.sum(np.abs(field_vec[:, :, :, 4]), axis=1), axis=0
        )

        field_vec_o = np.zeros_like(field_vec)

        for idx, iv in enumerate(ivec):
            if iv:
                if np.any(field_vec[:, :, idx, 3, None] == 0):
                    field_vec_o[:, :, idx, :] = field_vec[:, :, idx, :]
                else:
                    field_vec_o[:, :, idx, :] = (
                        field_vec[:, :, idx, :] / field_vec[:, :, idx, 3, None]
                    )
            else:
                if np.any(field_vec[:, :, idx, 4, None] == 0):
                    field_vec_o[:, :, idx, :] = field_vec[:, :, idx, :]
                else:
                    field_vec_o[:, :, idx, :] = (
                        field_vec[:, :, idx, :] / field_vec[:, :, idx, 4, None]
                    )

        return field_vec_o

    def interface_matrix(
        self, wavenumber: np.ndarray, angle: float = None, wavevector: np.ndarray = None
    ) -> np.ndarray:
        """Calculates the interface matrix given that layer's eigenvalues
        corresponding to the out-of-plane wavevector and eigenvectors

        Parameters
        ----------

        wavenumber : 1d array
            a 1d array, containing the probing wavenumbers

        eigs : array
            a 2d array of dimensions (len(wavenumber) x 10) where 10 is the number of
            modes in the problem. This corresponds to the out of plane wavevector
            divided by the wavenumber

        fields : array
            the array of sorted fields


        Returns
        -------

        interface_matrix : array
            a matrix of size (len(wavenumber)x10x10) containing the propagation factors for
            each eigenmode
        """
        # Checks for non-array inputs and converts to numpy arrays
        if (wavevector is not None) and type(wavevector) != np.ndarray:
            wavevector = np.array([wavevector])
        if type(wavenumber) != np.ndarray:
            wavenumber = np.array(wavenumber)

        # Checks for an input angle, if present calculates zeta using this else
        # uses the input wavevector array
        if angle:
            zeta = np.zeros_like(wavenumber)
            zeta[:] = np.sin(angle)
            interface_matrix = np.zeros((1, len(wavenumber), 10, 10), dtype=complex)
            zeta = zeta[np.newaxis, :, np.newaxis]
        else:
            arrays = [(wavenumber), (wavevector)]
            arr = np.array(list(product(*arrays)))
            zeta = arr[:, 1] / arr[:, 0]
            zeta = np.reshape(zeta, (len(wavevector), len(wavenumber)), order="F")
            interface_matrix = np.zeros(
                (len(wavevector), len(wavenumber), 10, 10), dtype=complex
            )
            zeta = zeta[:, :, np.newaxis]

        # In-plane electric field (x)
        interface_matrix[:, :, 0, :] = self.fields[:, :, :, 3]
        # In-plane electric field (y)
        interface_matrix[:, :, 1, :] = self.fields[:, :, :, 4]

        # In-plane magnetic field (x)
        interface_matrix[:, :, 2, :] = self.fields[:, :, :, 0]
        # In-plane magnetic field (y)
        interface_matrix[:, :, 3, :] = self.fields[:, :, :, 1]

        # In-plane ionic displacement field (x)
        interface_matrix[:, :, 4, :] = self.fields[:, :, :, 9]
        # In-plane ionic displacement ield (y)
        interface_matrix[:, :, 5, :] = self.fields[:, :, :, 10]
        #  In-plane ionic displacement field (y)
        interface_matrix[:, :, 6, :] = self.fields[:, :, :, 11]

        # In-plane stress tensor (x)
        interface_matrix[:, :, 7, :] = (
            0.5
            * self.beta_c**2
            * (self.eigs * self.fields[:, :, :, 9] + zeta * self.fields[:, :, :, 11])
        )

        # In-plane stress tensor (y)
        interface_matrix[:, :, 8, :] = (
            0.5 * self.beta_c**2 * (self.eigs * self.fields[:, :, :, 10])
        )

        # In-plane stress tensor (z)
        interface_matrix[
            :, :, 9, :
        ] = +0.5 * self.beta_l**2 * self.eigs * self.fields[:, :, :, 11] + (
            self.beta_l**2 - 2 * self.beta_t**2
        ) * (
            zeta * self.fields[:, :, :, 9]
        )

        if angle:
            return interface_matrix[0]
        else:
            return interface_matrix
