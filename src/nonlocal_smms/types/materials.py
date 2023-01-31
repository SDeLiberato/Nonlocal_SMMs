"""Generic material dielectric function models."""
from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
from typing import List
from typing import Union

import numpy as np
import numpy.ma as ma
import numpy.typing as npt

class Material:
    """A generic material."""

    def refractive_index(
        self, frequency: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        """The refractive index.

        Args:
            frequency (npt.NDArray[np.float64]): Driving frequency in rad / s

        Returns:
            npt.NDArray[np.complex128]: Complex refractive index at frequency.
        """
        return np.sqrt(self.dielectric_function(frequency))

    def dielectric_function(
        self, frequency: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        """Complex dielectric function with a tensorflow backend.

        This will be implemented directly by child classes.

        Args:
            frequency (npt.NDArray[np.float64]): Driving frequency in rad / s

        Raises:
            NotImplementedError: If called from parent directly.
        """
        raise NotImplementedError("dielectric function unimplemented")

    def dielectric_function_tensor(
        self, frequency: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        """Complex dielectric function with a tensorflow backend.

        Scalar materials are uprated to isotropic tensors

        Args:
            frequency (npt.NDArray[np.float64]): Driving frequency in rad / s
                with shape (num_frequency, 1, num_angles)

        Returns:
            npt.NDArray[np.complex128]: The dielectric function as a tensor of
                shape (num_frequency, 1, num_angles, 3, 3). Isotropic materials
                are uprated to diagonal tensors in the latter dimensions.
        """
        V = np.shape(frequency)[0]  # We need to unpack separately to
        W = np.shape(frequency)[2]  # execute without eager
        dielectric_function = self.dielectric_function(frequency)
        # Matching on the full shape fails when using a tabulated dielectric
        # function. A numerical function is of shape (Batch, 1, Num_Angles)
        # while the tabulated functions from tfp are of shape (Batch, None, Num_Angles)
        # Matching on full shape fails to promote tabulated functions to Tensors
        if len(dielectric_function.shape) == len(frequency.shape):
            extended = np.stack(
                 [dielectric_function, dielectric_function, dielectric_function], axis=3
             )
            result = np.zeros((V, 1, W, 3, 3), dtype=np.complex128)
            for idx in range(9):
                result[:, :, :, idx // 3, idx % 3] = extended[idx // 3, idx % 3]
            return result
        else:  # A diagonal material
            return dielectric_function

    def relative_permeability(
        self, frequency: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        """Relative permeability of the material as a tensor.

        All materials are assumed to be non-magnetic at present.

        Args:
            frequency (npt.NDArray[np.float64]): Driving frequency in rad / s

        Returns:
            npt.NDArray[np.float64]: The relative permeability of the layer
        """
        V = np.shape(frequency)[0]  # We need to unpack separately to
        W = np.shape(frequency)[2]  # execute without eager
        result = np.zeros((V, 1, W, 3, 3), dtype=np.complex128)
        result[:, :, :, 0, 0] = 1.0 + 0j
        result[:, :, :, 1, 1] = 1.0 + 0j
        result[:, :, :, 2, 2] = 1.0 + 0j
        return result

    def gamma(
        self, frequency: npt.NDArray[np.float64], zeta: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        """Complex dielectric function with a tensorflow backend.

        This will be implemented directly by child classes.

        Args:
            frequency (npt.NDArray[np.float64]): Driving frequency in rad / s

        Raises:
            NotImplementedError: If called from parent directly.
        """
        raise NotImplementedError("dielectric function unimplemented")

    def is_local(self) -> bool:
        """Query whether the material is local, or nonlocal.

        Returns:
            bool: True if the layer is local.

        Raises:
            NotImplementedError: If called from the parent class.
        """
        raise NotImplementedError("locality check not implemented")

    def eig(
        self,
        frequency: npt.NDArray[np.float64],
        zeta: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        """Calculates sorted eigenvalues in the material.

        We assume all materials are diagonal, so the non-zero elements of
        the matrix Delta (Eq. 8) are only on the diagonal. The eigenvalues returned
        are sorted as described in doi: 10.1364/JOSAB.34.002128.
        The method can be extended to full anisotropic materials
        following 10.1364/JOSAB.34.002128.

        Args:
            frequency (npt.NDArray[np.float64]): Driving frequency in rad / s
            zeta (npt.NDArray[np.float64]): Zeta at the driving frequency

        Returns:
            npt.NDArray[np.complex128]: Sorted eigenfrequencies in the layer.
                These are in order
                [p-polarized transmitted, s-polarised transmitted,
                        p-polarized reflected, s-polarised reflected] and are
                        normalised by the free-space wavevector.
        """
        if self.is_local():
            local_eigenvalues = self.local_eig(frequency, zeta)
            return self.promote(local_eigenvalues)
        else:
            return self.nonlocal_eig(frequency, zeta)

    def local_eigenvalues(
        self,
        frequency: npt.NDArray[np.float64],
        zeta: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        """Calculates sorted eigenvalues in a local material.

        We assume all materials are diagonal, so the non-zero elements of
        the matrix Delta (Eq. 8) are only on the diagonal. The eigenvalues returned
        are sorted as described in doi: 10.1364/JOSAB.34.002128.
        The method can be extended to full anisotropic materials
        following 10.1364/JOSAB.34.002128.

        Args:
            frequency (npt.NDArray[np.float64]): Driving frequency in rad / s
            zeta (npt.NDArray[np.float64]): Zeta at the driving frequency

        Returns:
            npt.NDArray[np.complex128]: Sorted eigenfrequencies in the layer.
             [p-polarized transmitted, s-polarised transmitted,
                        p-polarized reflected, s-polarised reflected] and are
                        normalised by the free-space wavevector.
        """

        V = np.shape(frequency)[0]  # We need to unpack separately to
        W = np.shape(frequency)[2]  # executre without eager

        # An array of V, 1, W 2x2 identity matrices
        identity = np.zeros((V, 1, W, 2, 2), dtype=np.complex128)
        identity[:, :, :] = np.diag(np.ones(2), dtype=np.complex128)

        mu = self.relative_permeability(frequency)
        epsilon = self.dielectric_function_tensor(frequency)

        # b in a diagonal material
        b = epsilon[:, :, :, 2, 2] * mu[:, :, :, 2, 2]
        # The non-zero bits of a3n
        a35 = -zeta * mu[:, :, :, 2, 2] / b
        # The non-zero bits of a6n
        a62 = zeta * epsilon[:, :, :, 2, 2] / b
        # The non-zero bits of \Delta
        delta_12 = mu[:, :, :, 1, 1] + zeta * a35
        delta_21 = epsilon[:, :, :, 0, 0]
        delta_34 = mu[:, :, :, 0, 0]
        delta_43 = epsilon[:, :, :, 1, 1] - zeta * a62

        delta = np.zeros((V, 1, W, 4, 4), dtype=np.complex128)
        delta[:, :, :, 0, 1] = delta_12
        delta[:, :, :, 1, 0] = delta_21
        delta[:, :, :, 2, 3] = delta_34
        delta[:, :, :, 3, 2] = delta_43

        return delta


    def local_eig(
        self,
        frequency: npt.NDArray[np.float64],
        zeta: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        """Calculates sorted eigenvalues in a local material.

        We assume all materials are diagonal, so the non-zero elements of
        the matrix Delta (Eq. 8) are only on the diagonal. The eigenvalues returned
        are sorted as described in doi: 10.1364/JOSAB.34.002128.
        The method can be extended to full anisotropic materials
        following 10.1364/JOSAB.34.002128.

        Args:
            frequency (npt.NDArray[np.float64]): Driving frequency in rad / s
            zeta (npt.NDArray[np.float64]): Zeta at the driving frequency

        Returns:
            npt.NDArray[np.complex128]: Sorted eigenfrequencies in the layer.
             [p-polarized transmitted, s-polarised transmitted,
                        p-polarized reflected, s-polarised reflected] and are
                        normalised by the free-space wavevector.
        """

        eigenmatrix = self.local_eigenmatrix(frequency, zeta)

        eigenvalues, eigenvectors = np.linalg.eig(eigenmatrix)

        ordinary_eigenvalues = ma.masked_array(
            eigenvalues, mask=(eigenvectors[:, :, :, :, 0] == 0)
        )

        extraordinary_eigenvalues = ma.masked_array(
            eigenvalues, mask=(eigenvectors[:, :, :, :, 2] == 0)
        )




        # We can find the eigenvalues analytically
        ordinary_eigenvalue_minus = -np.sqrt(Delta_12) * np.sqrt(Delta_21)
        extraordinary_eigenvalue_minus = -np.sqrt(Delta_34) * np.sqrt(Delta_43)

        ordinary_masked_minus = ma.masked_array(
            ordinary_eigenvalue_minus,
            mask=~(np.imag(ordinary_eigenvalue_minus) == 0)
        )
        ordinary_anti_masked_minus = ma.masked_array(
            ordinary_eigenvalue_minus,
            mask=(np.imag(ordinary_eigenvalue_minus) == 0)
        )
        extraordinary_masked_minus = ma.masked_array(
            extraordinary_eigenvalue_minus,
            mask=~(np.imag(extraordinary_eigenvalue_minus) == 0)
        )
        extraordinary_anti_masked_minus = ma.masked_array(
            extraordinary_eigenvalue_minus,
            mask=(np.imag(extraordinary_eigenvalue_minus) == 0)
        )

        ordinary_up = (
            ordinary_masked_minus[~ordinary_masked_minus.mask].data
            * np.sign(np.real(ordinary_eigenvalue_minus))
            + ordinary_anti_masked_minus[~ordinary_anti_masked_minus.mask].data
            * np.sign(np.imag(ordinary_eigenvalue_minus))
        )

        extraordinary_up = (
            extraordinary_masked_minus[~extraordinary_masked_minus.mask].data
            * np.sign(np.real(extraordinary_eigenvalue_minus))
            + extraordinary_anti_masked_minus[
                    ~extraordinary_anti_masked_minus.mask
                ].data
            * np.sign(np.imag(extraordinary_eigenvalue_minus))
        )

        values = [extraordinary_up, ordinary_up, -extraordinary_up, -ordinary_up]

        values = np.stack(values, axis=0)
        values = np.transpose(values, perm=[1, 2, 3, 0])

        return values


    def nonlocal_eig(
        self,
        frequency: npt.NDArray[np.float64],
        zeta: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        """Calculates sorted eigenvalues in a nonlocal material.

        We assume all materials are diagonal, so the non-zero elements of
        the matrix Delta (Eq. 8) are only on the diagonal. The eigenvalues returned
        are sorted as described in doi: 10.1364/JOSAB.34.002128.
        The method can be extended to full anisotropic materials
        following 10.1364/JOSAB.34.002128.

        Args:
            frequency (npt.NDArray[np.float64]): Driving frequency in rad / s
            zeta (npt.NDArray[np.float64]): Zeta at the driving frequency

        Returns:
            npt.NDArray[np.complex128]: Sorted eigenfrequencies in the layer.
             [p-polarized transmitted, s-polarised transmitted,
                        p-polarized reflected, s-polarised reflected] and are
                        normalised by the free-space wavevector.
        """
        gamma = self.gamma(frequency, zeta)
        eigenvalues = np.linalg.eigvals(gamma)

    def sort_eigenvalues(
        self,
        eigenvalues: npt.NDArray[np.complex128]
        ) -> npt.NDArray[np.complex128]:






    def promote_eigenvalues(
            self,
            local_eigenvalues: npt.NDArray[np.complex128]
        ) -> npt.NDArray[np.complex128]:
        """Promote local eigenvalues to full nonlocal ones.

        Args:
            local_eigenvalues (npt.NDArray[np.complex128]): Local photonic
                eigenvalues in the layer.

        Returns:
            npt.NDArray[np.complex128]: Eigenvalues which are shape-compatible
                with the full non-local result
        """



      #   # Gamma is a 10x10 Block matrix.
      #   # The upper left block is the identity matrix (Eq. C36)
      #   a = np.zeros((V, 1, W, 5, 5), dtype=np.complex128)
      #   a[:, :, :] = np.diag(np.ones(5), dtype=np.complex128)

      #   # The upper right block is filled with zeros
      #   b = np.zeros((V, 1, W, 5, 5), dtype=np.complex128)

      #   # The lower left block







@dataclass
class DrudeMaterial(Material):
    """An isotropic Drude material.

    A Drude material characterised by a high-frequency dielectric constant,
    longitudinal (plasma) frequency and a damping frequency.

    Args:
        epsilon_infinity (float): The dielectric function at infinite frequency.
        longitudinal_frequency (float): Plasma frequency in the medium.
        damping_frequency (float): Damping frequency in the medium.
        longitudinal_velocity (Optional[float]): Velocity of plasma propagation
            in m/s
    """

    epsilon_infinity: float
    longitudinal_frequency: float
    damping_frequency: float
    longitudinal_velocity: Optional[float] = None

    @property
    def epsilon_infinity(self) -> float:
        """Return the high frequency dielectric constant."""
        return self.__epsilon_infinity

    @epsilon_infinity.setter
    def epsilon_infinity(self, value: float) -> None:
        """Set the high frequency dielectric constant.

        Args:
            value: Value to set

        Raises:
            ValueError: If the high frequency dielectric constant is negative.
        """
        if value > 0.0:
            self.__epsilon_infinity = value
        else:
            raise ValueError("High frequency dielectric constant must be positive")

    @property
    def longitudinal_frequency(self) -> float:
        """Return the longitudinal frequency in rad / s."""
        return self.__longitudinal_frequency

    @longitudinal_frequency.setter
    def longitudinal_frequency(self, value: float) -> None:
        """Set the longitudinal frequency.

        Args:
            value: Value to set in 1 / cm

        Raises:
            ValueError: If the longitudinal frequency is negative.
        """
        if value > 0:
            self.__longitudinal_frequency = icm_to_omega(value)
        else:
            raise ValueError("Longitudinal frequency must be positive")

    @property
    def damping_frequency(self) -> float:
        """Return the damping frequency."""
        return self.__damping_frequency

    @damping_frequency.setter
    def damping_frequency(self, value: float) -> None:
        """Set the damping frequency.

        Args:
            value: Value to set in 1 / cm

        Raises:
            ValueError: If the damping frequency is negative.
        """
        if value > 0:
            self.__damping_frequency = icm_to_omega(value)
        else:
            raise ValueError("Damping frequency must be positive")

    def dielectric_function(
            self, frequency: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        """Drude model dielectric function.

        Args:
            frequency (npt.NDArray[np.float64]): Driving frequency in rad / s

        Returns:
            npt.NDArray[np.complex128]: Complex Drude dielectric function at
                frequency
        """
        return self.epsilon_infinity - self.longitudinal_frequency**2 / frequency / (
            frequency + 1j * self.damping_frequency
        )

    def is_local(self) -> bool:
        if self.longitudinal_velocity:
            return True
        else:
            return False

    def eigenmatrix(
            self, frequency: npt.NDArray[np.float64], zeta: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        if self.is_local():
            return self.local_eigenmatrix(frequency, zeta)
        else:
            return self.nonlocal_eigenmatrix(frequency, zeta)


@dataclass
class LorentzMaterial(Material):
    """An isotropic Lorentz material.

    A Drude material characterised by a high-frequency dielectric constant,
    longitudinal (plasma) frequency and a damping frequency.

    Args:
        epsilon_infinity (float): The dielectric function at infinite frequency.
        longitudinal_frequency (float): Plasma frequency in the medium.
        damping_frequency (float): Damping frequency in the medium.
    """

    epsilon_infinity: float
    longitudinal_frequency: float
    transverse_frequency: float
    damping_frequency: float

    @property
    def epsilon_infinity(self) -> float:
        """Return the high frequency dielectric constant."""
        return self.__epsilon_infinity

    @epsilon_infinity.setter
    def epsilon_infinity(self, value: float) -> None:
        """Set the high frequency dielectric constant.

        Args:
            value: Value to set

        Raises:
            ValueError: If the high frequency dielectric constant is negative.
        """
        if value > 0.0:
            self.__epsilon_infinity = value
        else:
            raise ValueError("High frequency dielectric constant must be positive")

    @property
    def longitudinal_frequency(self) -> float:
        """Return the longitudinal frequency in rad / s."""
        return self.__longitudinal_frequency

    @longitudinal_frequency.setter
    def longitudinal_frequency(self, value: float) -> None:
        """Set the longitudinal frequency.

        Args:
            value: Value to set

        Raises:
            ValueError: If the longitudinal frequency is negative.
        """
        if value > 0.0:
            self.__longitudinal_frequency = icm_to_omega(value)
        else:
            raise ValueError("Longitudinal frequency must be positive")

    @property
    def transverse_frequency(self) -> float:
        """Return the transverse frequency in rad / s."""
        return self.__transverse_frequency

    @transverse_frequency.setter
    def transverse_frequency(self, value: float) -> None:
        """Set the transverse frequency.

        Args:
            value: Value to set

        Raises:
            ValueError: If the transverse frequency is negative.
        """
        if value > 0:
            self.__transverse_frequency = icm_to_omega(value)
        else:
            raise ValueError("Transverse frequency must be positive")

    @property
    def damping_frequency(self) -> float:
        """Return the damping frequency."""
        return self.__damping_frequency

    @damping_frequency.setter
    def damping_frequency(self, value: float) -> None:
        """Set the damping frequency.

        Args:
            value: Value to set

        Raises:
            ValueError: If the damping frequency is negative.
        """
        if value > 0:
            self.__damping_frequency = icm_to_omega(value)
        else:
            raise ValueError("Damping frequency must be positive")

    def dielectric_function(
            self, frequency: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        """Drude model dielectric function.

        Args:
            frequency (npt.NDArray[np.float64]): Driving frequency in rad / s

        Returns:
            npt.NDArray[np.complex128]: Complex Drude dielectric function at
                frequency
        """
        return self.epsilon_infinity * (
            (
                self.longitudinal_frequency**2
                - frequency * (frequency + 1j * self.damping_frequency)
            )
            / (
                self.transverse_frequency**2
                - frequency * (frequency + 1j * self.damping_frequency)
            )
        )


@dataclass
class DispersionlessMaterial(Material):
    """An isotropic dispersionless material.

    A material characterised by a frequency independent dielectric function.

    Args:
        epsilon (float): The frequency independent dielectric function.
    """

    epsilon: float

    @property
    def epsilon(self) -> float:
        """Return the static dielectric constant."""
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Set the dielectric constant.

        Args:
            value: Value to set

        Raises:
            ValueError: If the dielectric constant is negative.
        """
        if value > 0.0:
            self.__epsilon = value
        else:
            raise ValueError("Dielectric function must be positive")

    def dielectric_function(
            self, frequency: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.complex128]:
        """Dispersionless dielectric function.

        Args:
            frequency (npt.NDArray[np.float64]): Driving frequency in rad / s

        Returns:
            npt.NDArray[np.complex128]: Complex dielectric function at frequency.
        """
        return np.ones_like(frequency, dtype=np.complex128) * self.epsilon

