"""Helper functions for pretty plotting"""


__all__ = ["RefPlot"]

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# flake8: noqa: W605
def RefPlot(
    wavenumber: np.ndarray,
    wavevector: np.ndarray,
    values: np.ndarray,
    fig: Any,
    ax: Any,
    cax: bool = False,
) -> None:
    """A helper function to generate a reflection map.

    Takes input wavenumbers, wavevectors and values, typically reflectance or transmittance
    and generates a color map.

    Args:
        wavenumber (np.ndarray): The probe frequencies in inverse centimetres
        wavevector (np.ndarray): The probe in-plane wavevectors in inverse centimetres
        values (np.ndarray): The values to plot, must have dimension equal to the product
            of the lengths of the wavenumber and wavevector arrays

    Returns:
        None

    """
    X, Y = np.meshgrid(wavevector, wavenumber)
    im = ax.pcolor(X, Y, values, cmap="Reds_r", vmin=0.5, vmax=1)
    ax.plot(wavevector, wavevector, "r-")
    ax.set_ylim([min(wavenumber), max(wavenumber)])
    ax.set_xlabel("Rescaled in-plane-wavevector $K = k_x c/\omega_{\mathrm{L}1}$")
    ax.set_ylabel("Rescaled frequency $\Omega = \omega/\omega_{\mathrm{L}1}$")
    ax.set_title("Reflectance")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", direction="in", length=10)
    ax.tick_params(axis="x", direction="in", length=10)
    if cax is True:
        cax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.ax.set_ylabel("Reflectivity")
