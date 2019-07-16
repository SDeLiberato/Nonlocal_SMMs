__all__ = ["RefPlot"]

import numpy as np
import matplotlib.pyplot as plt


def RefPlot(o, k, vals):
    X, Y = np.meshgrid(k, o)
    fig, axes = plt.subplots(nrows=1, ncols=len(vals), figsize=(18, 9), sharex=True, sharey=True)
    for idx, val in enumerate(vals):
        ax = axes[idx]
        im = ax.pcolor(X, Y,  val, cmap='Reds_r', vmin=0.5, vmax=1)

        ax.plot(k, k, 'r-')
        ax.set_ylim([min(o), max(o)])
        ax.set_xlabel("Rescaled in-plane-wavevector $K = k_x c/\omega_{\mathrm{L}1}$")
        ax.set_ylabel("Rescaled frequency $\Omega = \omega/\omega_{\mathrm{L}1}$")
        ax.set_title("Reflectance")
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', direction='in', length=10)
        ax.tick_params(axis='x', direction='in', length=10)
    cax = fig.add_axes([0.95, 0.25, 0.05, 0.5])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel('Reflectivity')
    plt.show()