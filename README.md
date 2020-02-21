# Nonlocal Scattering Matrix Calculator
This module implements the nonlocal scattering matrix formalism available on the [arXiv](https://arxiv.org) or behind a paywall. It allows you to calculate the reflectance of anisotropic planar semiconductor heterostructures. Documentation is available [here](.....). If you use the code please cite the paper!

# Installation
This repository is packaged using [Poetry](https://python-poetry.org/). Having installed Poetry you can install the package and it's dependancies as:

``` Bash
$ git clone https://github.com/cgubbin/nonlocal_smms.git
$ cd nonlocal_smms
$ poetry install
```

Examples are demonstrated in Jupyter notebooks in `/examples`, to launch the notebook server run:

``` Bash
$ cd examples
$ poetry run jupyter notebook
```


# Basic Usage
To create a simple heterostructure such as the nitride stack studied in the accompanying manuscript write

``` python
from nonlocal_smms.core import scattering_matrix
angle = 75  # Incident angle in degrees
wavenumbers = [x for x in range(750, 1050)]  # Sets a wavenumber range to probe
thickness = 1e-9  # Set the layer thickness
repetitions = 50  # Set the sublattice repeats
sublattice = [['AlN', thickness], ['GaN', thickness]]  # Define the superlattice cell
superlattice = sublattice * repetitions
superlattice.insert(0, ['vacuum', 0])
superlattice.insert(len(superlattice), ['SiC4H', 0])
rte, rtm = scattering_matrix(wavenumbers, superlattice, angle=angle)
```
