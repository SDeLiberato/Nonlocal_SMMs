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

# Construct the heterostructure
sublattice = [['AlN', thickness], ['GaN', thickness]]  # Define the superlattice cell
superlattice = sublattice * repetitions
superlattice.insert(0, ['vacuum', 0])
superlattice.insert(len(superlattice), ['SiC4H', 0])

# Calculate the reflectance
rte, rtm = scattering_matrix(wavenumbers, superlattice, angle=angle)
```

# Adding a Material to the Database

To add a material to the database follow this template:

```python
from tinydb import TinyDB, Query

# Import the material database, you will need to change the import path depending
# on where you run the script
material_db = TinyDB(".../materials.json")

# Create a new material, in this case 3C-SiC. Note that frequencies are provided in wavenumbers
# while velocities (beta_) are provided in unitless form. They are the physical phonon velocity
# divided by the speed of light in vacuum.
new_material = {
  "beta_l": 3e-05, "beta_t": 3e-05, "beta_c": 5e-05, "rho": 3.21, "eps_inf_pa": 6.52,
  "eps_inf_pe": 6.52, "eps_0_pa": 9.7, "eps_0_pe": 9.7, "wto_pa": 797.4, "wto_pe": 797.5,
  "gamma": 4, "wlo_pa": 972.69, "wlo_pe": 972.7, "material": "SiC3C"
}

# Insert the new material into the database, automatically writing to file
material_db.insert(new_material)
```
