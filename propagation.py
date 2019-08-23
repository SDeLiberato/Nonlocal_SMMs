__all__ = ['prop_mat']

import numpy as np
from itertools import product
from materials.materials import Media, properties

def prop_mat(wn, material, thickness, ang=None, kx=None):