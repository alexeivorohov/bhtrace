from typing import Dict
from .base import CoordinateTransformation
from .implementations import (
    Ident,
    Cartesian2Spherical,
    Spherical2Cartesian,
    Cartesian2Axial,
    Axial2Cartesian,
    Shift,
    Rotation
)

relation_dict: Dict[str, Dict[str, CoordinateTransformation]] = {

    'Cartesian': {
        'Cartesian': Shift,
        'Spherical': Cartesian2Spherical, 
        'Axial': Cartesian2Axial,
        'Sym': None
                  }, 

    'Spherical': {
        'Cartesian': Spherical2Cartesian,
        'Spherical': None,
        'Axial': None,
        'Sym': None
                  },

    'Axial': {
        'Cartesian': Axial2Cartesian,
        'Spherical': None, 
        'Axial': None,
        'Sym' : None
        },

    'Sym': None
}
