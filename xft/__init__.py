"""
XFT - simple deep learning framework ❤️

A minimal array library built from scratch to 
understand how deep learning actually works. 
Pure Python + C++ + CUDA. No magic.
"""

from xft._core import (
    __version__,
    DType,
    Order,
    to_numpy,
    dtype_name,
    dtype_size,
)

from xft.array import (
    Array,
    array,
    from_numpy,
)

__author__ = "Suresh Neethimohan"
__license__ = "MIT"
__description__ = "XFT - simple deep learning framework ❤️"

__all__ = [
    '__version__',
    '__author__',
    '__license__',
    '__description__',
    'DType',
    'Order',
    'Array',
    'array',
    'from_numpy',
    'to_numpy',
    'dtype_name',
    'dtype_size',
]
