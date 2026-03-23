"""
Local copy of CuPy's Delaunay implementation for compatibility with newer CuPy versions
that may have deprecated or removed the Delaunay module.
"""

from .delaunay_2d import Delaunay

__all__ = ['Delaunay']