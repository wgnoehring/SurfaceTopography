#
# Copyright 2021 Lars Pastewka
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Scale-dependent slope
"""

import numpy as np

from ..HeightContainer import UniformTopographyInterface
from ..HeightContainer import NonuniformLineScanInterface


def scale_dependent_slope_1D(topography, **kwargs):
    r"""
    Compute the one-dimensional scale-dependent slope.

    The scale-dependent slope is given by

       .. math::
         :nowrap:

         \begin{equation}
         h_\text{rms}^\prime(\lambda) = \left[2A(\lambda)\right]^{1/2}/\lambda.
         \end{equation}


    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Container storing the uniform topography map
    direction : int
        Cartesian direction in which to compute the scale-dependent slope

    Returns
    -------
    r : array
        Distances. (Units: length)
    slope : array
        Slope. (Units: dimensionless)
    """  # noqa: E501
    r, A = topography.autocorrelation_1D(**kwargs)
    return r[1:], np.sqrt(2 * A[1:]) / r[1:]


def scale_dependent_slope_2D(topography, **kwargs):
    r"""
    Compute the two-dimensional, radially averaged scale-dependent slope.

    The scale-dependent slope is given by

       .. math::
         :nowrap:

         \begin{equation}
         h_\text{rms}^\prime(\lambda) = \left[2A(\lambda)\right]^{1/2}/\lambda.
         \end{equation}


    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Container storing the uniform topography map
    direction : int
        Cartesian direction in which to compute the scale-dependent slope

    Returns
    -------
    r : array
        Distances. (Units: length)
    slope : array
        Slope. (Units: dimensionless)
    """  # noqa: E501
    r, A = topography.autocorrelation_2D(**kwargs)
    return r[1:], np.sqrt(2 * A[1:]) / r[1:]


# Register analysis functions from this module
UniformTopographyInterface.register_function('scale_dependent_slope_1D',
                                             scale_dependent_slope_1D)
NonuniformLineScanInterface.register_function('scale_dependent_slope_1D',
                                              scale_dependent_slope_1D)
UniformTopographyInterface.register_function('scale_dependent_slope_2D',
                                             scale_dependent_slope_2D)
NonuniformLineScanInterface.register_function('scale_dependent_slope_2D',
                                              scale_dependent_slope_2D)
