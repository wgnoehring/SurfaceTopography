#
# Copyright 2018-2021 Lars Pastewka
#           2021 Wolfram Nöhring
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
Wavelet analysis for uniform topographies.
"""

import numpy as np
import pywt

from ..HeightContainer import UniformTopographyInterface


def wavelet_analysis_from_profile(topography,  # pylint: disable=invalid-name
                                wavelet_name="db4", signal_extension_mode="periodic",
                                average = lambda x: np.mean(np.abs(x))):
    """
    Compute wavelet coefficients (averaged over location) of a topography or 
    line scan stored on a uniform grid. 

    Parameters
    ----------
    topography : SurfaceTopography or UniformLineScan
        Container with height information.
    wavelet_name: str, optional
        Wavelet to use, see `pywt.families` and `pywt.wavelist`. 
        Default: "db4" for Daubechies 4 wavelets with four vanishing
        moments.
    signal_extension_mode : str, optional
        See Ref. [1]_ and `pywt.Modes.modes` for a list of available
        modes. Default: "periodic"
    average : callable, optional
        Function for computing the location average of the wavelet
        coefficients. Default: mean absolute value, following [2]_

    Returns
    -------
    s : array_like
        Scale factors.
    C_all : array_like
        Location averages of the wavelet coefficients. 

    References
    ----------
    .. [1] https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html
    .. [2] Simonsen, I., Hansen, A. & Nes, O. M. Determination of the Hurst exponent 
       by use of wavelet transforms. Phys. Rev. E 58, 2779–2787 (1998).
    """
    n = topography.nb_grid_pts
    s = topography.physical_sizes

    try:
        nx, ny = n
        sx, sy = s
    except ValueError:
        nx, = n
        sx, = s

    h = topography.heights()

    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(data_len=nx, filter_len=wavelet.dec_len)
    # The first element of the output of `pywt.wavedec`
    # corresponds to the approximation, the other elements
    # are the detail coefficients at decreasing scales
    levels = np.r_[max_level, np.arange(max_level, 0, -1)]
    scales = 2**levels

    location_averages = np.zeros((max_level+1, h.shape[1]))
    for i in range(h.shape[1]):
        coefficients = pywt.wavedec(h[:, i], axis=0, wavelet=wavelet_name, mode=signal_extension_mode)
        location_averages[:, i] = np.array([average(c) for c in coefficients])

    if topography.dim == 1:
        return scales, location_averages
    else:
        return scales, location_averages.mean(axis=1)


# Register analysis functions from this module
UniformTopographyInterface.register_function('wavelet_analysis_from_profile',
                                             wavelet_analysis_from_profile)
