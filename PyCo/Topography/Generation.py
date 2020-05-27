#
# Copyright 2018-2019 Antoine Sanner
#           2018-2019 Lars Pastewka
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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
Helper functions for the generation of random fractal surfaces
"""

import numpy as np
import scipy.stats as stats
from PyCo.Topography import Topography, UniformLineScan
from PyCo.Tools.common import compute_wavevectors, ifftn


# FIXME: Not sure topography generation should be classes. These should probably
# be turned into individual functions.

class CapillaryWavesExact(object):
    """Frozen capillary waves"""
    Error = Exception

    def __init__(self, nb_grid_pts, physical_sizes, mass_density, surface_tension,
                 bending_stiffness, seed=None):
        """
        Generates a surface with an exact power spectrum (deterministic
        amplitude)
        Keyword Arguments:
        nb_grid_pts        -- Tuple containing number of points in spatial directions.
                             The length of the tuple determines the spatial dimension
                             of the problem (for the time being, only 1D or square 2D)
        physical_sizes              -- domain physical_sizes. For multidimensional problems,
                             a tuple can be provided to specify the length per
                             dimension. If the tuple has less entries than dimensions,
                             the last value in repeated.
        mass_density      -- Mass density
        surface_tension   -- Topography tension
        bending_stiffness -- Bending stiffness
        rms_height        -- root mean square height of surface
        rms_slope         -- root mean square slope of surface
        seed              -- (default hash(None)) for repeatability, the random number
                             generator is seeded previous to outputting the generated
                             surface
        """
        if seed is not None:
            np.random.seed(hash(seed))
        if not hasattr(nb_grid_pts, "__iter__"):
            nb_grid_pts = (nb_grid_pts,)
        if not hasattr(physical_sizes, "__iter__"):
            physical_sizes = (physical_sizes,)

        self.dim = len(nb_grid_pts)
        if self.dim not in (1, 2):
            raise self.Error(
                ("Dimension of this problem is {}. Only 1 and 2-dimensional "
                 "problems are supported").format(self.dim))
        self.nb_grid_pts = nb_grid_pts
        tmpsize = list()
        for i in range(self.dim):
            tmpsize.append(physical_sizes[min(i, len(physical_sizes) - 1)])
        self.size = tuple(tmpsize)

        self.mass_density = mass_density
        self.surface_tension = surface_tension
        self.bending_stiffness = bending_stiffness

        max_pixelsize = max(
            (siz / res for siz, res in zip(self.size, self.nb_grid_pts)))

        self.q = compute_wavevectors(  # pylint: disable=invalid-name
            self.nb_grid_pts, self.size, self.dim)
        self.coeffs = self.generate_phases()
        self.generate_amplitudes()
        self.distribution = self.amplitude_distribution()
        self.active_coeffs = None

    def get_negative_frequency_iterator(self):
        " frequency complement"

        def iterator():  # pylint: disable=missing-docstring
            for i in range(self.nb_grid_pts[0]):
                for j in range(self.nb_grid_pts[1] // 2 + 1):
                    yield (i, j), (-i, -j)

        return iterator()

    def amplitude_distribution(self):  # pylint: disable=no-self-use
        """
        returns a multiplicative factor to apply to the fourier coeffs before
        computing the inverse transform (trivial in this case, since there's no
        stochastic distro in this case)
        """
        return 1.

    @property
    def abs_q(self):
        " radial distances in q-space"
        q_norm = np.sqrt((self.q[0] ** 2).reshape((-1, 1)) + self.q[1] ** 2)
        order = np.argsort(q_norm, axis=None)
        # The first entry (for |q| = 0) is rejected, since it's 0 by construct
        return q_norm.flatten()[order][1:]

    def generate_phases(self):
        """
        generates appropriate random phases (φ(-q) = -φ(q))
        """
        rand_phase = np.random.rand(*self.nb_grid_pts) * 2 * np.pi
        coeffs = np.exp(1j * rand_phase)
        for pos_it, neg_it in self.get_negative_frequency_iterator():
            if pos_it != (0, 0):
                coeffs[neg_it] = coeffs[pos_it].conj()
        if self.nb_grid_pts[0] % 2 == 0:
            r_2 = self.nb_grid_pts[0] // 2
            coeffs[r_2, 0] = coeffs[r_2, r_2] = coeffs[0, r_2] = 1
        return coeffs

    def generate_amplitudes(self):
        "compute an amplitude distribution"
        q_2 = self.q[0].reshape(-1, 1) ** 2 + self.q[1] ** 2
        q_2[0, 0] = 1  # to avoid div by zeros, needs to be fixed after
        self.coeffs *= 1 / (self.mass_density + self.surface_tension * q_2 +
                            self.bending_stiffness * q_2 * q_2)
        self.coeffs[0, 0] = 0  # et voilà
        # Fix Shannon limit:
        # self.coeffs[q_2 > self.q_max**2] = 0

    def get_topography(self):
        """
        Computes and returs a NumpySurface object with the specified
        properties. This follows appendices A and B of Persson et al. (2005)

        Persson et al., On the nature of surface roughness with application to
        contact mechanics, sealing, rubber friction and adhesion, J. Phys.:
        Condens. Matter 17 (2005) R1-R62, http://arxiv.org/abs/cond-mat/0502419

        Keyword Arguments:
        lambda_max -- (default None) specifies a cutoff value for the longest
                      wavelength. By default, this is the domain physical_sizes in the
                      smallest dimension
        lambda_min -- (default None) specifies a cutoff value for the shortest
                      wavelength. by default this is determined by Shannon's
                      Theorem.
        """
        active_coeffs = self.coeffs.copy()
        active_coeffs *= self.distribution
        area = np.prod(self.size)
        profile = ifftn(active_coeffs, area).real
        self.active_coeffs = active_coeffs
        return Topography(profile, self.size)


###

def _irfft2(karr, rarr, progress_callback=None):
    """
    Inverse 2d real-to-complex FFT with callback that allows to report status
    of the computation to user.

    Parameters
    ----------
    karr : array_like
        Fourier-space representation
    rarr : array_like
        Real-space representation
    progress_callback : function(i, n)
        Function that is called to report progress.
    """
    nrows, ncolumns = karr.shape
    for i in range(ncolumns):
        if progress_callback is not None:
            progress_callback(i, ncolumns + nrows - 1)
        karr[:, i] = np.fft.ifft(karr[:, i])
    for i in range(nrows):
        if progress_callback is not None:
            progress_callback(i + ncolumns, ncolumns + nrows - 1)
        if rarr.shape[1] % 2 == 0:
            rarr[i, :] = np.fft.irfft(karr[i, :])
        else:
            rarr[i, :] = np.fft.irfft(karr[i, :], n=rarr.shape[1])


def self_affine_prefactor(nb_grid_pts, physical_sizes, Hurst, rms_height=None,
                          rms_slope=None, short_cutoff=None, long_cutoff=None):
    r"""
    Compute prefactor :math:`C_0` for the power-spectrum density of an ideal
    self-affine topography given by

    .. math ::

        C(q) = C_0 q^{-2-2H}

    for two-dimensional topography maps and

    .. math ::

        C(q) = C_0 q^{-1-2H}

    for one-dimensional line scans. Here :math:`H` is the Hurst exponent.

    Note:
    In the 2D case:

    .. math ::

        h^2_{rms} = \frac{1}{2 \pi} \int_{0}^{\infty} q C^{iso}(q) dq

    whereas in the 1D case:

    .. math ::

        h^2_{rms} = \frac{1}{\pi} \int_{0}^{\infty} C^{1D}(q) dq

    See Equations (1) and (4) in [1].


    Parameters
    ----------
    nb_grid_pts : array_like
        Resolution of the topography map or the line scan.
    physical_sizes : array_like
        Physical physical_sizes of the topography map or the line scan.
    Hurst : float
        Hurst exponent.
    rms_height : float
        Root mean-squared height.
    rms_slope : float
        Root mean-squared slope of the topography map or the line scan.
    short_cutoff : float
        Short-wavelength cutoff.
    long_cutoff : float
        Long-wavelength cutoff.

    Returns
    -------
    prefactor : float
        Prefactor :math:`\sqrt{C_0}`

    References
    -----------
    [1]: Jacobs, Junge, Pastewka, Surf. Topgogr.: Metrol. Prop. 5, 013001 (2017)

    """
    nb_grid_pts = np.asarray(nb_grid_pts)
    physical_sizes = np.asarray(physical_sizes)

    if short_cutoff is not None:
        q_max = 2 * np.pi / short_cutoff
    else:
        q_max = np.pi * np.min(nb_grid_pts / physical_sizes)

    if long_cutoff is not None:
        q_min = 2 * np.pi / long_cutoff
    else:
        q_min = 2 * np.pi * np.max(1 / physical_sizes)

    area = np.prod(physical_sizes)

    if rms_height is not None:
        # Assuming no rolloff region
        fac = 2 * rms_height / np.sqrt(q_min ** (-2 * Hurst) -
                                       q_max ** (-2 * Hurst)) * np.sqrt(Hurst * np.pi)
    elif rms_slope is not None:
        fac = 2 * rms_slope / np.sqrt(q_max ** (2 - 2 * Hurst) -
                                      q_min ** (2 - 2 * Hurst)) * np.sqrt((1 - Hurst) * np.pi)
    else:
        raise ValueError('Neither rms height nor rms slope is defined!')

    if len(nb_grid_pts) == 1:
        fac /= np.sqrt(2)

    return fac * np.prod(nb_grid_pts) / np.sqrt(area)


def fourier_synthesis(nb_grid_pts, physical_sizes, hurst,
                      rms_height=None, rms_slope=None, c0=None,
                      short_cutoff=None, long_cutoff=None, rolloff=1.0,
                      amplitude_distribution=lambda n: np.random.normal(size=n),
                      rfn=None, kfn=None, progress_callback=None):
    r"""
    Create a self-affine, randomly rough surface using a Fourier filtering
    algorithm. The algorithm is described in:
    Ramisetti et al., J. Phys.: Condens. Matter 23, 215004 (2011);
    Jacobs, Junge, Pastewka, Surf. Topgogr.: Metrol. Prop. 5, 013001 (2017)

    Parameters
    ----------
    nb_grid_pts : array_like
        Resolution of the topography map.
    physical_sizes : array_like
        Physical physical_sizes of the topography map.
    hurst : float
        Hurst exponent.
    rms_height : float
        Root mean-squared height.
    rms_slope : float
        Root mean-squared slope.
    c0: float
        self affine prefactor :math:`C_0`:
        :math:`C(q) = C_0 q^{-2-2H}`
    short_cutoff : float
        Short-wavelength cutoff.
    long_cutoff : float
        Long-wavelength cutoff.
    rolloff : float
        Value for the power-spectral density (PSD) below the long-wavelength
        cutoff. This multiplies the value at the cutoff, i.e. unit will give a
        PSD that is flat below the cutoff, zero will give a PSD that is vanishes
        below cutoff. (Default: 1.0)
    amplitude_distribution : function
        Function that generates the distribution of amplitudes.
        (Default: np.random.normal)
    rfn : str
        Name of file that stores the real-space array. If specified, real-space
        array will be created as a memory mapped file. This is useful for
        creating very large topography maps. (Default: None)
    kfn : str
        Name of file that stores the Fourie-space array. If specified, real-space
        array will be created as a memory mapped file. This is useful for
        creating very large topography maps. (Default: None)
    progress_callback : function(i, n)
        Function that is called to report progress.

    Returns
    -------
    topography : UniformTopography or UniformLineScan
        The topography.
    """
    if short_cutoff is not None:
        q_max = 2 * np.pi / short_cutoff
    else:
        q_max = np.pi * np.min(np.asarray(nb_grid_pts) / np.asarray(physical_sizes))

    if long_cutoff is not None:
        q_min = 2 * np.pi / long_cutoff
    else:
        q_min = None

    if c0 is None:
        fac = self_affine_prefactor(nb_grid_pts, physical_sizes, hurst, rms_height=rms_height,
                                    rms_slope=rms_slope, short_cutoff=short_cutoff,
                                    long_cutoff=long_cutoff)
    else:
        # prefactor for the fourier heights
        fac = np.sqrt(c0) * np.prod(nb_grid_pts) / np.sqrt(np.prod(physical_sizes))
        #                   ^                       ^ C(q) = c0 q^(-2-2H) = 1 / A |fh(q)|^2
        #                   |                         and h(x,y) = sum(1/A fh(q) e^(iqx)))
        #                   compensate for the np.fft normalisation

    if len(nb_grid_pts) == 2:
        nx, ny = nb_grid_pts
        sx, sy = physical_sizes
        kny = ny // 2 + 1
        kshape = (nx, kny)
    else:
        nx = 1
        ny, = nb_grid_pts
        sx = 1
        sy, = physical_sizes
        kny = ny // 2 + 1
        kshape = (kny,)

    # Create in-memory or memory-mapped arrays as storage buffers
    if rfn is None:
        rarr = np.empty(nb_grid_pts, dtype=np.float64)
    else:
        rarr = np.memmap(rfn, np.float64, 'w+', shape=nb_grid_pts)
    if kfn is None:
        karr = np.empty(kshape, dtype=np.complex128)
    else:
        karr = np.memmap(kfn, np.complex128, 'w+', shape=kshape)

    qy = 2 * np.pi * np.arange(kny) / sy
    for x in range(nx):
        if progress_callback is not None:
            progress_callback(x, nx - 1)
        if x > nx // 2:
            qx = 2 * np.pi * (nx - x) / sx
        else:
            qx = 2 * np.pi * x / sx
        q_sq = qx ** 2 + qy ** 2
        if x == 0:
            q_sq[0] = 1.
        phase = np.exp(2 * np.pi * np.random.rand(kny) * 1j)
        ran = fac * phase * amplitude_distribution(kny)
        if len(nb_grid_pts) == 2:
            karr[x, :] = ran * q_sq ** (-(1 + hurst) / 2)
            karr[x, q_sq > q_max ** 2] = 0.
        else:
            karr[:] = ran * q_sq ** (-(0.5 + hurst) / 2)
            karr[q_sq > q_max ** 2] = 0.
        if q_min is not None:
            mask = q_sq < q_min ** 2
            if len(nb_grid_pts) == 2:
                karr[x, mask] = rolloff * ran[mask] * q_min ** (-(1 + hurst))
            else:
                karr[mask] = rolloff * ran[mask] * q_min ** (-(0.5 + hurst))
    if len(nb_grid_pts) == 2:
        for iy in [0, -1] if ny % 2 == 0 else [0]:
            # Enforce symmetry
            if nx % 2 == 0:
                karr[0, iy] = np.real(karr[0, iy])
                karr[nx // 2, iy] = np.real(karr[nx // 2, iy])
                karr[1:nx // 2, iy] = karr[-1:nx // 2:-1, iy].conj()
            else:
                karr[0, iy] = np.real(karr[0, iy])
                karr[1:nx // 2 + 1, iy] = karr[-1:nx // 2:-1, iy].conj()
        _irfft2(karr, rarr, progress_callback)
        return Topography(rarr, physical_sizes, periodic=True)
    else:
        karr[0] = np.real(karr[0])
        rarr[:] = np.fft.irfft(karr)
        return UniformLineScan(rarr, sy, periodic=True)
