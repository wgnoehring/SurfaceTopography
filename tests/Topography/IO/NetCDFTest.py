#
# Copyright 2019 Lars Pastewka
#           2019 Antoine Sanner
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

import os

import numpy as np

from NuMPI import MPI

from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.Topography.IO import read_topography
from PyCo.Topography.IO.NC import NCReader
from PyCo.Topography.Generation import fourier_synthesis

def test_save_and_load(comm):
    nb_grid_pts = (128, 128)
    size = (3, 3)

    np.random.seed(1)
    t = fourier_synthesis(nb_grid_pts, size, 0.8, rms_slope=0.1)
    t.info['unit'] = 'μm'

    substrate = PeriodicFFTElasticHalfSpace(nb_grid_pts, 1,
                                            fft='serial' if comm is None or comm.size == 1 else 'mpi',
                                            communicator=comm)
    dt = t.domain_decompose(substrate.subdomain_locations, substrate.nb_subdomain_grid_pts,
                            communicator=comm)
    assert t.info['unit'] == 'μm'
    if comm.size > 1:
        assert dt.is_domain_decomposed

    # Save file
    dt.to_netcdf('parallel_save_test.nc')

    # Attempt to open full file on each process
    t2 = read_topography('parallel_save_test.nc')

    assert t.physical_sizes == t2.physical_sizes
    assert t.info['unit'] == t2.info['unit']
    np.testing.assert_array_almost_equal(t.heights(), t2.heights())

    # Attempt to open file in parallel
    r = NCReader('parallel_save_test.nc', communicator=comm)

    assert r.channels[0].nb_grid_pts == nb_grid_pts

    t3 = r.topography(subdomain_locations=substrate.subdomain_locations,
                      nb_subdomain_grid_pts=substrate.nb_subdomain_grid_pts)

    assert t.physical_sizes == t3.physical_sizes
    assert t.info['unit'] == t3.info['unit']
    np.testing.assert_array_almost_equal(dt.heights(), t3.heights())

    assert t3.is_periodic

    os.remove('parallel_save_test.nc')

def test_save_and_load_no_unit():
    nb_grid_pts = (128, 128)
    size = (3, 3)

    np.random.seed(1)
    t = fourier_synthesis(nb_grid_pts, size, 0.8, rms_slope=0.1)

    # Save file
    t.to_netcdf('no_unit.nc')

    # Attempt to open full file on each process
    t2 = read_topography('no_unit.nc')

    assert t.physical_sizes == t2.physical_sizes
    assert 'unit' not in t2.info
    np.testing.assert_array_almost_equal(t.heights(), t2.heights())

    os.remove('no_unit.nc')

if __name__ == '__main__':
    test_save_and_load(MPI.COMM_WORLD)
