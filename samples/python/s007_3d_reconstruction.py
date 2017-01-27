# -----------------------------------------------------------------------
# Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
#            2013-2016, CWI, Amsterdam
#
# Contact: astra@uantwerpen.be
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------

import astra
import numpy as np
import pygpu
import pylab


vol_geom = astra.create_vol_geom(128, 128, 128)

angles = np.linspace(0, 2 * np.pi, 180, False)
proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, 128, 192, angles, 1000, 0)

# Create a simple hollow cube phantom
cube = np.zeros((128, 128, 128))
cube[17:113, 17:113, 17:113] = 1
cube[33:97, 33:97, 33:97] = 0

# Create projection data from this
proj_id, proj_data = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)
ctx = pygpu.init('cuda')
pygpu.set_default_context(ctx)
# proj_gpuarr = pygpu.gpuarray.array(proj_data, dtype='float32', order='C')
# vol_gpuarr = pygpu.gpuarray.zeros(cube.shape, dtype='float32', order='C')
proj_gpuarr = pygpu.gpuarray.zeros(proj_data.shape, dtype='float32', order='C')
vol_gpuarr = pygpu.gpuarray.array(cube, dtype='float32', order='C')

# The pitch is ...
z, y, x = proj_gpuarr.shape
proj_data_link = astra.data3d.GPULink(proj_gpuarr.gpudata, x, y, z,
                                      proj_gpuarr.strides[-2])
z, y, x = vol_gpuarr.shape
vol_link = astra.data3d.GPULink(vol_gpuarr.gpudata, x, y, z,
                                vol_gpuarr.strides[-2])

proj_id = astra.data3d.link('-sino', proj_geom, proj_data_link)
rec_id = astra.data3d.link('-vol', vol_geom, vol_link)

# Display a single projection image
# pylab.gray()
# pylab.figure(1)
# pylab.imshow(proj_data[:, 20, :])

# Create a data object for the reconstruction
# rec_id = astra.data3d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('FP3D_CUDA')
cfg['VolumeDataId'] = rec_id
# cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id


# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run 150 iterations of the algorithm
# Note that this requires about 750MB of GPU memory, and has a runtime
# in the order of 10 seconds.
astra.algorithm.run(alg_id, 1)

# Get the result
# rec = astra.data3d.get(rec_id)
pylab.figure(2)
pylab.gray()
# pylab.imshow(vol_gpuarr[65, :, :])
pylab.imshow(proj_gpuarr[:, 20, :])
pylab.show()


# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)
