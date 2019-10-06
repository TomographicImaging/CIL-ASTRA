# -*- coding: utf-8 -*-
#    This work is independent part of the Core Imaging Library developed by
#    Visual Analytics and Imaging System Group of the Science Technology
#    Facilities Council, STFC
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ccpi.optimisation.operators import Operator, LinearOperator
from ccpi.framework import AcquisitionData, ImageData, DataContainer, ImageGeometry, AcquisitionGeometry
from ccpi.astra.processors import AstraForwardProjector, AstraBackProjector, \
     AstraForwardProjector3D, AstraBackProjector3D, AstraFDK, AstraFilteredBackProjector
from ccpi.astra.operators import AstraProjectorSimple
import numpy as np

class AstraProjector3DSimple(LinearOperator):
    
    """ASTRA projector modified to use DataSet and geometry."""
    def __init__(self, geomv, geomp):
        
        super(AstraProjector3DSimple, self).__init__()
        
        # Store volume and sinogram geometries.
        self.sinogram_geometry = geomp
        self.volume_geometry = geomv
        
        self.fp = AstraForwardProjector3D(volume_geometry=geomv,
                                        sinogram_geometry=geomp,
                                        output_axes_order=['vertical','angle','horizontal'])
        
        self.bp = AstraBackProjector3D(volume_geometry=geomv,
                                        sinogram_geometry=geomp,
                                        output_axes_order=['vertical','horizontal_y','horizontal_x'])
                
        # Initialise empty for singular value.
        self.s1 = None
    
    def direct(self, IM, out=None):
        self.fp.set_input(IM)
        
        if out is None:
            return self.fp.get_output()
        else:
            out.fill(self.fp.get_output())
    
    def adjoint(self, DATA, out=None):
        self.bp.set_input(DATA)
        
        if out is None:
            return self.bp.get_output()
        else:
            out.fill(self.bp.get_output())    
    
    def domain_geometry(self):
        return self.volume_geometry
    
    def range_geometry(self):
        return self.sinogram_geometry 
        
    def FBP(self, DATA, filter_type):
        
        if self.sinogram_geometry.geom_type != 'parallel':
            raise ValueError('We need a parallel geometry')

    
        # if we have parallel geometry for 3D volume, astra has no implementation
        # then we perform FBP for each slice, hence we need 2D acquisition geometry
        # if we have cone geometry, then we perform FDK
        
        ig2Dslice = ImageGeometry(voxel_num_x = self.volume_geometry.voxel_num_x,
                                      voxel_num_y = self.volume_geometry.voxel_num_y,
                                      voxel_size_x = self.volume_geometry.voxel_size_x,
                                      voxel_size_y = self.volume_geometry.voxel_size_y)
            
        ag2Dslice = AcquisitionGeometry(geom_type = self.sinogram_geometry.geom_type,
                                            dimension = '2D',
                                            angles = self.sinogram_geometry.angles,
                                            pixel_num_h=self.sinogram_geometry.pixel_num_h,
                                            pixel_num_v=self.sinogram_geometry.pixel_num_v,
                                            pixel_size_h=self.sinogram_geometry.pixel_size_h,
                                            pixel_size_v=self.sinogram_geometry.pixel_size_v)
            
        # Since it is 3D we use device = 'gpu' for the AstraProjectorSimple
        A2D_slice = AstraProjectorSimple(ig2Dslice, ag2Dslice, 'gpu')
                        
        out3D = self.domain_geometry().allocate()
        num_slices = self.sinogram_geometry.pixel_num_v
            
        for i in range(num_slices):
            tmp_fbp2D = A2D_slice.FBP(DATA.subset(vertical=i), filter_type)
            np.copyto(out3D.array[i], tmp_fbp2D.array)
            
        return out3D
           
            
    def FDK(self, DATA, filter_type):
        
        if self.sinogram_geometry.geom_type != 'cone':
            raise ValueError('We need cone geometry')
            
        self.fdk = AstraFDK(volume_geometry = self.volume_geometry,
                            sinogram_geometry = self.sinogram_geometry,
                            filter_type = filter_type)  
        self.fdk.set_input(DATA)
        out = self.fdk.get_output()
        
        return out 
            
    def norm(self):
        x0 = self.volume_geometry.allocate('random')
        self.s1, sall, svec = LinearOperator.PowerMethodNonsquare(self, 50, x0)
        return self.s1