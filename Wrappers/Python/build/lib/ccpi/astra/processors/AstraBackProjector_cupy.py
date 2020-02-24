from ccpi.framework import DataProcessor_cupy
from ccpi.framework import ImageData_cupy as ImageData
from ccpi.astra.utils import convert_geometry_to_astra
import astra
import cupy as cp



class AstraBackProjector_cupy(DataProcessor_cupy):
    '''AstraBackProjector
    
    Back project AcquisitionData to ImageData using ASTRA proj_id.
    
    Input: AcquisitionData
    Parameter: proj_id
    Output: ImageData
    '''
    
    def __init__(self,
                 volume_geometry=None,
                 sinogram_geometry=None,
                 proj_id=None,
                 device='cpu'):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_id'  : proj_id,
                  'device'  : device
                  }
        
        #DataProcessor.__init__(self, **kwargs)
        super(AstraBackProjector_cupy, self).__init__(**kwargs)
        
        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)
        
                # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
        
        # ASTRA projector, to be stored
        if device == 'cpu':
            # Note that 'line' only one option
            if self.sinogram_geometry.geom_type == 'parallel':
                self.set_projector(astra.create_projector('line', proj_geom, vol_geom) )
            elif self.sinogram_geometry.geom_type == 'cone':
                self.set_projector(astra.create_projector('line_fanflat', proj_geom, vol_geom) )
            else:
                NotImplemented 
        elif device == 'gpu':
            self.set_projector(astra.create_projector('cuda', proj_geom, vol_geom) )
        else:
            NotImplemented
    
    def check_input(self, dataset):
        if dataset.number_of_dimensions == 3 or dataset.number_of_dimensions == 2:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))
    
    def set_projector(self, proj_id):
        self.proj_id = proj_id
        
    def set_ImageGeometry(self, volume_geometry):
        self.volume_geometry = volume_geometry
        
    def set_AcquisitionGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry
    
    def process(self, out=None):
        DATA = self.get_input()
        IM = ImageData(geometry=self.volume_geometry)
        rec_id, IM.array = astra.create_backprojection(cp.asnumpy(DATA.as_array()),
                            self.proj_id)
        IM.array = cp.array(IM.array)
        astra.data2d.delete(rec_id)
        
        if self.device == 'cpu':
            ret = IM
        else:
            scaling = self.volume_geometry.voxel_size_x**3  
            ret = scaling*IM
        
        if out is None:
            return ret
        else:
            out.fill(ret)