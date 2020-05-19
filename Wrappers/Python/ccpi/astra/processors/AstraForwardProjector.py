import ccpi.cfg as cfg
if cfg.run_with_cupy:
    try:
        import cupy
    except:
        print("There is no cupy installed")        

from ccpi.framework import DataProcessor, AcquisitionData
from ccpi.astra.utils import convert_geometry_to_astra
import astra



class AstraForwardProjector(DataProcessor):
    '''AstraForwardProjector
    
    Forward project ImageData to AcquisitionData using ASTRA proj_id.
    
    Input: ImageData
    Parameter: proj_id
    Output: AcquisitionData
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
        super(AstraForwardProjector, self).__init__(**kwargs)
        
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
        if dataset.number_of_dimensions == 3 or\
           dataset.number_of_dimensions == 2:
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
        IM = self.get_input()
        DATA = AcquisitionData(geometry=self.sinogram_geometry)

        if cfg.run_with_cupy:
            sinogram_id, DATA.array = astra.create_sino(cupy.asnumpy(IM.as_array()), 
                                                           self.proj_id)
            DATA.array = cupy.array(DATA.array)
        else:
            sinogram_id, DATA.array = astra.create_sino(IM.as_array(), 
                                                           self.proj_id)
        astra.data2d.delete(sinogram_id)
        
        if self.device == 'cpu':
            ret = DATA
        else:
            if self.sinogram_geometry.geom_type == 'cone':
                ret = DATA
            else:
                 scaling = 1.0/self.volume_geometry.voxel_size_x
                 ret = scaling*DATA
        
        if out is None:
            return ret
        else:
            out.fill(ret)