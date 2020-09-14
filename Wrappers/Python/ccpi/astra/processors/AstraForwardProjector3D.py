from ccpi.framework import DataProcessor, AcquisitionData
from ccpi.astra.utils import convert_geometry_to_astra
import astra

class AstraForwardProjector3D(DataProcessor):
    '''AstraForwardProjector3D
    
    Forward project ImageData to AcquisitionData using ASTRA proj_geom and 
    vol_geom.
    
    Input: ImageData
    Parameter: proj_geom, vol_geom
    Output: AcquisitionData
    '''
    
    def __init__(self,
                 volume_geometry=None,
                 sinogram_geometry=None,
                 proj_geom=None,
                 vol_geom=None,
                 output_axes_order=None):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_geom'  : proj_geom,
                  'vol_geom'  : vol_geom,
                  'output_axes_order'  : output_axes_order
                  }
        
        super(AstraForwardProjector3D, self).__init__(**kwargs)
        
        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)
        
        # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
        
        # Also store ASTRA geometries
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
    
    def check_input(self, dataset):
        if dataset.number_of_dimensions == 3:
               return True
        else:
            raise ValueError("Expected input dimensions 3, got {0}"\
                             .format(dataset.number_of_dimensions))
    
    def set_ImageGeometry(self, volume_geometry):
        self.volume_geometry = volume_geometry
    
    def set_AcquisitionGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry
    
    def set_vol_geom(self, vol_geom):
        self.vol_geom = vol_geom
    
    def process(self, out=None):
                     
        IM = self.get_input()
        DATA = AcquisitionData(geometry=self.sinogram_geometry,
                               dimension_labels=self.output_axes_order)
        
        if cfg.run_with_cupy:
            sinogram_id, DATA.array = astra.create_sino3d_gpu(cupy.asnumpy(IM.as_array()), 
                                                           self.proj_geom,
                                                           self.vol_geom) 
            DATA.array = cupy.array(DATA.array)
            astra.data3d.delete(sinogram_id)
            
        else:    
            sinogram_id, DATA.array = astra.create_sino3d_gpu(IM.as_array(), 
                                                           self.proj_geom,
                                                           self.vol_geom)
            astra.data3d.delete(sinogram_id)
        # 3D CUDA FP does not need scaling
        
        if out is None:
            return DATA
        else:
            out.fill(DATA)