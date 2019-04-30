from ccpi.framework import DataProcessor, ImageData, AcquisitionData
from ccpi.astra.utils import convert_geometry_to_astra
import astra

class AstraBackProjector3D(DataProcessor):
    '''AstraBackProjector3D
    
    Back project AcquisitionData to ImageData using ASTRA proj_geom, vol_geom.
    
    Input: AcquisitionData
    Parameter: proj_geom, vol_geom
    Output: ImageData
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
        
        #DataProcessor.__init__(self, **kwargs)
        super(AstraBackProjector3D, self).__init__(**kwargs)
        
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
            raise ValueError("Expected input dimensions is 3, got {0}"\
                             .format(dataset.number_of_dimensions))
        
    def set_ImageGeometry(self, volume_geometry):
        self.volume_geometry = volume_geometry
        
    def set_AcquisitionGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry
    
    def process(self, out=None):
        DATA = self.get_input()
        IM = ImageData(geometry=self.volume_geometry,
                       dimension_labels=self.output_axes_order)
        rec_id, IM.array = astra.create_backprojection3d_gpu(DATA.as_array(),
                            self.proj_geom,
                            self.vol_geom)
        astra.data3d.delete(rec_id)
        
        # Scaling of 3D ASTRA backprojector, works both parallel and cone.
        scaling = 1/self.volume_geometry.voxel_size_x**2  
        ret = scaling*IM
        
        if out is None:
            return ret
        else:
            out.fill(ret)
