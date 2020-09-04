from ccpi.framework import DataProcessor, ImageData
from ccpi.astra.utils import convert_geometry_to_astra
import astra

class AstraBackProjector3D(DataProcessor):
    '''AstraBackProjector3D
    
    Back project AcquisitionData to ImageData using ASTRA proj_geom, vol_geom.
    
    Input: AcquisitionData
    Parameter: proj_geom, vol_geom
    Output: ImageData
    '''

    ASTRA_LABELS_VOL = ['vertical','horizontal_y','horizontal_x']
    ASTRA_LABELS_PROJ = ['vertical','angle','horizontal']

    def __init__(self,
                 volume_geometry=None,
                 sinogram_geometry=None,
                 proj_geom=None,
                 vol_geom=None):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_geom'  : proj_geom,
                  'vol_geom'  : vol_geom
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
        if dataset.number_of_dimensions != 3:
            raise ValueError("Expected input dimensions 3, got {0}"\
                             .format(dataset.number_of_dimensions))

        order = [dataset.dimension_labels[0],dataset.dimension_labels[1],dataset.dimension_labels[2]]
        if order != AstraBackProjector3D.ASTRA_LABELS_PROJ:
            dataset.subset(dimensions = AstraBackProjector3D.ASTRA_LABELS_PROJ)
            print("Transposing the data for ASTRA compatibility")

        return True


    def set_ImageGeometry(self, volume_geometry):
        self.volume_geometry = volume_geometry.copy()
        self.volume_geometry.dimension_labels = AstraBackProjector3D.ASTRA_LABELS_VOL
        
    def set_AcquisitionGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry.copy()
        self.sinogram_geometry.dimension_labels = AstraBackProjector3D.ASTRA_LABELS_PROJ
        
    def process(self, out=None):

        DATA = self.get_input()
        data_temp = DATA.as_array()

        rec_id, arr_out = astra.create_backprojection(data_temp, self.proj_id)
        astra.data2d.delete(rec_id)
        
        if out is None:
            out = ImageData(arr_out, deep_copy=False, geometry=self.volume_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)