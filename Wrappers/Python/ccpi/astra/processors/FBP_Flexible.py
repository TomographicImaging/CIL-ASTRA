from ccpi.framework import DataProcessor, ImageGeometry, AcquisitionGeometry, ImageData
from ccpi.astra.processors.FDK_Flexible import FDK_Flexible
from ccpi.astra.utils import convert_geometry_to_astra_vec
import astra
import numpy

class FBP_Flexible(FDK_Flexible):

    '''FBP_Flexible Filtered Back Projection is a reconstructor for 2D and 3D parallel-beam geometries.
    It is able to back-project circular trajectories with 2 PI anglar range and equally spaced anglular steps.

    This uses the ram-lak filter
    This is a GPU version only
    
    Input: Volume Geometry
           Sinogram Geometry
                             
    Example:  fbp = FBP_Flexible(ig, ag)
              fbp.set_input(data)
              reconstruction = fbp.get_ouput()
                           
    Output: ImageData                                 
    '''

    def __init__(self, volume_geometry, 
                       sinogram_geometry): 
        
        #convert parallel geomerty to cone with large source to object
        sino_geom_cone = sinogram_geometry.copy()
        sino_geom_cone.config.system.update_reference_frame()

        #reverse ray direction unit-vector direction and extend to inf
        cone_source = sino_geom_cone.config.system.ray.direction * -10000000
        detector_position = sino_geom_cone.config.system.detector.position
        detector_direction_row = sino_geom_cone.config.system.detector.direction_row

        if sinogram_geometry.dimension == '2D':
            tmp = AcquisitionGeometry.create_Cone2D(cone_source, detector_position, detector_direction_row)
        else:
            detector_direction_col = sino_geom_cone.config.system.detector.direction_col
            tmp = AcquisitionGeometry.create_Cone3D(cone_source, detector_position, detector_direction_row, detector_direction_col)

        sino_geom_cone.config.system = tmp.config.system.copy()

        vol_geom_astra, proj_geom_astra = convert_geometry_to_astra_vec(volume_geometry, sino_geom_cone)
 

        super(FDK_Flexible, self).__init__( volume_geometry = volume_geometry,
                                            sinogram_geometry = sinogram_geometry,
                                            vol_geom_astra = vol_geom_astra,
                                            proj_geom_astra = proj_geom_astra)
                          
    def check_input(self, dataset):
        
        if self.sinogram_geometry.channels != 1:
            raise ValueError("Expected input data to be single channel, got {0}"\
                 .format(self.sinogram_geometry.channels))  

        if self.sinogram_geometry.geom_type != 'parallel':
            raise ValueError("Expected input data to be parallel beam geometry , got {0}"\
                 .format(self.sinogram_geometry.geom_type))  

        return True
        