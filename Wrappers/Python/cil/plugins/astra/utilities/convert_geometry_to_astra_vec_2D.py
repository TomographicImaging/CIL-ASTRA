import astra
import numpy

def convert_geometry_to_astra_vec_2D(volume_geometry, sinogram_geometry_in):

    '''Set up ASTRA Volume and projection geometry, not stored

       :param volume_geometry: ccpi.framework.ImageGeometry
       :param sinogram_geometry: ccpi.framework.AcquisitionGeometry
       
       :returns ASTRA volume and sinogram geometry'''
 
    sinogram_geometry = sinogram_geometry_in.copy()
    
    #this catches behaviour modified after CIL 21.3.1 
    try:
        sinogram_geometry.config.system.align_reference_frame('cil')
    except:
        sinogram_geometry.config.system.update_reference_frame()

    angles = sinogram_geometry.config.angles
    system = sinogram_geometry.config.system
    panel = sinogram_geometry.config.panel

    #get units
    degrees = angles.angle_unit == sinogram_geometry.DEGREE
    
    #create a 2D astra geom from 2D CIL geometry, 2D astra geometry has axis flipped compared to 3D
    volume_geometry_temp = volume_geometry.copy()

    row = numpy.zeros((2,1))
    row[0] = panel.pixel_size[0] * system.detector.direction_x[0]
    row[1] = panel.pixel_size[0] * system.detector.direction_x[1]

    if 'right' in panel.origin:
        row *= -1

    det = numpy.zeros((2,1))
    det[0] = system.detector.position[0]
    det[1] = -system.detector.position[1]

    src = numpy.zeros((2,1))
    if sinogram_geometry.geom_type == 'parallel':
        src[0] = system.ray.direction[0]
        src[1] = -system.ray.direction[1]
        projector = 'parallel_vec'
    else:
        src[0] = system.source.position[0]
        src[1] = -system.source.position[1]
        projector = 'fanflat_vec'

    vectors = numpy.zeros((angles.num_positions, 6))

    for i, theta in enumerate(angles.angle_data):
        ang = + angles.initial_angle + theta

        rotation_matrix = rotation_matrix_z_from_euler(ang, degrees=degrees)

        vectors[i, :2]  = rotation_matrix.dot(src).reshape(2)
        vectors[i, 2:4] = rotation_matrix.dot(det).reshape(2)
        vectors[i, 4:6] = rotation_matrix.dot(row).reshape(2)

    
    proj_geom = astra.creators.create_proj_geom(projector, panel.num_pixels[0], vectors)    
    vol_geom = astra.create_vol_geom(volume_geometry_temp.voxel_num_y,
                                    volume_geometry_temp.voxel_num_x,
                                    volume_geometry_temp.get_min_x(),
                                    volume_geometry_temp.get_max_x(),
                                    volume_geometry_temp.get_min_y(),
                                    volume_geometry_temp.get_max_y(),
                                    )

    return vol_geom, proj_geom

def rotation_matrix_z_from_euler(angle, degrees):
    '''Returns 2D rotation matrix for z axis using direction cosine

    :param angle: angle or rotation around z axis
    :type angle: float
    :param degrees: if radian or degrees
    :type bool: defines the unit measure of the angle
    '''
    if degrees:
        alpha = angle / 180. * numpy.pi
    else:
        alpha = angle

    rot_matrix = numpy.zeros((2,2), dtype=numpy.float64)
    rot_matrix[0][0] = numpy.cos(alpha)
    rot_matrix[0][1] = -numpy.sin(alpha)
    rot_matrix[1][0] = numpy.sin(alpha)
    rot_matrix[1][1] = numpy.cos(alpha)
    
    return rot_matrix
