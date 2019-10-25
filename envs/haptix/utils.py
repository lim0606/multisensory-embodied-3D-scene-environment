import os
import math
import numpy as np
from envs.haptix.transformers import quaternion_from_matrix, rotation_matrix, quaternion_matrix

''' set model path '''
def set_model_path(model_path):
    if model_path.startswith("/"):
        fullpath = model_path
        prefixpath = None
    else:
        fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        prefixpath = os.path.join(os.path.dirname(__file__), "assets") 
    if not os.path.exists(fullpath):
        raise IOError('File {} does not exist'.format(fullpath))

    return fullpath, prefixpath

''' convert angle and axis to quat '''
def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat

'''
convert angle and axis to rotation matrix 
https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
'''
def r_from_angle_axis(angle, axis):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """

    # Trig factors.
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis[0], axis[1], axis[2]

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    # Update the rotation matrix
    matrix = np.zeros((3,3))
    matrix[0, 0] = x*xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y*yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z*zC + ca

    return matrix

'''
convert pitch, yaw to quat
https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
'''
def r_from_pitch_yaw(pitch, yaw):
    # apply global yaw -> R_z
    R_z = rotation_matrix(yaw, np.array([0., 0., 1.], dtype=np.float32))[:3, :3]

    # infer new rotation axis given R_z
    new_y_axis = R_z.dot(np.array([0., 1., 0.], dtype=np.float32))
    R_local_y = rotation_matrix(pitch, new_y_axis)[:3, :3]

    # combine rotations
    R = R_local_y.dot(R_z)

    return R

def quat_from_pitch_yaw(pitch, yaw):
    # convert pitch, yaw to R
    R = r_from_pitch_yaw(pitch, yaw)

    # infer quaternion from rotation matrix
    quat = quaternion_from_matrix(R)

    return quat

'''
convert roll, yaw to quat
https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
'''
def r_from_roll_yaw(roll, yaw):
    # apply global yaw -> R_z
    R_z = rotation_matrix(yaw, np.array([0., 0., 1.], dtype=np.float32))[:3, :3]

    # infer new rotation axis given R_z
    new_x_axis = R_z.dot(np.array([1., 0., 0.], dtype=np.float32))
    R_local_x = rotation_matrix(roll, new_x_axis)[:3, :3]

    # combine rotations
    R = R_local_x.dot(R_z)

    return R

def quat_from_roll_yaw(roll, yaw):
    # convert roll, yaw to R
    R = r_from_roll_yaw(roll, yaw)

    # infer quaternion from rotation matrix
    quat = quaternion_from_matrix(R)

    return quat
