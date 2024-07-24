"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-01-08
"""

import sys
sys.path.append('..')
import numpy as np
from torch.nn import functional as F
# from tools import tgm_conversion as tgm
# import tools.tgm_conversion as tgm
import torch
from .torch3d_transforms import *

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def inputpose2smplpose(pose):
    bs = pose.shape[0]
    pose = pose.contiguous().view(bs, -1)
    pose = torch.cat((pose, torch.zeros(bs, 6).to(pose.device)), dim=1)
    return pose

def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    assert (torch.stack([torch.det(mat) for mat in matrix]) < 0).sum() == 0
    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)

    return out

def euler(rots, order='xyz', degrees=True):
    from scipy.spatial.transform import Rotation as R
    return R.from_euler(order, rots, degrees=degrees).as_matrix()


def aa2matrot_np(pose):
    '''
    :param Nx3
    :return: pose_matrot: Nx3x3
    '''
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(pose).as_matrix()


def matrot2aa_np(pose):
    '''
    :param Nx3
    :return: pose_matrot: Nx3x3
    '''
    from scipy.spatial.transform import Rotation as R
    return R.from_matrix(pose).as_rotvec()

def matrot2quat_np(pose):
    '''
    :param Nx3
    :return: pose_matrot: Nx3x3
    '''
    from scipy.spatial.transform import Rotation as R
    return R.from_matrix(pose).as_quat()

def quat2matrot_np(pose):
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(pose).as_matrix()


def matrot2euler_np(pose, order='xyz', degrees=True):
    '''
    :param Nx3x3
    :return: pose_matrot: Nx3
    '''
    from scipy.spatial.transform import Rotation as R
    return R.from_matrix(pose).as_euler(order, degrees=degrees)


def euler2aa_np(pose, order='xyz'):
    '''
    :param Nx3x3
    :return: pose_matrot: Nx3
    '''
    from scipy.spatial.transform import Rotation as R
    return R.from_euler(order, pose, degrees=True).as_rotvec()


def rot6d2matrot_np(np_r6d):
    '''
    :param Nx6
    :return: pose_matrot: Nx3x3
    '''
    shape = np_r6d.shape
    np_r6d = np.reshape(np_r6d, [-1, 6])
    x_raw = np_r6d[:, 0:3]
    y_raw = np_r6d[:, 3:6]
    x = x_raw / np.linalg.norm(x_raw, ord=2, axis=-1).reshape(-1, 1)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, ord=2, axis=-1).reshape(-1, 1)
    y = np.cross(z, x)
    x = np.reshape(x, [-1, 3, 1])
    y = np.reshape(y, [-1, 3, 1])
    z = np.reshape(z, [-1, 3, 1])
    np_matrix = np.concatenate([x, y, z], axis=-1)
    return np_matrix


def matrot2rot6d_np(pose):
    '''
    :param Nx3x3
    :return: pose_matrot: Nx6
    '''
    res = pose[:, :, :2].reshape(-1, 6, order='F')
    return res


def rot6d2aa_np(np_r6d):
    '''
    :param Nx6
    :return: pose_matrot: Nx3
    '''
    res = matrot2aa_np(rot6d2matrot_np(np_r6d))
    return res


def aa2rot6d_np(aa):
    '''
    :param Nx3 or N x3k
    :return: pose_matrot: Nx6
    '''
    N = aa.shape[0]
    aa = aa.reshape(-1, 3)
    res = matrot2rot6d_np(aa2matrot_np(aa)).reshape(N, -1)
    return res

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrot2aa(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))




def local2global_pose(local_pose, kintree):
    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose

def em2euler(em):
    '''

    :param em: rotation in expo-map (3,)
    :return: rotation in euler angles (3,)
    '''
    from transforms3d.euler import axangle2euler

    theta = np.sqrt((em ** 2).sum())
    axis = em / theta
    return np.array(axangle2euler(axis, theta))


def euler2em(ea):
    '''

    :param ea: rotation in euler angles (3,)
    :return: rotation in expo-map (3,)
    '''
    from transforms3d.euler import euler2axangle
    axis, theta = euler2axangle(*ea)
    return np.array(axis*theta)


def remove_zrot(pose):
    noZ = em2euler(pose[:3].copy())
    noZ[2] = 0
    pose[:3] = euler2em(noZ).copy()
    return pose

def matrot2aa(pose_matrot):
    '''
    :param pose_matrot: Nx3x3
    :return: Nx3
    '''
    bs = pose_matrot.size(0)
    homogen_matrot = F.pad(pose_matrot, [0,1])
    # pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot)
    pose = rotation_matrix_to_angle_axis(homogen_matrot)
    return pose

def aa2matrot(pose):
    '''
    :param Nx3
    :return: pose_matrot: Nx3x3
    '''
    bs = pose.size(0)
    num_joints = pose.size(1)//3
    # pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose)[:, :3, :3].contiguous()#.view(bs, num_joints*9)
    pose_body_matrot = angle_axis_to_rotation_matrix(pose)[:, :3, :3].contiguous()  # .view(bs, num_joints*9)
    return pose_body_matrot

def matrot2rot6d(pose):
    '''
    :param Nx3x3
    :return: pose_matrot: Nx6
    '''
    bs = pose.size(0)
    num_joints = pose.size(1)//3
    p0 = pose[:, :, 0].view(bs,3)
    p1 = pose[:, :, 1].view(bs, 3)

    return torch.cat((p0,p1),dim = 1 )

# def matrot2rot6d_np(pose):
#     '''
#     :param Nx3x3
#     :return: pose_matrot: Nx6
#     '''
#     res = pose[:, :, :2].reshape(-1, 6, order='F')
#     return res

def noisy_zrot(rot_in):
    '''

    :param rot_in: np.array Nx3 rotations in axis-angle representation
    :return:
        will add a degree from a full circle to the zrotations
    '''
    is_batched = False
    if rot_in.ndim == 2: is_batched = True
    if not is_batched:
        rot_in = rot_in[np.newaxis]

    rnd_zrot = np.random.uniform(-np.pi, np.pi)
    rot_out = []
    for bId in range(len(rot_in)):
        pose_cpu = rot_in[bId]
        pose_euler = em2euler(pose_cpu)

        pose_euler[2] += rnd_zrot

        pose_aa = euler2em(pose_euler)
        rot_out.append(pose_aa.copy())

    return np.array(rot_out)

def rotate_points_xyz(mesh_v, Rxyz):
    '''

    :param mesh_v: Nxnum_vx3
    :param Rxyz: Nx3
    :return:
    '''

    mesh_v_rotated = []

    for fId in range(mesh_v.shape[0]):
        angle = np.radians(Rxyz[fId, 0])
        rx = np.array([
            [1., 0., 0.           ],
            [0., np.cos(angle), -np.sin(angle)],
            [0., np.sin(angle), np.cos(angle) ]
        ])

        angle = np.radians(Rxyz[fId, 1])
        ry = np.array([
            [np.cos(angle), 0., np.sin(angle)],
            [0., 1., 0.           ],
            [-np.sin(angle), 0., np.cos(angle)]
        ])

        angle = np.radians(Rxyz[fId, 2])
        rz = np.array([
            [np.cos(angle), -np.sin(angle), 0. ],
            [np.sin(angle), np.cos(angle), 0. ],
            [0., 0., 1. ]
        ])
        mesh_v_rotated.append(rz.dot(ry.dot(rx.dot(mesh_v[fId].T))).T)

    return np.array(mesh_v_rotated)


# @staticmethod
def fromScipy(quatScipy):
    return quatScipy[:, [3, 0, 1, 2]]


# @staticmethod
def toScipy(quatOur):
    return quatOur[:, [1, 2, 3, 0]] * -1


def fromTransformation2Vector(Transformation):
    rotation = fromScipy(matrot2quat_np(Transformation[:3, :3]).reshape(1, -1))
    t = Transformation[:3, 3].reshape(1, -1)
    translation = np.zeros((1, 4))
    translation[0, 0] = -0.5 * (t[0, 0] * rotation[0, 1] + t[0, 1] * rotation[0, 2] + t[0, 2] * rotation[0, 3])
    translation[0, 1] = 0.5 * (t[0, 0] * rotation[0, 0] + t[0, 1] * rotation[0, 3] - t[0, 2] * rotation[0, 2])
    translation[0, 2] = 0.5 * (-t[0, 0] * rotation[0, 3] + t[0, 1] * rotation[0, 0] + t[0, 2] * rotation[0, 1])
    translation[0, 3] = 0.5 * (t[0, 0] * rotation[0, 2] - t[0, 1] * rotation[0, 1] + t[0, 2] * rotation[0, 0])
    return np.concatenate([rotation, translation], 1)

def fromTransformation2VectorTorch(Transformation):
    """
        Inputs:
            Transformation: B,4,4 or T,B,4,4
        Outputs:
            Vector: B,8 or T, B, 8
    """
    if Transformation.ndim == 3:
        B = Transformation.shape[0]
        rotation = matrix_to_quaternion(Transformation[:, :3, :3]).reshape(B, -1)
        t = Transformation[:, :3, 3].reshape(B, -1)
        # flags = torch.where(rotation[:,:1]<0, 1, -1)
        translation = torch.zeros((B, 4)).to(rotation.device)
        translation[:, 0] = -0.5 * (t[:, 0] * rotation[:, 1] + t[:, 1] * rotation[:, 2] + t[:, 2] * rotation[:, 3])
        translation[:, 1] = 0.5 * (t[:, 0] * rotation[:, 0] + t[:, 1] * rotation[:, 3] - t[:, 2] * rotation[:, 2])
        translation[:, 2] = 0.5 * (-t[:, 0] * rotation[:, 3] + t[:, 1] * rotation[:, 0] + t[:, 2] * rotation[:, 1])
        translation[:, 3] = 0.5 * (t[:, 0] * rotation[:, 2] - t[:, 1] * rotation[0:, 1] + t[:, 2] * rotation[:, 0])
        return torch.cat([rotation, translation], 1)
    elif Transformation.ndim == 4:
        T, B = Transformation.shape[:2]
        rotation = matrix_to_quaternion(Transformation[:, :, :3, :3]).reshape(T * B, -1)
        t = Transformation[:, :, :3, 3].reshape(B*T, -1)
        translation = torch.zeros((B*T, 4)).to(rotation.device)
        translation[:, 0] = -0.5 * (t[:, 0] * rotation[:, 1] + t[:, 1] * rotation[:, 2] + t[:, 2] * rotation[:, 3])
        translation[:, 1] = 0.5 * (t[:, 0] * rotation[:, 0] + t[:, 1] * rotation[:, 3] - t[:, 2] * rotation[:, 2])
        translation[:, 2] = 0.5 * (-t[:, 0] * rotation[:, 3] + t[:, 1] * rotation[:, 0] + t[:, 2] * rotation[:, 1])
        translation[:, 3] = 0.5 * (t[:, 0] * rotation[:, 2] - t[:, 1] * rotation[0:, 1] + t[:, 2] * rotation[:, 0])
        return torch.cat([rotation, translation], 1).reshape(T, B, 8)
    else:
        raise NotImplementedError

def normalizeDQ(Vector):
    Vector = Vector.reshape(1, -1)
    rotation = Vector[:, :4]
    translation = Vector[:, 4:]
    scale = 1 / np.linalg.norm(toScipy(rotation))
    return Vector * scale

def normalizeDQTorch(Vector):
    """
        Inputs:
            Vector: B,8 or T, B, 8
        Outputs:
            Vector: B,8 or T, B, 8
    """
    # if len(Vector.shape)==1:
    #     Vector = Vector.reshape(1,8)
    if Vector.ndim == 2:
        B = Vector.shape[0]
        # Vector = Vector.reshape(1, -1)
        rotation = Vector[:, :4]
        translation = Vector[:, 4:]
        # scale = 1 / np.linalg.norm(toScipy(rotation))
        scale = 1 / torch.linalg.norm(rotation,dim=1).reshape(-1,1)
        return Vector * scale
    elif Vector.ndim == 3:
        rotation = Vector[:, :, :4]
        scale = 1 / torch.linalg.norm(rotation, dim=2, keepdims = True)
        return Vector * scale
    else:
        raise NotImplementedError


def imodDQ(Vector1, Vector2):
    r1 = Vector1[:, :4]
    r2 = Vector2[:, :4]
    t1 = Vector1[:, 4:]
    t2 = Vector2[:, 4:]
    RQ = quatMultiply(r1, r2)
    tQ1 = quatMultiply(r1, t2)
    tQ2 = quatMultiply(t1, r2)
    tQ = tQ1 + tQ2
    return np.concatenate([RQ, tQ], 1)

def imodDQTorch(Vector1, Vector2):
    r1 = Vector1[:, :4]
    r2 = Vector2[:, :4]
    t1 = Vector1[:, 4:]
    t2 = Vector2[:, 4:]
    B = Vector1.shape[0]
    temp = quatMultiplyTorch( torch.cat([r1,r1,t1],dim=0), torch.cat([r2,t2,r2],dim=0))
    return torch.cat([temp[:B, :], temp[B:2*B, :] + temp[2*B:3*B, :]], 1)

def quatMultiply(q1, q2):
    w1, x1, y1, z1 = q1[0, 0], q1[0, 1], q1[0, 2], q1[0, 3]
    w2, x2, y2, z2 = q2[0, 0], q2[0, 1], q2[0, 2], q2[0, 3]
    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([[w3, x3, y3, z3]])

def quatMultiplyTorch(q1, q2):
    temp = torch.einsum('bm,bn->bmn', q1, q2)
    w3 = temp[:,0,0] - temp[:,1,1] - temp[:,2,2]- temp[:,3,3]
    x3 = temp[:,0,1] + temp[:,1,0] + temp[:,2,3] - temp[:,3,2]
    y3 = temp[:,0,2] - temp[:,1,3] + temp[:,2,0] + temp[:,3,1]
    z3 = temp[:,0,3] + temp[:,1,2] - temp[:,2,1] + temp[:,3,0]
    return torch.stack([w3, x3, y3, z3], dim=-1)

def fromVector2Transformation(Vector):
    Vector = Vector.reshape(1, -1)
    rotation = Vector[:, :4]
    translation = Vector[:, 4:]
    scale = 1 / np.linalg.norm(toScipy(rotation))
    rotation = rotation * scale
    translation = translation * scale

    R = quat2matrot_np(toScipy(rotation)[0])
    rw, rx, ry, rz = rotation[0, 0], rotation[0, 1], rotation[0, 2], rotation[0, 3]
    tw, tx, ty, tz = translation[0, 0], translation[0, 1], translation[0, 2], translation[0, 3]
    t = np.zeros((1, 3))
    t[0, 0] = 2 * (-tw * rx + tx * rw - ty * rz + tz * ry)
    t[0, 1] = 2 * (-tw * ry + tx * rz + ty * rw - tz * rx)
    t[0, 2] = 2 * (-tw * rz - tx * ry + ty * rx + tz * rw)
    Transformation = np.eye(4)
    Transformation[:3, :3] = R
    Transformation[:3, 3:] = t.reshape(-1, 1)

    return Transformation


def fromVector2TransformationTorch(Vector):
    """
        Inputs:
            Vector: B,8 or T, B, 8
        Outputs:
            Transformation: B,4,4 or T,B,4,4
    """
    if Vector.ndim == 2:
        B = Vector.shape[0]
        VectorNormalized = normalizeDQTorch(Vector)
        rotation = VectorNormalized[:, :4]
        translation = VectorNormalized[:, 4:]

        # R = quat2matrot_np(toScipy(rotation)[0])
        R = quaternion_to_matrix(rotation)
        rw, rx, ry, rz = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]
        tw, tx, ty, tz = translation[:, 0], translation[:, 1], translation[:, 2], translation[:, 3]
        t = torch.zeros((B, 3), dtype=torch.float32)
        t[:, 0] = 2 * (-tw * rx + tx * rw - ty * rz + tz * ry)
        t[:, 1] = 2 * (-tw * ry + tx * rz + ty * rw - tz * rx)
        t[:, 2] = 2 * (-tw * rz - tx * ry + ty * rx + tz * rw)
        Transformation = torch.eye(4).repeat(B,1,1)
        Transformation[:, :3, :3] = R
        Transformation[:, :3, 3] = t
        return Transformation.to(Vector.device)
    elif Vector.ndim == 3:
        T, B = Vector.shape[:2]
        VectorNormalized = normalizeDQTorch(Vector)
        rotation = VectorNormalized[:, :, :4].reshape(-1,4)
        translation = VectorNormalized[:, :, 4:].reshape(-1,4)

        # R = quat2matrot_np(toScipy(rotation)[0])
        R = quaternion_to_matrix(rotation)
        rw, rx, ry, rz = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]
        tw, tx, ty, tz = translation[:, 0], translation[:, 1], translation[:, 2], translation[:, 3]
        t = torch.zeros((B * T, 3), dtype=torch.float32)
        t[:, 0] = 2 * (-tw * rx + tx * rw - ty * rz + tz * ry)
        t[:, 1] = 2 * (-tw * ry + tx * rz + ty * rw - tz * rx)
        t[:, 2] = 2 * (-tw * rz - tx * ry + ty * rx + tz * rw)
        Transformation = torch.eye(4).repeat(B*T, 1, 1)
        Transformation[:, :3, :3] = R
        Transformation[:, :3, 3] = t
        return Transformation.reshape(T,B,4,4).to(Vector.device)

if __name__ == '__main__':

    print()