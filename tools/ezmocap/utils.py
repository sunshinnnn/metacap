# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from typing import NewType, Union, Optional
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch
import copy
from ..rotation_tools import aa2matrot_np, matrot2aa_np
Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)


def move_Rh_onRoot(params, smpl):
    from scipy.spatial.transform import Rotation as R
    def params_moveRh2smpl(params, root_t, cc=False):
        ret = copy.deepcopy(params)
        rotmat = R.from_rotvec(ret['Rh'][:]).as_matrix()
        # rotmat = aa2matrot_np(ret['Rh'][:])
        # rotmat = to_cpu(batch_rodrigues(to_tensor(sdata['Rh'][:])))
        rotated_root_t = np.dot(rotmat, root_t)
        if ret['poses'].shape[1] == 72:
            ret['poses'][:, :3] = ret['Rh'][:]
        elif ret['poses'].shape[1] == 69:
            poses = np.concatenate([ret['Rh'][:], ret['poses'][:, :]], 1)
            ret['poses'] = poses
        else:
            print('ERROR! your poses dim 1!')
        ret['Rh'] = np.zeros((1, 3))
        ret['Th'] += rotated_root_t - root_t
        return ret

    params_for_root = {}
    params_for_root['shapes'] = params['shapes'].reshape(1,10)
    params_for_root['Th'] = np.zeros((1, 3))
    params_for_root['Rh'] = np.zeros((1, 3))
    params_for_root['poses'] = np.zeros((1, 72))

    # get 24 smpl joints, 0: root
    joints = smpl(return_verts=False, return_tensor=False, true_smpl_joints=True, return_smpl_joints=True, **params_for_root)
    params_ret = params_moveRh2smpl(params, joints[0, 0, :])
    return params_ret


def combine_object_to_poses(poses, Robjs):
    T = poses.shape[0]
    kintree_table = [4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    poses_obj = []

    for i in range(T):
        R_local_aa_temp = poses[i].copy().reshape(24, 3)
        temp_matrot_local = aa2matrot_np(R_local_aa_temp)
        temp_matrot_global = []
        for j in range(0, 24):
            if j == 0:
                temp_matrot_global.append(temp_matrot_local[j])
            else:
                temp_matrot_global.append(temp_matrot_global[kintree_table[j]].dot(temp_matrot_local[j]))
        temp_matrot_global.append(Robjs[i])
        Robj_matrot_local = temp_matrot_global[21].T.dot(temp_matrot_global[24])
        Robj_aa_local = matrot2aa_np(Robj_matrot_local)
        poses_obj.append(Robj_aa_local)
    poses_obj = np.stack(poses_obj)
    return np.concatenate([poses,poses_obj],-1)

@dataclass
class ModelOutput:
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    v_shaped: Optional[Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


@dataclass
class SMPLOutput(ModelOutput):
    betas: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None


@dataclass
class SMPLHOutput(SMPLOutput):
    left_hand_pose: Optional[Tensor] = None
    right_hand_pose: Optional[Tensor] = None
    transl: Optional[Tensor] = None


@dataclass
class SMPLXOutput(SMPLHOutput):
    expression: Optional[Tensor] = None
    jaw_pose: Optional[Tensor] = None


@dataclass
class MANOOutput(ModelOutput):
    betas: Optional[Tensor] = None
    hand_pose: Optional[Tensor] = None


@dataclass
class FLAMEOutput(ModelOutput):
    betas: Optional[Tensor] = None
    expression: Optional[Tensor] = None
    jaw_pose: Optional[Tensor] = None
    neck_pose: Optional[Tensor] = None


def find_joint_kin_chain(joint_id, kinematic_tree):
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain


def to_tensor(
        array: Union[Array, Tensor], dtype=torch.float32
) -> Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)
