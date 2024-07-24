'''
  @ Date: 2020-11-18 14:33:20
  @ Author: Qing Shuai
  @ LastEditors: Guoxing Sun
  @ LastEditTime: 2022-06-11 16:44:32
  @ FilePath: projectroot/smplmodel/__init__.py
'''
from .body_model import SMPLlayer
from .body_param import load_model
from .body_param import merge_params, select_nf, check_keypoints
from .utils import *
import os
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_dir_path = os.path.dirname(dir_path)
SMPL_DIR = os.path.join(os.path.dirname(dir_dir_path),'data','vibe_data')

SMPL24_JOINTS = [
    'PELVIS','LHIP','RHIP','SPINE1',
    'LKNEE','RKNEE','SPINE2',
    'LANKLE','RANKLE','SPINE3',
    'LFOOT', 'RFOOT', 'NECK',
    'LCLAVICLE', 'RCLAVICLE', 'HEAD',
    'LSHOULDER', 'RSHOULDER', 'LELBOW', 'RELBOW',
    'LWRIST', 'RWRIST', 'LHAND', 'RHAND']


OP25_JOINTS = [
    'OP Nose','OP Neck','OP RShoulder','OP RElbow',
    'OP RWrist','OP LShoulder','OP LElbow',
    'OP LWrist','OP MidHip','OP RHip','OP RKnee',
    'OP RAnkle','OP LHip','OP LKnee','OP LAnkle',
    'OP REye','OP LEye','OP REar','OP LEar',
    'OP LBigToe','OP LSmallToe','OP LHeel','OP RBigToe',
    'OP RSmallToe','OP RHeel']

SK27_JOINTS = [
    'SK TOP',
    'OP Neck',
    'OP RShoulder', 'OP RElbow', 'OP RWrist',
    'OP LShoulder', 'OP LElbow', 'OP LWrist',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP MidHip',
    'OP RBigToe', 'OP LBigToe',
    'OP LEye', 'OP REye', 'OP Nose',
    'SK Chin',
    'OP REar', 'OP LEar',
    'OP LSmallToe', 'OP LHeel',
    'OP RSmallToe', 'OP RHeel'
    ]

OP25_WEIGHTS = [
    1.5, 1.0, 1.0, 1.2, 1.5,
    1.0, 1.2, 1.5,
    1.5, 1.0, 1.2, 1.5,
    1.0, 1.2, 1.5,
    1.2, 1.2, 1.2, 1.2,
    1.7, 1.7, 1.5, 1.7,
    1.7, 1.5]

OP21_JOINTS = [
    'OP LWrist', 'OP RWrist', 'OP LAnkle', 'OP RAnkle',
    'OP LElbow', 'OP RElbow', 'OP LKnee', 'OP RKnee',
    'OP LShoulder','OP RShoulder',
    'OP LHip',  'OP MidHip','OP RHip',
    'OP Nose', 'OP Neck',
    'OP LEye', 'OP REye',  'OP LEar', 'OP REar',
    'OP LBigToe', 'OP RBigToe',
]

OP17_JOINTS = [
    'OP MidHip', 'OP LHip', 'OP RHip',
    'OP LWrist', 'OP RWrist', 'OP LAnkle', 'OP RAnkle',
    'OP LElbow', 'OP RElbow', 'OP LKnee', 'OP RKnee',
    'OP LShoulder', 'OP RShoulder',
    'OP Nose', 'OP Neck',
    'OP LBigToe', 'OP RBigToe',
]

CO18_JOINTS = [
    'OP Nose', 'OP Neck',
    'OP RShoulder', 'OP RElbow', 'OP RWrist',
    'OP LShoulder', 'OP LElbow', 'OP LWrist',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar', 'OP LEar',
]
SMPL24_MAP = {SMPL24_JOINTS[i]: i for i in range(len(SMPL24_JOINTS))}
OP25_MAP = {OP25_JOINTS[i]: i for i in range(len(OP25_JOINTS))}
OP21_MAP = {OP21_JOINTS[i]: i for i in range(len(OP21_JOINTS))}
OP17_MAP = {OP17_JOINTS[i]: i for i in range(len(OP17_JOINTS))}
CO18_MAP = {CO18_JOINTS[i]: i for i in range(len(CO18_JOINTS))}
SK27_MAP = {SK27_JOINTS[i]: i for i in range(len(SK27_JOINTS))}



OP25_TO_OP25_MAP = [OP25_MAP[item] for item in OP25_JOINTS]
OP25_TO_OP21_MAP = [OP25_MAP[item] for item in OP21_JOINTS]
OP25_TO_OP17_MAP = [OP25_MAP[item] for item in OP17_JOINTS]
OP25_TO_CO18_MAP = [OP25_MAP[item] for item in CO18_JOINTS]
SK27_TO_CO25_MAP = [SK27_MAP[item] for item in OP25_JOINTS]



OP25_WEIGHTS = [OP25_WEIGHTS[OP25_MAP[item]] for item in OP25_JOINTS]
OP21_WEIGHTS = [OP25_WEIGHTS[OP25_MAP[item]] for item in OP21_JOINTS]
OP17_WEIGHTS = [OP25_WEIGHTS[OP25_MAP[item]] for item in OP17_JOINTS]
CO18_WEIGHTS = [OP25_WEIGHTS[OP25_MAP[item]] for item in CO18_JOINTS]

OP_JOINT_PAIRS = [('OP Neck', 'OP MidHip'), ('OP RHip', 'OP RKnee'), ('OP RKnee', 'OP RAnkle'), ('OP MidHip', 'OP RHip'), ('OP MidHip', 'OP LHip'), ('OP LHip', 'OP LKnee'),
              ('OP LKnee', 'OP LAnkle'), ('OP Neck', 'OP RShoulder'), ('OP RShoulder', 'OP RElbow'), ('OP RElbow', 'OP RWrist'), ('OP Neck', 'OP LShoulder'),
              ('OP LShoulder', 'OP LElbow'), ('OP LElbow', 'OP LWrist'), ('OP Neck', 'OP Nose'), ('OP Nose', 'OP REye'), ('OP Nose', 'OP LEye'),
              ('OP REye', 'OP REar'), ('OP LEye', 'OP LEar'), ('OP LAnkle', 'OP LBigToe'), ('OP RAnkle', 'OP RBigToe')]
JOINT_ID_PAIRS_OP17 = [(OP17_MAP[first], OP17_MAP[second]) for first, second in OP_JOINT_PAIRS if first in OP17_JOINTS and second in OP17_JOINTS ]
JOINT_ID_PAIRS_OP21 = [(OP21_MAP[first], OP21_MAP[second]) for first, second in OP_JOINT_PAIRS if first in OP21_JOINTS and second in OP21_JOINTS ]
JOINT_ID_PAIRS_OP25 = [(OP25_MAP[first], OP25_MAP[second]) for first, second in OP_JOINT_PAIRS if first in OP25_JOINTS and second in OP25_JOINTS ]
JOINT_ID_PAIRS_CO18 = [(CO18_MAP[first], CO18_MAP[second]) for first, second in OP_JOINT_PAIRS if first in CO18_JOINTS and second in CO18_JOINTS ]

OP25_TO_WHICH_DICT = {
    '17': OP25_TO_OP17_MAP,
    '18': OP25_TO_CO18_MAP,
    '21': OP25_TO_OP21_MAP,
    '25': OP25_TO_OP25_MAP,
}

OPX_WEIGHTS_DICT = {
    '17': OP17_WEIGHTS,
    '18': CO18_WEIGHTS,
    '21': OP21_WEIGHTS,
    '25': OP25_WEIGHTS,
}

OPX_PAIRS_DICT = {
    '17': JOINT_ID_PAIRS_OP17,
    '18': JOINT_ID_PAIRS_CO18,
    '21': JOINT_ID_PAIRS_OP21,
    '25': JOINT_ID_PAIRS_OP25,
}

OPX_LIMBS_DICT = {
    #TODO: add 18
    '17': list(np.arange(3,11)),
    '21': list(np.arange(0,8)),
    '25': [7, 4, 14, 11, 6, 3, 13, 10],
}

FOOTLR_IDXS = [3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, \
          3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, \
          3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, \
          3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, \
          3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, \
          3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, \
          3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, \
          3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, \
          3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, \
          3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, \
          3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, \
          3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, \
          3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 6611, 6612, 6613, 6614, 6615, 6616, 6617, 6618, 6619, \
          6620, 6621, 6622, 6623, 6624, 6625, 6626, 6627, 6628, 6629, 6630, 6631, 6632, 6633, 6634, 6635, 6636, 6637, 6638, 6639, \
          6640, 6641, 6642, 6643, 6644, 6645, 6646, 6647, 6648, 6649, 6650, 6651, 6652, 6653, 6654, 6655, 6656, 6657, 6658, 6659, \
          6660, 6661, 6662, 6663, 6664, 6665, 6666, 6667, 6668, 6669, 6670, 6671, 6672, 6673, 6674, 6675, 6676, 6677, 6678, 6679, \
          6680, 6681, 6682, 6683, 6684, 6685, 6686, 6687, 6688, 6689, 6690, 6691, 6692, 6693, 6694, 6695, 6696, 6697, 6698, 6699, \
          6700, 6701, 6702, 6703, 6704, 6705, 6706, 6707, 6708, 6709, 6710, 6711, 6712, 6713, 6714, 6715, 6716, 6717, 6718, 6727, \
          6728, 6729, 6730, 6731, 6732, 6733, 6734, 6735, 6736, 6737, 6738, 6739, 6740, 6741, 6742, 6743, 6744, 6745, 6746, 6747, \
          6748, 6749, 6750, 6751, 6752, 6753, 6754, 6755, 6756, 6757, 6758, 6759, 6760, 6761, 6762, 6763, 6764, 6765, 6766, 6767, \
          6768, 6769, 6770, 6771, 6772, 6773, 6774, 6775, 6776, 6777, 6778, 6779, 6780, 6781, 6782, 6783, 6784, 6785, 6786, 6787, \
          6788, 6789, 6790, 6791, 6792, 6793, 6794, 6795, 6796, 6797, 6798, 6799, 6800, 6801, 6802, 6803, 6804, 6805, 6806, 6807, \
          6808, 6809, 6810, 6811, 6812, 6813, 6814, 6815, 6816, 6817, 6818, 6819, 6820, 6821, 6822, 6823, 6824, 6825, 6826, 6827, \
          6828, 6829, 6830, 6831, 6832, 6833, 6834, 6835, 6836, 6837, 6838, 6839, 6840, 6841, 6842, 6843, 6844, 6845, 6846, 6847, \
          6848, 6849, 6850, 6851, 6852, 6853, 6854, 6855, 6856, 6857, 6858, 6859, 6860, 6861, 6862, 6863, 6864, 6865, 6866, 6867, 6868, 6869]

FOOT_NORMAL_IDX = 3443

if __name__ == '__main__':
    print()