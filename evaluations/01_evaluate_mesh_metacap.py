"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-11-03
"""

import os
import os.path as osp
import sys
sys.path.append('..')
import glob
import numpy as np
import trimesh
from sys import platform
import shutil
from multiprocessing import Pool
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from tools.omni_tools import makePath, checkPlatformDir, resizeImg, setup_logger, copyFile
from tools.cam_tools import loadCameraJson
from tools.metric_tools import get_chamfer_dist, get_surface_dist, get_psnr, get_ssim, my_lpips, get_normal_dist, get_mesh_iou
from datasets.human_info import human_info


if __name__ == '__main__':
    logger, _ = setup_logger(save=False)
    baseDir = checkPlatformDir('/CT/HOIMOCAP4/work/results/MetaCap_Summary')
    subjectList = [
        'Subject0002',
        # 'Subject0003',
        # 'Subject0005',
        # 'Subject0027',
    ]
    parser = argparse.ArgumentParser(description="MetaCap")
    parser.add_argument("-subi", '--subject_idx', default=0, help="max epoch", type=int)
    args = parser.parse_args()
    subject_idx = args.subject_idx
    subj = subjectList[subject_idx]
    clothType = human_info[subj]['cloth']
    idxList = human_info[subj]['test']['evalFrame']

    for idx in tqdm(idxList):
        meshPath = osp.join(baseDir, subj, clothType, "MetaCap", str(idx).zfill(6), "{}.obj".format(str(idx).zfill(6)))
        meshPathGT = osp.join(baseDir, subj, clothType, "GT", str(idx).zfill(6), str(idx).zfill(6) + ".obj" )
        outDir = osp.dirname(meshPath)
        GTDir = osp.dirname(meshPathGT)
        logger.info(osp.dirname(meshPath))

        meshPred = trimesh.load(meshPath, process=False)
        # meshPred.apply_transform(T_w2s)
        meshGT = trimesh.load(meshPathGT, process=False)

        chamfer = get_chamfer_dist(meshGT, meshPred)
        p2s = get_surface_dist(meshGT, meshPred)
        iou = get_mesh_iou(meshGT, meshPred)
        nmlcosList = []
        nmll2List = []
        maskCommonList = []
        for i in range(6):
            normPredTemp = cv2.imread(osp.join(outDir, str(i) + '.png'), -1)
            normPred = normPredTemp[:, 3600:4800,:3][:,:,::-1]
            maskPred = normPredTemp[:, 4800:, :3]

            normGTTemp = cv2.imread(osp.join(GTDir, str(i) + '.png'), -1)
            normGT = normGTTemp[:, 3600:4800,:3][:,:,::-1]
            maskGT = normGTTemp[:, 4800:,:3]

            maskPred = cv2.cvtColor(maskPred, cv2.COLOR_BGR2GRAY)
            _, maskPred = cv2.threshold(maskPred, 210, 255, cv2.THRESH_BINARY)
            maskGT = cv2.cvtColor(maskGT, cv2.COLOR_BGR2GRAY)
            _, maskGT = cv2.threshold(maskGT, 210, 255, cv2.THRESH_BINARY)

            maskCommon = cv2.bitwise_and(maskPred, maskGT)
            maskCommonList.append(maskCommon)

            nmlcos, nmll2, cos_diff_map_pred, l2_diff_map_pred = get_normal_dist(normGT, normPred, maskCommon[:,:,None])
            nmlcosList.append(nmlcos)
            nmll2List.append(nmll2)
            # logger.info('{}_{} | nmlcos: {} nmll2: {} nml0.3: {}'.format(idx, i, nmlcos, nmll2, (cos_diff_map_pred[maskCommon>0]<0.01).sum() / (maskCommon>0).sum()))

        logger.info('{} | chamfer: {} p2s: {} iou: {}'.format(idx,  chamfer, p2s, iou))
        cv2.imwrite(osp.join(outDir, 'debug_mask_common.png'), np.concatenate(maskCommonList,1))
        outPath = osp.join( outDir, 'eval_geo.npz')
        np.savez(outPath,
                 chamfer = np.array(chamfer).reshape(1, 1),
                 p2s = np.array(p2s).reshape(1, 1),
                 iou = np.array(iou).reshape(1, 1),
                 nmlcos = np.array(nmlcosList).reshape(1, 6),
                 nmll2 = np.array(nmll2List).reshape(1, 6),
                 )
