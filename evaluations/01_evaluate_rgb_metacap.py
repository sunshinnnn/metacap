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
    get_lpips = my_lpips()
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

        psnrList = []
        ssimList = []
        lpipsList = []
        maskCommonList = []
        for i in range(6):
            rgbPredTemp = cv2.imread(osp.join(outDir, str(i) + '.png'), -1)
            rgbPred = rgbPredTemp[:, 1200: 2400,:3][:,:,::-1]
            maskPred = rgbPredTemp[:, 4800:, :3]
            _, maskPred = cv2.threshold(maskPred, 210, 255, cv2.THRESH_BINARY)
            rgbPred = np.where(maskPred==255, rgbPred, 0)

            rgbGTTemp = cv2.imread(osp.join(GTDir, 'RGB_' + str(i) + '.png'), -1)
            rgbGT = rgbGTTemp[:, 0: 1200,:3][:,:,::-1]
            maskGT = rgbGTTemp[:, 0 : 1200, 3:]
            _, maskGT = cv2.threshold(maskGT, 210, 255, cv2.THRESH_BINARY)

            lpips = get_lpips.forward(rgbGT/255.0, rgbPred/255.0, maskGT)
            psnr = get_psnr(rgbGT/255.0, rgbPred/255.0)
            ssim = get_ssim(rgbGT/255.0, rgbPred/255.0, maskGT)

            logger.info('{}_{} | lpips: {} psnr: {} ssim: {}'.format(idx, i, lpips, psnr, ssim))
            psnrList.append(  psnr )
            ssimList.append(  ssim )
            lpipsList.append(  lpips )

        outPath = osp.join( outDir, 'eval_rgb.npz')
        np.savez(outPath,
                 psnr = np.array(psnrList).reshape(1, 6),
                 ssim = np.array(ssimList).reshape(1, 6),
                 lpips=np.array(lpipsList).reshape(1, 6),
                 )

