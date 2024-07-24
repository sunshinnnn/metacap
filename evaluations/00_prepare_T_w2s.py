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
    baseDir = checkPlatformDir('/CT/HOIMOCAP4/work/results/metacap/exp')
    outDir = checkPlatformDir('/CT/HOIMOCAP4/work/results/MetaCap_Summary')
    subjectList = [
        'Subject0002',
        # 'Subject0003',
        # 'Subject0005',
        # 'Subject0027',
    ]
    for subj in subjectList:
        clothType = human_info[subj]['cloth']
        fileList = glob.glob(
            checkPlatformDir(
                '{}/{}/neus-domedense-{}-*-test/World_GT@*/save/T_w2s.npz'.format(
                    baseDir, subj, subj)
                ))
        fileList = [ item.replace('\\', '/')  for item in fileList]
        for item in fileList:
            fIdx = item.split('/')[8].split('-')[3]
            # ckptPath = item
            # print(fIdx)
            pathIn = osp.dirname(item)
            baseName = osp.basename(item)
            pathOut = osp.join(outDir, subj, clothType, "T_w2s", str(fIdx).zfill(6))
            makePath(pathOut)
            copyFile(pathIn, pathOut, baseName)
        print()
#