"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-11-03
"""

import os
import os.path as osp
import pdb
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
            pathInTemp = osp.dirname(item)
            fileListSub = os.listdir(pathInTemp)
            if 'it0-test' in fileListSub:
                imgFolder = 'it0-test'
                meshName = 'it0-mc[256, 512, 256]_world.obj'
            else:
                imgFolder = 'it3000-test'
                meshName = 'it3000-mc[256, 512, 256]_world.obj'
            pathOut = osp.join(outDir, subj, clothType, "GT", str(fIdx).zfill(6))
            makePath(pathOut)
            logger.info(pathInTemp + ' |  | ' + pathOut)
            for i in range(6):
                pathIn = osp.join(pathInTemp, imgFolder)
                copyFile(pathIn, pathOut, '{}.png'.format(i))
            copyFile(pathInTemp, pathOut, meshName, str(fIdx).zfill(6) + '.obj')