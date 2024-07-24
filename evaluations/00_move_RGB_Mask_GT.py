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

np.set_printoptions(precision = 3, suppress=True)

if __name__ == '__main__':
    logger, _ = setup_logger(save=False)
    baseDir = checkPlatformDir('/CT/HOIMOCAP4/work/data/WildDynaCap_official')
    outDir = checkPlatformDir('/CT/HOIMOCAP4/work/results/MetaCap_Summary')
    subjectList = [
        'Subject0002',
        # 'Subject0003',
        # 'Subject0005',
        # 'Subject0027',
    ]
    for subj in subjectList:
        clothType = human_info[subj]['cloth']
        evalIdxList = human_info[subj]['test']['evalView']

        idxList = human_info[subj]['test']['evalFrame']
        for fIdx in tqdm(idxList):
            pathOut = osp.join(outDir, subj, clothType, "GT", str(fIdx).zfill(6))
            makePath(pathOut)
            for i in range(6):
                cIdx = evalIdxList[i]
                pathIn = osp.join(baseDir, subj, clothType, "testing", "recon_neus2", "imgs", str(fIdx).zfill(6), )
                copyFile(pathIn, pathOut, "image_c_{}_f_{}.png".format(str(cIdx).zfill(3) , str(fIdx).zfill(6)) , 'RGB_{}.png'.format(i))
