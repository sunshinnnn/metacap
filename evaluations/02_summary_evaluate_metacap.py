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
    # clothType = human_info[subj]['cloth']
    # idxList = human_info[subj]['test']['evalFrame']

    for subj in subjectList:
        # subj = subjectList[slurm_id]
        clothType = human_info[subj]['cloth']
        idxList = human_info[subj]['test']['evalFrame']

        chamferList = []
        p2sList = []
        iouList = []
        nmlcosList = []
        nmll2List = []

        psnrList = []
        ssimList = []
        lpipsList = []
        for idx in tqdm(idxList):
            meshPath = osp.join(baseDir, subj, clothType, "MetaCap", str(idx).zfill(6), "{}.obj".format(str(idx).zfill(6)))
            meshPathGT = osp.join(baseDir, subj, clothType, "GT", str(idx).zfill(6), str(idx).zfill(6) + ".obj" )
            outDir = osp.dirname(meshPath)
            GTDir = osp.dirname(meshPathGT)

            outPath = osp.join( outDir, 'eval_geo.npz')
            dataz = np.load(outPath)
            chamferList.append( dataz['chamfer'] )
            p2sList.append( dataz['p2s'] )
            iouList.append( dataz['iou'] )
            nmlcosList.append( dataz['nmlcos'] )
            nmll2List.append( dataz['nmll2'] )

            outPath2 = osp.join( outDir, 'eval_rgb.npz')
            dataz2 = np.load(outPath2)
            psnrList.append(dataz2['psnr'])
            ssimList.append(dataz2['ssim'])
            lpipsList.append(dataz2['lpips'])
        chamferList = np.concatenate(chamferList, 0)
        p2sList = np.concatenate(p2sList, 0)
        iouList = np.concatenate(iouList, 0)
        nmlcosList = np.concatenate(nmlcosList, 0)
        nmll2List = np.concatenate(nmll2List, 0)


        psnrList = np.concatenate(psnrList, 0)
        ssimList = np.concatenate(ssimList, 0)
        lpipsList = np.concatenate(lpipsList, 0)
        print("\n {:<30} \n  NmlCos : {:2.3f} \n NmlL2 : {:2.3f} \n\n Chamfer : {:.3f} \n P2S : {:.3f} \n IOU : {:.3f}  \n\n".format(subj,
                                                                                          nmlcosList.mean(),  nmll2List.mean(),
                                                                                          chamferList.mean(),
                                                                                          p2sList.mean(), iouList.mean(),
                                                                                           ))


        print("\n {:<30} \n  PSNR : {:2.3f} \n SSIM : {:2.3f} \n LPIPS : {:2.3f} \n\n ".format(subj,
                                                                                          psnrList.mean(),  ssimList.mean(),
                                                                                          lpipsList.mean(),
                                                                                           ))
