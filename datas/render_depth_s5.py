"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-10-16
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
sys.path.append('..')
import os.path as osp
import numpy as np
import time

from tools.omni_tools import makePath, checkPlatformDir
from tools.skel_tools import Skeleton, loadMotion, saveMotion, SkinnedCharacter, EmbeddedGraph
from tools.torch3d_transforms import euler_angles_to_matrix, matrix_to_euler_angles
from tools.cam_tools import loadCamera
from tools.mesh_tools import save_off
from tools.pyrender_tools import get_extrinc_from_sphere, Renderer, make_rotate, create_point, Mesh, colorDict
import torch

from tqdm import tqdm
import cv2
import glob
import json

def load_motion_deform_param(path):
    params = dict(np.load(str(path)))

    return {
        "motion": params["motionList"].astype(np.float32),
        "deltaR": params["deltaRList"].astype(np.float32),
        "deltaT": params["deltaTList"].astype(np.float32),
        "displacement": params["displacementList"].astype(np.float32),
        "frame": params['frameList'].astype(np.int32),
    }


def load_smpl_param(path, returnTensor = False):
    smpl_params = dict(np.load(str(path)))
    if returnTensor:
        if "thetas" in smpl_params:
            smpl_params["body_pose"] = torch.Tensor(smpl_params["thetas"][..., 3:])
            smpl_params["global_orient"] = torch.Tensor(smpl_params["thetas"][..., :3])
        return {
            "betas": torch.Tensor(smpl_params["betas"].astype(np.float32).reshape(1, 10)),
            "body_pose": torch.Tensor(smpl_params["body_pose"].astype(np.float32)),
            "global_orient": torch.Tensor(smpl_params["global_orient"].astype(np.float32)),
            "transl": torch.Tensor(smpl_params["transl"].astype(np.float32)),
        }
    else:
        if "thetas" in smpl_params:
            smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
            smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
        return {
            "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
            "body_pose": smpl_params["body_pose"].astype(np.float32),
            "global_orient": smpl_params["global_orient"].astype(np.float32),
            "transl": smpl_params["transl"].astype(np.float32),
        }

dataDir = './Subject0005/loose/smoothCharacter'
baseDir = './Subject0005/loose'
charPath = osp.join(dataDir, 'actor.character')
skinPath = osp.join(dataDir, 'actor.skin')
skelPath = osp.join(dataDir, 'actor.skeleton')
motionBasePath = osp.join(dataDir, 'actor.motion')

motionPath = osp.join(dataDir, r'training/motions/poseAngles.motion')
meshPath = osp.join(dataDir, 'actor.obj')
graphPath = osp.join(dataDir, 'actorSimplified.obj')

paramPath = osp.join(baseDir, r'training/motions/ddc_all_smooth_less_zero.npz')
# outDir =  checkPlatformDir('/CT/HOIMOCAP3/nobackup/debug/renderSubjectFullFrames_Subject0005Depth')
# makePath(outDir)

connectionPath = None
device = 'cpu'
useDQ = True
verbose = True
sc = SkinnedCharacter(charPath=charPath, useDQ=useDQ, verbose=verbose,
                      device=device, segWeightsFlag=True, computeAdjacencyFlag=True)
eg = EmbeddedGraph(character=sc, graphPath=graphPath, computeConnectionFlag=False, connectionPath=connectionPath,
                   useDQ=useDQ, verbose=verbose, device=device)
params =  load_motion_deform_param(paramPath)
motionList = torch.Tensor(params["motion"]).to(device)
deltaTList = torch.Tensor(params["deltaT"]).to(device)
deltaRList = torch.Tensor(params["deltaR"]).to(device)
displacementList = torch.Tensor(params["displacement"]).to(device)

skinRs, skinTs = eg.updateNode(motionList[:])
deltaRs = deltaRList[:]
deltaTs = deltaTList[:]
displacements = displacementList[:]

root = checkPlatformDir('/CT/HOIMOCAP3/work/data/Subject0005/loose/training')

fIdx = 1000

cam_path = sorted(glob.glob(f"{root}/recon_neus2/imgs/{str(fIdx).zfill(6)}/*.json"))[0]
Ks, Es, Esc2w = [], [], []
with open(cam_path, 'r') as f:
    cam_data = json.load(f)
    H, W = cam_data['h'], cam_data['w']

render = Renderer(height=H, width=W, camera_type=1, use_ground=False, use_axis=False)


frames =  [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800,
         4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800, 6000, 6200, 6400, 6600, 6800,
         7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000, 9200, 9400, 9600, 9800,
         10000, 10200, 10400, 10600, 10800, 11000, 11200, 11400, 11600, 11800, 12000, 12200, 12400,
         12600, 12800, 13000, 13200, 13400, 13600, 13800, 14000, 14200, 14400, 14600, 14800, 15000,
         15200, 15400, 15600, 15800, 16000, 16200, 16400, 16600, 16800, 17000, 17200, 17400, 17600,
         17800, 18000, 18200, 18400, 18600, 18800, 19000, 19200, 19400]

for fIdx in frames:
    idx = list(params["frame"]).index(fIdx)
    Ks, Es, Esc2w = [], [], []
    cam_path = sorted(glob.glob(f"{root}/recon_neus2/imgs/{str(fIdx).zfill(6)}/*.json"))[0]

    with open(cam_path, 'r') as f:
        cam_data = json.load(f)
        H, W = cam_data['h'], cam_data['w']
        for cIdx in range(len(cam_data['frames'])):
            K = np.array(cam_data['frames'][cIdx]['intrinsic_matrix'])[:3, :3]
            tempE = np.array(cam_data['frames'][cIdx]['transform_matrix'])
            tempE[:3, 3] /= 1000
            tempEw2c = np.linalg.inv(tempE)
            Ks.append(K)
            Es.append(tempE)



    # for idx in tqdm(range(len(skinRs))):
    for cIdx in tqdm(range(len(cam_data['frames']))):
        img_path = sorted(glob.glob(f"{root}/recon_neus2/imgs/{str(fIdx).zfill(6)}/*.png"))[cIdx]
        cam_path = sorted(glob.glob(f"{root}/recon_neus2/imgs/{str(fIdx).zfill(6)}/*.json"))[0]

        outDir = makePath(f"{root}/recon_neus2/depths/{str(fIdx).zfill(6)}")
        baseName = osp.basename(img_path)

        render.set_camera(Ks[cIdx], Es[cIdx])
        vertsF = eg.forwardF(deltaRs=deltaRs[idx: idx+1], deltaTs=deltaTs[idx: idx+1] * 1000, skinRs=skinRs[idx: idx+1], skinTs=skinTs[idx: idx+1], displacements=displacements[idx: idx+1] * 1000.0)

        # mesh = eg.getMeshF(vertsF, idx)
        mesh = eg.getMeshF(vertsF, 0)
        mesh.vertices /= 1000.0
        render.add_mesh(mesh)
        color, depth = render.render()
        color = color[:, :, :3][:, :, ::-1]
        render.del_mesh()

        img = cv2.imread(img_path,-1)[:,:,:3]

        color = cv2.addWeighted(img, 0.5 , color, 0.5, 0)

        jellyOffset = 0.2
        kernel = np.ones((5, 5), np.uint8)
        depth2 = cv2.erode(depth, kernel, iterations=1)
        mask2 = depth2 > 0.0


        depth3 = depth2.copy()
        depth3 = np.where(depth2 > 0, depth3 - jellyOffset, 0.0)

        jellyOffset = 0.2
        kernel = np.ones((5, 5), np.uint8)
        depth2 = cv2.erode(depth, kernel, iterations=1)
        depth4 = depth2.copy()

        depth4 = np.where(depth2 > 0, depth4, 10.0)
        depth4[mask2] = depth4[mask2] + jellyOffset
        out = (np.stack([depth3 / 10.0, depth4 / 10.0, np.zeros_like(depth3)], -1) * 255.0 ).astype(np.uint8)
        cv2.imwrite(osp.join(outDir, baseName), out[:,:,::-1])
