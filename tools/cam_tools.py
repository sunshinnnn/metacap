"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-03-14
"""
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pandas
import json

def load_camera_param(cam_path, scale = 1.0):
    """
    Load camera parameters from a file or a list of files.

    param:
          cam_path: The path to the camera parameter file or a list of paths to multiple camera parameter files.

    return:
          A tuple containing the intrinsic matrices (Ks), extrinsic matrices (Es), image height (H), and image width (W).
    """
    if isinstance(cam_path, str):
        Ks, Es = [], []
        with open(cam_path, 'r') as f:
            cam_data = json.load(f)
            for i in range(len(cam_data['frames'])):
                K = np.eye(4)
                K[:3, :3] = np.array(cam_data['frames'][i]['intrinsic_matrix'])[:3, :3]
                Ks.append(K)
                tempE = np.array(cam_data['frames'][i]['transform_matrix'])
                tempE[:3, 3] /= scale #1000
                tempE = np.linalg.inv(tempE)
                Es.append(tempE)
        # camera = np.load(f"{root}/cameras.npz")
        H, W = cam_data['h'], cam_data['w']
        Ks, Es = np.stack(Ks, 0).astype(np.float32), np.stack(Es, 0).astype(np.float32)
        return Ks, Es, H, W
    elif isinstance(cam_path, list):
        KsAll, EsAll = [], []
        for cam_path_temp in cam_path:
            Ks, Es = [], []
            with open(cam_path_temp, 'r') as f:
                cam_data = json.load(f)
                for i in range(len(cam_data['frames'])):
                    K = np.eye(4)
                    K[:3, :3] = np.array(cam_data['frames'][i]['intrinsic_matrix'])[:3, :3]
                    Ks.append(K)
                    # Ks.append(np.array(cam_data['frames'][i]['intrinsic_matrix'])[:3, :3])
                    tempE = np.array(cam_data['frames'][i]['transform_matrix'])
                    tempE[:3, 3] /= scale #1000
                    tempE = np.linalg.inv(tempE)
                    Es.append(tempE)
            # camera = np.load(f"{root}/cameras.npz")
            H, W = cam_data['h'], cam_data['w']
            Ks, Es = np.stack(Ks, 0), np.stack(Es, 0)
            KsAll.append(Ks), EsAll.append(Es)
        KsAll, EsAll = np.stack(KsAll, 0).astype(np.float32), np.stack(EsAll, 0).astype(np.float32)
        return KsAll, EsAll, H, W
    else:
        raise TypeError("Invalid input type. Expected a string or a list.")


def depth2pointcloud_real(dep, mask, K, E):
    h, w = dep.shape[:2]
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    xx, yy = np.meshgrid(x, y)
    xx = xx.astype('float32')
    yy = yy.astype('float32')

    dirx = (xx - K[0][2]) / K[0][0]
    diry = (yy - K[1][2]) / K[1][1]
    dirz = np.ones((h, w))
    dnm = np.sqrt(dirx ** 2 + diry ** 2 + dirz ** 2);
    #     dep_z = dep/dnm
    dep_z = dep
    #     plt.imshow(dnm)

    pcx = (xx - K[0][2]) / K[0][0] * dep_z
    pcy = (yy - K[1][2]) / K[1][1] * dep_z
    pcz = dep_z
    pc = np.stack([pcx, pcy, pcz], -1).reshape(-1,3)
    pc_m = pc[mask.reshape(-1) > 0]
    # pc_m_new = pc_m
    # pc_m_new = E[:3, :3].dot(pc_m.T) + E[:3, 3:]
    # pc_m_new = pc_m_new.T
    pc_m_new = pc_m.dot(E[:3, :3].T) + + E[:3, 3:].T

    return pc_m_new

def loadCameraJson(camPath, scale = 1000.0):
    Ks, Es, Esc2w = [], [], []
    with open(camPath, 'r') as f:
        cam_data = json.load(f)
        for i in range(len(cam_data['frames'])):
            Ks.append(np.array(cam_data['frames'][i]['intrinsic_matrix'])[:3, :3])
            tempE = np.array(cam_data['frames'][i]['transform_matrix'])
            tempE[:3, 3] /= scale
            tempEw2c = np.linalg.inv(tempE)
            Es.append(tempEw2c)
            Esc2w.append(tempE)
    # camera = np.load(f"{root}/cameras.npz")
    H, W = cam_data['h'], cam_data['w']

    Ks, Es, Esc2w = np.stack(Ks, 0), np.stack(Es, 0), np.stack(Esc2w, 0)
    Size = [H, W]
    return Ks, Es, Size


def loadCamera(camPath, H=-1, W=-1, returnTensor= False, returnNames=False, device=None):
    """
        Input:
            camPath
        Output:
            Ks, Es, PsGL, Sizes[0]
        =====
        Es: world 2 camera
        Size: H,W
    """
    with open(os.path.join(camPath), 'r') as f:
        data = f.readlines()
    assert data[0] == 'Skeletool Camera Calibration File V1.0\n'
    Names = []
    Ks = []
    Es = []
    PsGL = []  #opengl style
    Sizes = []
    for line in data:
        splittedLine = line.split()
        if len(splittedLine) > 0:
            if (splittedLine[0] == 'name'):
                Names.append(splittedLine[1])
            if (splittedLine[0] == 'intrinsic'):
                tempK = np.zeros(16)
                for i in range(1, len(splittedLine)):
                    tempK[i-1] = float(splittedLine[i])
                Ks.append(tempK.reshape(4,4))
            if (splittedLine[0] == 'extrinsic'):
                tempE = np.zeros(16)
                for i in range(1, len(splittedLine)):
                    tempE[i-1] = float(splittedLine[i])
                Es.append(tempE.reshape(4,4))
            if (splittedLine[0] == 'size'):
                Sizes.append( [ float(splittedLine[2]), float(splittedLine[1]) ]) #H,W

    for i in range(len(Ks)):
        K = Ks[i]
        h, w = Sizes[0][0], Sizes[0][1]
        near, far = 0.01, 10000.0
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        tempP = np.array([
            [2 * fx / w, 0.0, (w - 2 * cx) / w, 0],
            [0, 2 * fy / h, (h - 2 * cy) / h, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0]
        ])
        PsGL.append(tempP)
    Ks = np.stack(Ks,0)
    Es = np.stack(Es,0)
    PsGL = np.stack(PsGL,0)
    Sizes = np.stack(Sizes, 0)
    if H>0 or W>0:
        assert H/Sizes[0, 0] == W/Sizes[0, 1]
        scale = H/Sizes[0, 0]
        Ks *= scale
        Ks[:, 2, 2] = 1
        Ks[:, 3, 3] = 1
        Sizes *= scale
        # Sizes = Es.astype('int')
    Sizes= Sizes.astype('int')

    if returnTensor:
        Ks = torch.from_numpy(Ks).float().to(device)
        Es = torch.from_numpy(Es).float().to(device)
        PsGL = torch.from_numpy(PsGL).float().to(device)
        # Sizes = torch.from_numpy(Sizes).to(device)
    if returnNames:
        return Ks, Es, PsGL, Sizes[0], Names
    else:
        return Ks, Es, PsGL, Sizes[0]

def projectPoints(verts, Ks, Es = None, H=-1, W=-1):
    """
        verts: B,N,3
        Ks: B,C,4,4
        Es: B,C,4,4   world2camera
        if H,W >0:
            return NDC  ==> u,v,Z
    """
    B, N, _ = verts.shape
    verts = torch.cat([verts, torch.ones(B,N,1).to(verts.device)], dim = -1) #B,N,4
    if not Es is None:
        verts = torch.einsum('bvm,bcmn->bcvn', verts, Es.transpose(2,3))
    verts = torch.einsum('bcvm,bcmn ->bcvn', verts, Ks.transpose(2, 3))
    verts[:,:,:, [0,1]] /= verts[:,:,:, [2]]
    if H>0 and W>0:
        verts[:, :, :, 0] = 2 * verts[:, :, :, 0] / W - 1
        verts[:, :, :, 1] = 2 * verts[:, :, :, 1] / H - 1
    return verts


def unprojectPoints(verts, PsGL, Es = None):
    """
        verts: B,N,3
        Ks: B,C,4,4
        Es: B,C,4,4   world2camera
        if H,W >0:
            return NDC  ==> u,v,Z
    """
    B, N, _ = verts.shape
    vertsCam = torch.cat([verts, torch.ones(B,N,1).to(verts.device)], dim = -1) #B,N,4
    if not Es is None:
        vertsCam = torch.einsum('bvm, bcmn->bcvn', vertsCam, Es.transpose(2,3))
    vertsCam[:,:,:,[2]] *= -1  # invert z-axis
    # vertsCam[:, :, :, [0, 1]] /= vertsCam[:, :, :, [2]]
    # print(vertsCam.shape)
    # print(PsGL.shape)
    vertsNDC = torch.einsum('bcvm,bcmn ->bcvn', vertsCam, PsGL.transpose(2, 3))

    return vertsNDC

def projectPointsGL(verts, PsGL, Es = None):
    """
        verts: B,N,3
        Ks: B,C,4,4
        Es: B,C,4,4   world2camera
        if H,W >0:
            return NDC  ==> u,v,Z
    """
    B, N, _ = verts.shape
    vertsCam = torch.cat([verts, torch.ones(B,N,1).to(verts.device)], dim = -1) #B,N,4
    if not Es is None:
        vertsCam = torch.einsum('bvm, bcmn->bcvn', vertsCam, Es.transpose(2,3))
    vertsCam[:,:,:,[2]] *= -1  # invert z-axis
    # vertsCam[:, :, :, [0, 1]] /= vertsCam[:, :, :, [2]]
    # print(vertsCam.shape)
    # print(PsGL.shape)
    vertsNDC = torch.einsum('bcvm,bcmn ->bcvn', vertsCam, PsGL.transpose(2, 3))

    return vertsNDC


def index(feat, uv):        # [B, 2, N]
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # feat [B,C,H,W]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]

def dilate(bin_img, ksize=3):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


def erode(bin_img, ksize=3):
    out = 1 - dilate(1 - bin_img, ksize)
    return out