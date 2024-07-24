"""
Credits to https://github.com/zju3dv/animatable_nerf, https://github.com/simpleig/Geo-PIFu, https://github.com/autonomousvision/occupancy_networks
"""

import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pandas
import trimesh
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import lpips
import pytorch3d
from pytorch3d import ops
import torch.nn as nn


def get_chamfer_dist(tgtMesh, srcMesh, num_samples=1000):
    # Chamfer
    src_surf_pts, _ = trimesh.sample.sample_surface(srcMesh, num_samples)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgtMesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgtMesh, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(srcMesh, tgt_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    src_tgt_dist = src_tgt_dist.mean()
    tgt_src_dist = tgt_src_dist.mean()

    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2
    return chamfer_dist * 100.0

def get_surface_dist(tgtMesh, srcMesh, num_samples=10000):
    # P2S
    src_surf_pts, _ = trimesh.sample.sample_surface(srcMesh, num_samples)
    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgtMesh, src_surf_pts)
    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    src_tgt_dist = src_tgt_dist.mean()
    return src_tgt_dist * 100.0

def get_mesh_iou(tgtMesh, srcMesh, RES = 128):
    from .extensions.libmesh.inside_mesh import check_mesh_contains
    min_xyz = tgtMesh.vertices.min(0)
    max_xyz = tgtMesh.vertices.max(0)
    min_xyz[:2] -= 0.05
    max_xyz[:2] += 0.05
    min_xyz[2] -= 0.15
    max_xyz[2] += 0.15

    x_coords = np.linspace(0, 1, num=RES, dtype=np.float32)
    y_coords = np.linspace(0, 1, num=RES, dtype=np.float32)
    z_coords = np.linspace(0, 1, num=RES, dtype=np.float32)
    xv, yv, zv = np.meshgrid(x_coords, y_coords, z_coords)
    xv = np.reshape(xv, (-1, 1))  # print(xv.shape) # (256*256*256, 1)
    yv = np.reshape(yv, (-1, 1))
    zv = np.reshape(zv, (-1, 1))
    pts = np.concatenate([xv, yv, zv], axis=-1)

    occ1 = check_mesh_contains(tgtMesh, pts)[0]
    occ2 = check_mesh_contains(srcMesh, pts)[0]

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

def get_psnr(img_gt, img_pred):
    # mse = np.mean((img_pred - img_gt)**2)
    # psnr = -10 * np.log(mse) / np.log(10)
    psnr = compute_psnr(img_gt, img_pred)
    return psnr

def get_ssim(img_gt, img_pred, mask_gt=None):
    if not mask_gt is None:
        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_gt.astype(np.uint8))
        img_pred = img_pred[y:y + h, x:x + w]
        img_gt = img_gt[y:y + h, x:x + w]
    # compute the ssim
    ssim = compare_ssim(img_pred, img_gt, multichannel=True)
    return ssim

class my_lpips():
    def __init__(self):
        self.device='cuda:0'
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)

    def forward(self, img_gt, img_pred, mask_gt=None):
        if not mask_gt is None:
            x, y, w, h = cv2.boundingRect(mask_gt.astype(np.uint8))
            img_pred = img_pred[y:y + h, x:x + w]
            img_gt = img_gt[y:y + h, x:x + w]
        else:
            h, w = img_gt.shape[:2]

        img_pred = torch.tensor(img_pred.copy(), dtype=torch.float32, device=self.device).reshape(1, h, w, 3).permute(0, 3, 1, 2)
        img_gt = torch.tensor(img_gt.copy(), dtype=torch.float32, device=self.device).reshape(1, h, w, 3).permute(0, 3, 1, 2)
        score = self.loss_fn_vgg(img_pred, img_gt, normalize=True)
        return score.item()

def get_mse(img_gt, img_pred):
    mse = np.mean((img_pred - img_gt)**2)
    return mse

def get_normal_dist(nml_gt, nml_pred, mask_gt= None, T = None):
    if mask_gt is None:
        # raise NotImplementedError
        mask_gt = np.ones(nml_gt[:,:,:1])
    # init.
    mask_gt = np.where(mask_gt>0,1,0)
    msk_sum = np.sum(mask_gt)

    nml_gt = nml_gt.astype(np.float32) / 255.0 * 2 - 1
    nml_pred = nml_pred.astype(np.float32) / 255.0 * 2 - 1
    if not T is None :
        R_ = np.linalg.inv(T[:3,:3]).T
        nml_pred = nml_pred.dot(R_.T)


    # ----- cos. dis in (0, 2) -----
    cos_diff_map_pred = mask_gt*(1-np.sum(nml_pred * nml_gt, axis=-1, keepdims=True))
    cos_error2 = (np.sum(cos_diff_map_pred) / msk_sum).astype(np.float32)

    # ----- l2 dis in (0, 4) -----
    l2_diff_map_pred = mask_gt*np.linalg.norm(nml_pred-nml_gt, axis=-1, keepdims=True)
    l2_error2 = (np.sum(l2_diff_map_pred) / msk_sum).astype(np.float32)

    return cos_error2, l2_error2, cos_diff_map_pred, l2_diff_map_pred


if __name__ == '__main__':
    pass



