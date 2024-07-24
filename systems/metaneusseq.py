import sys
import numpy as np
import cv2
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim_module
from torch.optim.lr_scheduler import LinearLR
from torch_efficient_distloss import flatten_eff_distloss
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays, transform_rays_w2s, get_rays_batch, transform_rays_w2s_batch
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy
from pytorch3d.loss import chamfer_distance
from systems.utils import parse_optimizer, parse_scheduler, update_module_step
from tools.omni_tools import resizeImg


@systems.register('metaneusseq-system')
class MetaNeuSSeqSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """

    def prepare(self):
        self.automatic_optimization = False
        self.criterions = {'psnr': PSNR()}
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays
        self.global_step_my = 0

    def configure_optimizers(self):
        optim = parse_optimizer(self.config.system.outer_optimizer, self.model)
        ret = {
            'optimizer': optim,
        }
        if 'scheduler' in self.config.system:
            ret.update({
                'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
            })
        return ret

    def forward(self, batch, idx = -1, model=None):
        if idx > -1:
            if model is None:
                return self.model(batch['rays'][idx], stage = batch['stage'], global_step = batch['global_step'])
            else:
                return model(batch['rays'][idx], stage=batch['stage'], global_step=batch['global_step'])
        else:
            if model is None:
                return self.model(batch['rays'] , stage=batch['stage'], global_step=batch['global_step'])
            else:
                return model(batch['rays'], stage=batch['stage'], global_step=batch['global_step'])

    def preprocess_data(self, batch, stage):

        if 'fIdx' in batch: # validation / testing
            fIdx = batch['fIdx'][0]
            cIdx = batch['cIdx']
        else:
            if self.config.dataset.preload:
                fIdx = torch.randint(0, len(self.dataset.frames), size=(1,), device=self.dataset.all_images.device)[0]
                cIdx = torch.randint(0, self.dataset.all_c2w.shape[1], size=(self.config.dataset.inner_steps, self.train_num_rays), device=self.dataset.all_images.device)
            else:
                fIdx = torch.randint(0, len(self.dataset.frames), size=(1,), device=self.dataset.all_c2w.device)[0]
                cIdx = torch.randint(0, self.dataset.all_c2w.shape[1], size=(self.config.dataset.inner_steps, self.train_num_rays), device=self.dataset.all_c2w.device)

        if stage in ['train']:
            c2w = self.dataset.all_c2w[fIdx, cIdx]  # B,4,4
            if self.config.dataset.preload:
                x = torch.randint(0, self.dataset.w, size=(self.config.dataset.inner_steps, self.train_num_rays,), device=self.dataset.all_images.device)
                y = torch.randint(0, self.dataset.h, size=(self.config.dataset.inner_steps, self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                x = torch.randint(0, self.dataset.w, size=(self.config.dataset.inner_steps, self.train_num_rays,), device=self.dataset.all_c2w.device)
                y = torch.randint(0, self.dataset.h, size=(self.config.dataset.inner_steps, self.train_num_rays,), device=self.dataset.all_c2w.device)

            # import pdb
            # pdb.set_trace()
            directions =  self.dataset.directions[y, x, :] # N,3
            K = self.dataset.Ks[fIdx, cIdx] # N,3
            directions = torch.einsum('tnd,tndw->tnw', directions, torch.linalg.inv(K[:,:,:3,:3]).permute([0,1,3,2]))

            if self.config.dataset.preload:
                rgb = self.dataset.all_images[fIdx.cpu(), cIdx.cpu(), y, x, :].to(self.rank)
                fg_mask = self.dataset.all_fg_masks[fIdx.cpu(), cIdx.cpu(), y, x].to(self.rank)
            else:
                all_images_temp_all = []
                all_fg_masks_temp_all = []
                all_depths_temp_all = []
                for fIdx_temp in [fIdx]:
                    all_images_temp, all_fg_masks_temp = [], []
                    all_depths_temp = []
                    for cIdx_temp in torch.arange(self.dataset.all_c2w.shape[1]):
                        # print("{}/{}".format(fIdx,cIdx), flush=True)
                        temp_img = cv2.imread(self.dataset.img_lists[fIdx_temp.cpu()][cIdx_temp.cpu()], -1)
                        if self.config.dataset.img_downscale != 1.0:
                            temp_img = resizeImg(temp_img, scale = 1 / self.config.dataset.img_downscale)
                        img = temp_img[:, :, :3][:, :, ::-1]
                        if self.config.dataset.blur:
                            img = cv2.GaussianBlur(img,(5,5),0)
                        msk = temp_img[:, :, 3] / 255
                        img = (img[..., :3] / 255).astype(np.float32)
                        msk = msk.astype(np.float32)
                        img = torch.Tensor(img)
                        msk = torch.Tensor(msk)
                        all_fg_masks_temp.append(msk) # (h, w)
                        all_images_temp.append(img)
                        if self.config.dataset.with_depth and stage in ['train']:
                            temp_depth = cv2.imread(self.dataset.dep_lists[fIdx_temp.cpu()][cIdx_temp.cpu()], -1)[:, :, ::-1]
                            temp_depth = temp_depth / 255.0 * 10.0
                            if self.config.dataset.img_downscale != 1.0:
                                temp_depth = resizeImg(temp_depth, scale=1 / self.config.dataset.img_downscale)
                            all_depths_temp.append(torch.Tensor(temp_depth))

                    all_images_temp_all.append(torch.stack(all_images_temp, dim=0))
                    all_fg_masks_temp_all.append(torch.stack(all_fg_masks_temp, dim=0))
                    if self.config.dataset.with_depth and stage in ['train']:
                        all_depths_temp_all.append(torch.stack(all_depths_temp, dim=0))

                rgb = torch.stack(all_images_temp_all, dim=0).to(self.rank)[0, cIdx.cpu(), y, x, :]
                fg_mask = torch.stack(all_fg_masks_temp_all, dim=0).to(self.rank)[0, cIdx.cpu(), y, x]

            rays_o, rays_d = get_rays_batch(directions, c2w)
            w2s = self.dataset.w2s[fIdx][None]  # T,4,4
            rays_o, rays_d = transform_rays_w2s_batch(rays_o, rays_d, w2s)

            if self.config.dataset.with_depth:
                if self.config.dataset.preload:
                    near = self.dataset.all_depths[fIdx.cpu(), cIdx.cpu(), y, x, 0:1].to(self.rank) + self.config.dataset.depth_shift
                    far = self.dataset.all_depths[fIdx.cpu(), cIdx.cpu(), y, x, 1:2].to(self.rank) - self.config.dataset.depth_shift
                else:
                    near =  torch.stack(all_depths_temp_all, dim=0).to(self.rank)[0, cIdx.cpu(), y, x, 0:1] + self.config.dataset.depth_shift
                    far =  torch.stack(all_depths_temp_all, dim=0).to(self.rank)[0, cIdx.cpu(), y, x, 1:2] - self.config.dataset.depth_shift
            else:
                near = self.dataset.near[fIdx, cIdx]
                far = self.dataset.far[fIdx,cIdx]

            T_inv = self.dataset.T_inv[fIdx][None]
            self.model.validInOut_grid = self.dataset.validInOut_grid[fIdx]
        else:
            c2w = self.dataset.all_c2w[fIdx, cIdx[0]][None] # 1,4,4
            directions = self.dataset.directions.reshape(-1, 3) # HxWx3
            K = self.dataset.Ks[fIdx, cIdx[0]] # N,3
            directions = torch.einsum('nd,dw-> nw', directions, torch.linalg.inv(K[:3,:3]).permute([1,0]))
            rays_o, rays_d = get_rays(directions, c2w, keepdim=False)
            if self.config.dataset.preload:
                rgb = self.dataset.all_images[fIdx.cpu(), cIdx.cpu()].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
                fg_mask = self.dataset.all_fg_masks[fIdx.cpu(), cIdx.cpu()].view(-1).to(self.rank)
            else:
                all_images_temp_all = []
                all_fg_masks_temp_all = []
                # all_depths_temp = []
                for fIdx_temp in [fIdx]:
                    all_images_temp, all_fg_masks_temp = [], []
                    for cIdx_temp in cIdx:
                        # print("{}/{}".format(fIdx,cIdx), flush=True)
                        temp_img = cv2.imread(self.dataset.img_lists[fIdx_temp.cpu()][cIdx_temp.cpu()], -1)
                        # if self.config.dataset.img_downscale != 1.0:
                        #     temp_img = resizeImg(temp_img, scale = 1 / self.config.dataset.img_downscale)
                        img = temp_img[:, :, :3][:, :, ::-1]
                        if self.config.dataset.blur:
                            img = cv2.GaussianBlur(img,(5,5),0)
                        msk = temp_img[:, :, 3] / 255
                        img = (img[..., :3] / 255).astype(np.float32)
                        msk = msk.astype(np.float32)
                        img = torch.Tensor(img)
                        msk = torch.Tensor(msk)
                        all_fg_masks_temp.append(msk) # (h, w)
                        all_images_temp.append(img)
                    all_images_temp_all.append(torch.stack(all_images_temp, dim=0))
                    all_fg_masks_temp_all.append(torch.stack(all_fg_masks_temp, dim=0))
                rgb = torch.stack(all_images_temp_all, dim=0).to(self.rank)[0, cIdx.cpu(), :, :, :].view(-1, img.shape[-1]).to(self.rank)
                fg_mask = torch.stack(all_fg_masks_temp_all, dim=0).to(self.rank)[0, cIdx.cpu(), :, :].view(-1).to(self.rank)

            w2s = self.dataset.w2s[fIdx][None]
            rays_o, rays_d = transform_rays_w2s(rays_o, rays_d, w2s)
            near = self.dataset.near[fIdx, cIdx]
            far = self.dataset.far[fIdx, cIdx]
            near = near.repeat(rays_o.shape[0], 1)
            far = far.repeat(rays_o.shape[0],1)
            T_inv = self.dataset.T_inv[fIdx][None]

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1), near, far], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])

        self.model.train_meta = True
        self.model.vertices = self.dataset.vertices[fIdx][None]
        # self.model.T_inv = self.dataset.T_inv[:1,:,:,:]
        self.model.face_vertices = self.dataset.face_vertices[fIdx]
        self.model.face_T_inv_flat = self.dataset.face_T_inv_flat[fIdx]
        if self.config.dataset.deformer == 'smpl':
            self.model.deformer = 'smpl'
        self.model.threshold_smpl = self.dataset.threshold_smpl
        self.model.threshold_ddc = self.dataset.threshold_ddc
        # self.model.deformLabels = self.dataset.deformLabels
        self.model.threshold_rigid = self.dataset.threshold_rigid
        self.model.threshold_outer = self.dataset.threshold_outer

        if self.model.faces is None:
            self.model.faces = self.dataset.faces
        batch.update({
            'rays': rays,  # [rays_o, rays_d, near, far] N,8
            'rgb': rgb,
            'fg_mask': fg_mask,
            # 'T_inv': T_inv,
            # 'vertices': vertices,
        })      
    
    def training_step(self, batch, batch_idx):
        batch.update({'stage' : 'train'})
        batch.update({'global_step': self.global_step})

        for p in self.model.parameters():
            p.grad = torch.zeros_like(p.data)

        outer_scheduler = self.lr_schedulers()
        outer_optim = self.optimizers()
        outer_optim.optimizer.zero_grad()

        learner = copy.deepcopy(self.model)
        optimizer_class = getattr(optim_module, self.config.system.inner_optimizer.name)
        inner_optim = optimizer_class(learner.parameters(), **self.config.system.inner_optimizer.args)
        if self.config.system.warmup_steps_inner>0:
            inner_scheduler = LinearLR(inner_optim, start_factor=0.01, end_factor = 1.0, total_iters=500)
        for idx in range(self.config.dataset.inner_steps):
            loss = 0.
            out = self(batch, idx, learner)
            # update train_num_rays
            if self.config.model.dynamic_ray_sampling:
                train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))
                self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

            if self.C(self.config.system.loss.lambda_rgb_huber) > 0:
                loss_rgb_huber = F.huber_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][idx][out['rays_valid_full'][...,0]], reduction="mean", delta=0.1)
                self.log('train/loss_rgb_huber', loss_rgb_huber)
                loss += loss_rgb_huber * self.C(self.config.system.loss.lambda_rgb_huber)

            if self.C(self.config.system.loss.lambda_eikonal) > 0:
                loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
                self.log('train/loss_eikonal', loss_eikonal)
                loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

            if self.C(self.config.system.loss.lambda_mask) > 0:
                opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
                loss_mask = binary_cross_entropy(opacity, batch['fg_mask'][idx].float())
                self.log('train/loss_mask', loss_mask)
                loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

            if self.C(self.config.system.loss.lambda_opaque) > 0:
                loss_opaque = binary_cross_entropy(opacity, opacity)
                self.log('train/loss_opaque', loss_opaque)
                loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

            if self.C(self.config.system.loss.lambda_sparsity) > 0:
                loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
                self.log('train/loss_sparsity', loss_sparsity)
                loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

            if self.config.system.loss.lambda_sdf_reg>0  and self.global_step > 500:
                def points2index(points, bounds, resolutions):
                    temp = (points.reshape(-1, 3) - bounds[0]) / (bounds[1] - bounds[0])
                    temp = (temp * (resolutions - 1))
                    temp2 = temp[:, 2] + temp[:, 1] * resolutions.reshape(-1)[2] + temp[:, 0] * resolutions.reshape(-1)[2] * \
                            resolutions.reshape(-1)[1]
                    return temp2.reshape(-1)

                idxs = points2index(out['positions'], self.dataset.cano_bounds, self.dataset.resolutions)
                valid = idxs < self.dataset.sdf.shape[0]

                loss_sdf_reg = ((out['sdf_samples'].reshape(-1)[valid] - self.dataset.sdf[idxs[valid].to(torch.int64)]) ** 2).mean()
                self.log('train/loss_sdf_reg', loss_sdf_reg)
                loss += loss_sdf_reg * self.C(self.config.system.loss.lambda_sdf_reg)

            # distortion loss proposed in MipNeRF360
            # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
            if self.C(self.config.system.loss.lambda_distortion) > 0:
                loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
                self.log('train/loss_distortion', loss_distortion)
                loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)

            if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
                loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
                self.log('train/loss_distortion_bg', loss_distortion_bg)
                loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)

            losses_model_reg = self.model.regularizations(out)
            for name, value in losses_model_reg.items():
                self.log(f'train/loss_{name}', value)
                loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
                loss += loss_

            self.log('train/inv_s', out['inv_s'], prog_bar=True)

            for name, value in self.config.system.loss.items():
                if name.startswith('lambda'):
                    self.log(f'train_params/{name}', self.C(value))

            self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)
            inner_optim.zero_grad()
            loss.backward()
            inner_optim.step()
            if self.global_step < self.config.system.warmup_steps_inner:
                inner_scheduler.step()

            # self.log('outer_lr', float(outer_optim.optimizer.param_groups[0]['lr']), prog_bar=True)
            # self.log('inner_lr', float(inner_optim.param_groups[0]['lr']), prog_bar=True)



        for p, l in zip(self.model.parameters(), learner.parameters()):
            p.grad.data.add_(-1.0, l.data).add_(p.data)

        outer_optim.step()
        self.global_step_my +=1

        outer_scheduler.step()

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        batch.update({'stage': 'val'})
        batch.update({'global_step': -1})

        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)

            if self.config.export.save_mesh:
                self.export()

    def test_step(self, batch, batch_idx):

        batch.update({'stage': 'test'})
        batch.update({'global_step': -1})
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )
            # self.
            if self.config.export.save_mesh:
                self.export( world = True)
    
    def export(self, world = False):
        meshPath = None
        mesh, level = self.model.export(self.config.export, self.dataset.valid, self.dataset.cano_bounds, self.dataset.validInOut, meshPath = meshPath)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )

        meshPath = None
        mesh, level = self.model.export(self.config.export, self.dataset.valid_world, self.dataset.world_bounds,
                                        self.dataset.validInOut_world, self.dataset.T_inv_world, meshPath=meshPath)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}_world.obj",
            **mesh
        )

