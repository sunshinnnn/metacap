import os
import os.path as osp
import pdb
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays, transform_rays_w2s
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy
from pytorch3d.loss import chamfer_distance


@systems.register('neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.automatic_optimization=False
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays
        if self.config.model.dynamic_ray_sampling:
            pass
        else:
            self.train_num_rays = self.config.model.max_train_num_rays
    def forward(self, batch):
        if self.config.dataset.compute_occlusion:
            return self.model(batch['rays'], None, stage=batch['stage'], global_step=batch['global_step'],
                              rays_template= batch['rays_template'], background_color=batch['background_color'])
        else:
            return self.model(batch['rays'], None, stage = batch['stage'], global_step = batch['global_step'], background_color=batch['background_color'])

    def preprocess_data(self, batch, stage):
        # if 'index' in batch: # validation / testing
        #     index = batch['index']
        # else:
        #     if self.config.model.batch_image_sampling:
        #         index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
        #     else:
        #         index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:
            rays = batch['rays'][0].to(self.model.rank)
            rgb = batch['rgb'][0].to(self.model.rank)
            fg_mask = batch['fg_mask'][0].to(self.model.rank)
            background_color = batch['background_color'][0].to(self.model.rank)
            if self.config.dataset.compute_occlusion:
                rgb_template = batch['rgb_template'][0].to(self.model.rank)
                fg_mask_template = batch['fg_mask_template'][0].to(self.model.rank)
                rays_template = batch['rays_template'][0].to(self.model.rank)
            else:
                rgb_template = None
                fg_mask_template = None
                rays_template = None
            # pdb.set_trace()

            # pdb.set_trace()
            # print(fg_mask.sum())
            # 'rays': rays,
            # 'rgb': rgb,
            # 'fg_mask': fg_mask,
            # 'rgb_template': rgb_template,
            # 'fg_mask_template': fg_mask_template,
            # 'rays_template': rays_template,

            # c2w = self.dataset.all_c2w[index]
            # if self.config.model.dynamic_ray_sampling:
            #     x = torch.randint(0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device)
            #     y = torch.randint(0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device)
            # elif not self.config.model.dynamic_ray_sampling:
            #     x = torch.randint(0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device)
            #     y = torch.randint(0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device)
            # # else:
            #     # x = self.x
            #     # y = self.y
            #
            # if self.dataset.directions.ndim == 3: # (H, W, 3)
            #     directions = self.dataset.directions[y, x]
            # elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
            #     directions = self.dataset.directions[index, y, x]
            #
            # rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]) #.to(self.rank)
            # fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1) #.to(self.rank)
            # rays_o, rays_d = get_rays(directions, c2w, keepdim=False)
            # # print(fg_mask.sum())
            # # pdb.set_trace()
            # w2s = self.dataset.w2s[:]
            # rays_o, rays_d = transform_rays_w2s(rays_o, rays_d, w2s)
            #
            # if self.config.dataset.with_depth:
            #     near = self.dataset.all_depths[index, y,x, 0:1] + self.config.dataset.depth_shift
            #     far = self.dataset.all_depths[index, y, x, 1:2] - self.config.dataset.depth_shift
            # else:
            #     near = self.dataset.near[index]
            #     far = self.dataset.far[index]
            #
            # if self.config.dataset.compute_occlusion:
            #     self.train_num_rays_template = int(self.train_num_rays * self.dataset.occlusion_rate)
            #     index_template = torch.randint(0, len(self.dataset.all_images_template), size=(self.train_num_rays_template,), device=self.dataset.all_images_template.device)
            #     rgb_template = self.dataset.all_images_template[index_template]
            #     fg_mask_template = self.dataset.all_fg_masks_template[index_template]
            #     rays_template = self.dataset.rays_template[index_template]
        else:
            rays = batch['rays'][0].to(self.model.rank)
            rgb = batch['rgb'][0].to(self.model.rank)
            fg_mask = batch['fg_mask'][0].to(self.model.rank)
            background_color = batch['background_color'][0].to(self.model.rank)
            # c2w = self.dataset.all_c2w[index]
            # if self.dataset.directions.ndim == 3: # (H, W, 3)
            #     directions = self.dataset.directions
            # elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
            #     directions = self.dataset.directions[index].reshape(-1,3)
            # rays_o, rays_d = get_rays(directions, c2w, keepdim=False)
            # rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            # fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)
            #
            # w2s = self.dataset.w2s[:]
            # rays_o, rays_d = transform_rays_w2s(rays_o, rays_d, w2s)
            #
            # near = self.dataset.near[index]
            # far = self.dataset.far[index]
            # near = near.repeat(rays_o.shape[0], 1)
            # far = far.repeat(rays_o.shape[0],1)

        # rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1), near, far], dim=-1)
        #
        # if stage in ['train']:
        #     if self.config.model.background_color == 'white':
        #         self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        #     elif self.config.model.background_color == 'random':
        #         self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank) # affect a lot !
        #     else:
        #         raise NotImplementedError
        # else:
        #     self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        # #
        # if self.dataset.apply_mask:
        #     rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        #
        #     if self.config.dataset.compute_occlusion and stage=='train':
        #         rgb_template = rgb_template * fg_mask_template[..., None] + self.model.background_color * (1 - fg_mask_template[..., None])
        # pdb.set_trace()

        # if stage in ['train']:
        #     if self.config.model.background_color == 'white':
        #         self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        #     elif self.config.model.background_color == 'random':
        #         self.model.background_color = background_color
        #     else:
        #         raise NotImplementedError
        # else:
        #     self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)

        # if self.global_step == 0:
        if self.model.vertices_template is None:
            # print(self.global_step)
            if self.config.model.grid_pre:
                self.model.validInOut_grid = self.dataset.valid_occ
            if stage in ['test']:
                self.model.rotate_template = self.dataset.rotate_template

            if self.model.vertices_template is None:
                self.model.vertices_template = self.dataset.vs_template

            if self.model.vertices is None:
                self.model.vertices = self.dataset.vertices

            if self.model.T_inv is None:
                self.model.T_inv = self.dataset.T_inv[:1,:,:,:]

            if self.model.face_vertices is None:
                self.model.face_vertices = self.dataset.face_vertices

            # if self.model.face_featOnehot is None:
            #     self.model.face_featOnehot = self.dataset.face_featOnehot

            if self.model.face_T_inv_flat is None:
                self.model.face_T_inv_flat = self.dataset.face_T_inv_flat

            if self.config.dataset.deformer == 'smpl':
                self.model.deformer = 'smpl'

            # self.model.deformLabels = self.dataset.deformLabels
            self.model.threshold_rigid = self.dataset.threshold_rigid
            self.model.threshold_smpl = self.dataset.threshold_smpl
            self.model.threshold_ddc = self.dataset.threshold_ddc
            self.model.threshold_outer = self.dataset.threshold_outer

            if self.model.faces is None:
                self.model.faces = self.dataset.faces

        # if self.global_step == 0:
        #     print(stage)
            if stage in ['train']:
                if self.config.model.background_color == 'white':
                    self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
                elif self.config.model.background_color == 'random':
                    self.model.background_color = torch.rand((3,), dtype=torch.float32,
                                                             device=self.rank)  # affect a lot !
                else:
                    raise NotImplementedError
        if not stage in ['train']:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)

        batch.update({
            'rays': rays, # [rays_o, rays_d, near, far] N,8
            'rgb': rgb,
            'fg_mask': fg_mask,
            'background_color': background_color,
            # 'T_inv': T_inv,
            # 'vertices': vertices,
            'rays_template': rays_template if self.config.dataset.compute_occlusion and stage == 'train' else None,
            'rgb_template': rgb_template if self.config.dataset.compute_occlusion and stage == 'train' else None,
            'fg_mask_template': fg_mask_template if self.config.dataset.compute_occlusion and stage == 'train' else None,
        })      
    
    def training_step(self, batch, batch_idx):
        batch.update({'stage' : 'train'})
        batch.update({'global_step': self.global_step})
        optimizer = self.optimizers()
        lr_schedulers = self.lr_schedulers()
        out = self(batch)

        loss = 0.
        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        if self.C(self.config.system.loss.lambda_rgb_l1) > 0:
            loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
            self.log('train/loss_rgb', loss_rgb_l1)
            loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)

        if self.C(self.config.system.loss.lambda_rgb_huber) > 0:
            loss_rgb_huber = F.huber_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]], reduction="mean", delta=0.1)
            self.log('train/loss_rgb_huber', loss_rgb_huber)
            loss += loss_rgb_huber * self.C(self.config.system.loss.lambda_rgb_huber)

        if self.C(self.config.system.loss.lambda_eikonal) > 0:
            loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
            self.log('train/loss_eikonal', loss_eikonal)
            loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

        if self.C(self.config.system.loss.lambda_mask) > 0:
            opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
            loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
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
        # pdb.set_trace()
        if self.config.dataset.compute_occlusion:
            loss_rgb_huber_template = F.huber_loss(
                out['comp_rgb_template_full'][out['rays_valid_template_full'][..., 0]],
                batch['rgb_template'][out['rays_valid_template_full'][..., 0]], reduction="mean", delta=0.1)
            self.log('train/loss_rgb_huber_template', loss_rgb_huber_template)
            loss += loss_rgb_huber_template * self.C(
                self.config.system.loss.lambda_rgb_huber) * 1.0  # *self.dataset.occlusion_rate #* 0.0 *

            opacity_template = torch.clamp(out['opacity_template'].squeeze(-1), 1.e-3, 1. - 1.e-3)
            loss_mask_template = binary_cross_entropy(opacity_template, batch['fg_mask_template'].float())
            self.log('train/loss_mask_template', loss_mask_template)
            loss += loss_mask_template * (self.C(
                self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0) * 1.0 * self.dataset.occlusion_rate  # * 0.0

            loss_eikonal_template = (
                        (torch.linalg.norm(out['sdf_grad_samples_template'], ord=2, dim=-1) - 1.) ** 2).mean()
            # loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples_cano'], ord=2, dim=-1) - 1.)**2).mean()
            self.log('train/loss_eikonal_template', loss_eikonal_template)
            loss += loss_eikonal_template * self.C(
                self.config.system.loss.lambda_eikonal) * 1.0 * self.dataset.occlusion_rate  # * 0.0

            loss_opaque_template = binary_cross_entropy(opacity_template, opacity_template)
            self.log('train/loss_opaque_template', loss_opaque_template)
            loss += loss_opaque_template * self.C(
                self.config.system.loss.lambda_opaque) * 1.0 * self.dataset.occlusion_rate  # * 0.0

            loss_sparsity_template = torch.exp(
                -self.config.system.loss.sparsity_scale * out['sdf_samples_template'].abs()).mean()
            self.log('train/loss_sparsity_template', loss_sparsity_template)
            loss += loss_sparsity_template * self.C(
                self.config.system.loss.lambda_sparsity) * 1.0 * self.dataset.occlusion_rate  # * 0.0

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
        if self.config.dataset.compute_occlusion:
            self.log('train/num_rays_template', float(self.dataset.train_num_rays_template), prog_bar=True)

        optimizer.optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        lr_schedulers.step()
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
        # print("val: ", batch['index'][0].item())
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png",
         ([{'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}] if not self.config.dataset.rotate else []) +
        [{'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ]   + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3) * torch.where(out['opacity'].view(H, W, 1)> 0.5 , 1, 0), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}, },
            {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None}},
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
                self.export(stage='val')

    def test_step(self, batch, batch_idx):
        batch.update({'stage': 'test'})
        batch.update({'global_step': -1})
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh

        out_normal = out['comp_normal'].view(H, W, 3)
        out_normal = (out_normal + 1)/2
        out_normal = torch.where( torch.cat(  [out['opacity'].view(H, W, 1), out['opacity'].view(H, W, 1), out['opacity'].view(H, W, 1)], -1)> 0.5, out_normal, 1)

        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png",
         ([{'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}] if not self.config.dataset.rotate else []) +
        [{'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ]   + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
                 {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
                 {'type': 'rgb', 'img': out_normal, 'kwargs': {'data_format': 'HWC', 'data_range': (0, 1)}, },
                 {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None}},
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
            if self.config.export.save_mesh:
                self.export( stage = 'test')
    
    def export(self, stage = None):
        if self.config.dataset.cano_motion=='world':
            pass
        else:
            if stage=='val' and (self.config.trainer.val_check_interval==10 or self.config.trainer.val_check_interval==1):
                pass
            else:
                meshPath = None
                mesh, level = self.model.export(self.config.export, self.dataset.valid, None, self.dataset.cano_bounds, self.dataset.validInOut, meshPath = meshPath)
                self.save_mesh(
                    f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
                    **mesh
                )
        meshPath = None
        mesh, level = self.model.export(self.config.export, self.dataset.valid_world,  self.dataset.featOnehot_world, self.dataset.world_bounds,
                                        self.dataset.validInOut_world, self.dataset.T_inv_world, meshPath=meshPath)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}_world.obj",
            **mesh
        )

