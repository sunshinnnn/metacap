import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import scale_anything, get_activation, cleanup, chunk_batch, save_ply
from models.network_utils import get_encoding, get_mlp, get_encoding_with_network
from tools.utils.misc import get_rank
from systems.utils import update_module_step
from nerfacc import ContractionType

def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_anything(x, (-radius, radius), (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x


class MarchingCubeHelper(nn.Module):
    def __init__(self, resolution, use_torch=True):
        super().__init__()
        self.resolution = resolution
        self.use_torch = use_torch
        self.points_range = (0, 1)
        if self.use_torch:
            import torchmcubes
            self.mc_func = torchmcubes.marching_cubes
        else:
            import mcubes
            self.mc_func = mcubes.marching_cubes
        self.verts = None

    def grid_vertices(self):
        if self.verts is None:
            x, y, z = torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)
            x, y, z = torch.meshgrid(x, y, z, indexing='ij')
            verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts
        return self.verts

    def forward(self, level, threshold=0.):
        level = level.float().view(self.resolution[0], self.resolution[1], self.resolution[2])
        if self.use_torch:
            verts, faces = self.mc_func(level.to(get_rank()), threshold)
            verts, faces = verts.cpu(), faces.cpu().long()
            verts = verts / torch.Tensor(self.resolution).reshape(-1,3)
        else:
            verts, faces = self.mc_func(-level.numpy(), threshold) # transform to numpy
            verts, faces = torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64)) # transform back to pytorch
            verts = verts / np.array(self.resolution).reshape(-1, 3)

        return {
            'v_pos': verts.float(),
            't_pos_idx': faces
        }


class BaseImplicitGeometry(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        if self.config.isosurface is not None:
            assert self.config.isosurface.method in ['mc', 'mc-torch']
            if self.config.isosurface.method == 'mc-torch':
                raise NotImplementedError("Please do not use mc-torch. It currently has some scaling issues I haven't fixed yet.")
            self.helper = MarchingCubeHelper(self.config.isosurface.resolution, use_torch=self.config.isosurface.method=='mc-torch')
        self.radius = self.config.radius
        self.contraction_type = None # assigned in system

    def forward_level(self, points):
        raise NotImplementedError

    def isosurface_(self, valid = None, featOnehot= None,  bounds = None, validInOut = None, T_inv =None, meshPath = None):
        def batch_func(x, featOnehot=None):
            rv = self.forward_level(x, featOnehot = featOnehot).cpu()
            cleanup()
            return rv

        if not valid is None:
            x_coords = torch.linspace(0, 1, steps=self.config.isosurface.resolution[0],
                                      dtype=torch.float32, device=self.rank).detach()
            y_coords = torch.linspace(0, 1, steps=self.config.isosurface.resolution[1],
                                      dtype=torch.float32, device=self.rank).detach()
            z_coords = torch.linspace(0, 1, steps=self.config.isosurface.resolution[2],
                                      dtype=torch.float32, device=self.rank).detach()
            xv, yv, zv = torch.meshgrid(x_coords, y_coords, z_coords)  # print(xv.shape) # (256, 256, 256)
            xv = torch.reshape(xv, (-1, 1))  # print(xv.shape) # (256*256*256, 1)
            yv = torch.reshape(yv, (-1, 1))
            zv = torch.reshape(zv, (-1, 1))
            pts = torch.cat([xv, yv, zv], dim=-1)
            pts = pts * (bounds[1] - bounds[0]) + bounds[0]
            pts = pts.to(self.rank)
            level = 10 * torch.ones(self.config.isosurface.resolution[0] * self.config.isosurface.resolution[1] * self.config.isosurface.resolution[2]) #.to(self.rank)
            if not validInOut is None:
                level[validInOut.cpu()] *= -1

            if not T_inv is None:
                pts_valid =  (T_inv[..., :3, :3] @ pts[valid][..., None]).squeeze(-1) + T_inv[..., :3, 3]
                level_ = chunk_batch(batch_func, self.config.isosurface.chunk, True, pts_valid, featOnehot)
            else:
                pts_valid = pts[valid]
                level_ = chunk_batch(batch_func, self.config.isosurface.chunk, True, pts[valid], featOnehot)

            level[valid.cpu()] = level_.to(level.dtype)
            mesh = self.helper(level, threshold=self.config.isosurface.threshold)

            bounds_cpu = bounds.cpu()
            mesh['v_pos'] = mesh['v_pos'] * (bounds_cpu[1] - bounds_cpu[0]) + bounds_cpu[0]

            if not meshPath is None:
                import cv2
                errmap = ((level_.cpu().numpy().reshape(-1) / 0.05) + 1 )/2
                colors = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                save_ply(meshPath, points=pts[valid].data.cpu().numpy().reshape(-1,3), colors = colors)

        else:
            level = chunk_batch(batch_func, self.config.isosurface.chunk, True, self.helper.grid_vertices())
            mesh = self.helper(level, threshold=self.config.isosurface.threshold)
            vmin = (-self.radius, -self.radius, -self.radius)
            vmax = (self.radius, self.radius, self.radius)
            mesh['v_pos'] = torch.stack([
                scale_anything(mesh['v_pos'][..., 0], (0, 1), (vmin[0], vmax[0])),
                scale_anything(mesh['v_pos'][..., 1], (0, 1), (vmin[1], vmax[1])),
                scale_anything(mesh['v_pos'][..., 2], (0, 1), (vmin[2], vmax[2]))
            ], dim=-1)

        return mesh, level

    @torch.no_grad()
    def isosurface(self, valid = None, featOnehot = None, bounds = None, validInOut = None, T_inv = None, meshPath = None):
        if self.config.isosurface is None:
            raise NotImplementedError
        mesh_coarse, level = self.isosurface_(valid = valid, featOnehot = featOnehot, bounds = bounds, validInOut = validInOut, T_inv = T_inv, meshPath= meshPath)
        return mesh_coarse, level

@models.register('volume-density')
class VolumeDensity(BaseImplicitGeometry):
    def setup(self):
        self.n_input_dims = self.config.get('n_input_dims', 3)
        self.n_output_dims = self.config.feature_dim
        self.encoding_with_network = get_encoding_with_network(self.n_input_dims, self.n_output_dims, self.config.xyz_encoding_config, self.config.mlp_network_config)
    
    def forward(self, points, T_inv = None):
        points = points.clone()
        if not T_inv is None:
            T_inv = T_inv.clone()
        # points.requires_grad_(True)

        if not T_inv is None:
            points_cano = (T_inv[..., :3, :3] @ points[..., None]).squeeze(-1) + T_inv[..., :3, 3]
        else:
            points_cano = points

        points_cano = contract_to_unisphere(points_cano, self.radius, self.contraction_type)
        out = self.encoding_with_network(points_cano.view(-1, self.n_input_dims)).view(*points_cano.shape[:-1], self.n_output_dims).float()
        density, feature = out[...,0], out
        if 'density_activation' in self.config:
            density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        if 'feature_activation' in self.config:
            feature = get_activation(self.config.feature_activation)(feature)
        return density, feature
    
    def forward_level(self, points, T_inv = None):
        if not T_inv is None:
            T_inv = T_inv.clone()
        if not T_inv is None:
            points_cano = (T_inv[..., :3, :3] @ points[..., None]).squeeze(-1) + T_inv[..., :3, 3]
        else:
            points_cano = points

        points_cano = contract_to_unisphere(points_cano, self.radius, self.contraction_type)
        density = self.encoding_with_network(points_cano.reshape(-1, self.n_input_dims)).reshape(*points_cano.shape[:-1], self.n_output_dims)[...,0]
        if 'density_activation' in self.config:
            density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        return -density      
    
    def update_step(self, epoch, global_step):
        update_module_step(self.encoding_with_network, epoch, global_step)


@models.register('volume-sdf')
class VolumeSDF(BaseImplicitGeometry):
    def setup(self):
        self.n_output_dims = self.config.feature_dim
        encoding = get_encoding(3, self.config.xyz_encoding_config)
        network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding, self.network = encoding, network
        self.grad_type = self.config.grad_type
    
    def forward(self, points, T_inv = None, featOnehot = None,  with_grad=True, with_feature=True):
        with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
            with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
                # if with_grad and self.grad_type == 'analytic':
                if with_grad:
                    if not self.training:
                        points = points.clone()
                        if not T_inv is None:
                            T_inv = T_inv.clone()
                    points.requires_grad_(True)

                if not T_inv is None:
                    points_cano = (T_inv[..., :3, :3] @ points[..., None]).squeeze(-1) + T_inv[..., :3, 3]
                else:
                    points_cano = points
                # points_cano_ = points_cano
                # points_ = points
                points_cano = contract_to_unisphere(points_cano, self.radius, self.contraction_type) # points normalized to (0, 1)
                out = self.network(self.encoding(points_cano.view(-1, 3), feat = featOnehot)).view(*points.shape[:-1], self.n_output_dims).float()

                sdf, feature = out[...,0], out
                if 'sdf_activation' in self.config:
                    sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                if 'feature_activation' in self.config:
                    feature = get_activation(self.config.feature_activation)(feature)               
                if with_grad:
                    if self.grad_type == 'analytic':
                        grad = torch.autograd.grad(
                            sdf, points, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]

                        # grad_cano = torch.autograd.grad(
                        #     sdf, points_cano, grad_outputs=torch.ones_like(sdf),
                        #     create_graph=True, retain_graph=True, only_inputs=True
                        # )[0]
                        # grad_cano = ( T_inv[..., :3, :3] @ grad[..., None]).squeeze(-1)
                        # sdf_grad = (T_fw[..., :3, :3] @ sdf_grad_cano[..., None]).squeeze(-1)
                    elif self.grad_type == 'finite_difference':
                        # This part is not working now
                        # But it could be helpful as demonstrated in paper "https://research.nvidia.com/labs/dir/neuralangelo"
                        pass
                        # eps = 0.0005
                        # if not T_inv is None:
                        #     points_cano = (T_inv[..., :3, :3] @ points_[..., None]).squeeze(-1) + T_inv[..., :3, 3]
                        # else:
                        #     points_cano = points_
                        # points_d_cano = torch.stack([
                        #     points_cano + torch.as_tensor([eps, 0.0, 0.0]).to(points_cano),
                        #     points_cano + torch.as_tensor([-eps, 0.0, 0.0]).to(points_cano),
                        #     points_cano + torch.as_tensor([0.0, eps, 0.0]).to(points_cano),
                        #     points_cano + torch.as_tensor([0.0, -eps, 0.0]).to(points_cano),
                        #     points_cano + torch.as_tensor([0.0, 0.0, eps]).to(points_cano),
                        #     points_cano + torch.as_tensor([0.0, 0.0, -eps]).to(points_cano)
                        # ], dim=0).clamp(-self.radius, self.radius)
                        # points_d_cano = scale_anything(points_d_cano, (-self.radius, self.radius), (0, 1))
                        # points_d_sdf = self.network(self.encoding(points_d_cano.view(-1, 3)))[..., 0].view(6,*points.shape[:-1]).float()
                        # grad_cano = torch.stack([0.5 * (points_d_sdf[0] - points_d_sdf[1]) / eps, 0.5 * (points_d_sdf[2] - points_d_sdf[3]) / eps, 0.5 * (points_d_sdf[4] - points_d_sdf[5]) / eps, ], dim=-1)
                        # grad = grad_cano

        rv = [sdf]
        if with_grad:
            rv.append(grad)
            # rv.append(grad_cano)
        if with_feature:
            rv.append(feature)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv
    
    def forward_level(self, points, T_inv=None, featOnehot = None):
        if not T_inv is None:
            points = (T_inv[..., :3, :3] @ points[..., None]).squeeze(-1) + T_inv[..., :3, 3]

        points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
        if featOnehot is None:
            sdf = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]
        else:
            sdf = self.network(self.encoding(points.view(-1, 3), feat=featOnehot)).view(*points.shape[:-1],self.n_output_dims)[...,0]

        if 'sdf_activation' in self.config:
            sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        if not T_inv is None:
            pass
        return sdf        
    
    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)    
        update_module_step(self.network, epoch, global_step)


@models.register('volume-sdf-uv')
class VolumeSDF(BaseImplicitGeometry):
    def setup(self):
        self.n_output_dims = self.config.feature_dim
        encoding = get_encoding(3, self.config.xyz_encoding_config)
        network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding, self.network = encoding, network
        self.grad_type = self.config.grad_type

    def forward(self, points, T_inv=None, with_grad=True, with_feature=True):
        with torch.inference_mode(
                torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
            with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
                # if with_grad and self.grad_type == 'analytic':
                if with_grad:
                    if not self.training:
                        points = points.clone()
                        if not T_inv is None:
                            T_inv = T_inv.clone()
                    points.requires_grad_(True)

                if not T_inv is None:
                    points_cano = (T_inv[..., :3, :3] @ points[..., None]).squeeze(-1) + T_inv[..., :3, 3]
                else:
                    points_cano = points
                points_cano_ = points_cano
                points_ = points
                points_cano = contract_to_unisphere(points_cano, self.radius,
                                                    self.contraction_type)  # points normalized to (0, 1)
                out = self.network(self.encoding(points_cano.view(-1, 3))).view(*points.shape[:-1],
                                                                                self.n_output_dims).float()
                sdf, feature = out[..., 0], out
                if 'sdf_activation' in self.config:
                    sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                if 'feature_activation' in self.config:
                    feature = get_activation(self.config.feature_activation)(feature)
                if with_grad:
                    if self.grad_type == 'analytic':
                        grad = torch.autograd.grad(
                            sdf, points, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]

                        grad_cano = torch.autograd.grad(
                            sdf, points_cano, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]

                    elif self.grad_type == 'finite_difference':
                        # This part is not working now
                        # But it could be helpful as demonstrated in paper "https://research.nvidia.com/labs/dir/neuralangelo"

                        pass
                        # eps = 0.001
                        # eps = 0.003
                        # points_d_ = torch.stack([
                        #     points_ + torch.as_tensor([eps, 0.0, 0.0]).to(points_),
                        #     points_ + torch.as_tensor([-eps, 0.0, 0.0]).to(points_),
                        #     points_ + torch.as_tensor([0.0, eps, 0.0]).to(points_),
                        #     points_ + torch.as_tensor([0.0, -eps, 0.0]).to(points_),
                        #     points_ + torch.as_tensor([0.0, 0.0, eps]).to(points_),
                        #     points_ + torch.as_tensor([0.0, 0.0, -eps]).to(points_)
                        # ], dim=0).clamp(0, 1)
                        #
                        # if not T_inv is None:
                        #     points_d_cano = (T_inv[..., :3, :3] @ points_d_[..., None]).squeeze(-1) + T_inv[..., :3, 3]
                        # else:
                        #     points_d_cano = points_d_
                        #
                        # points_d_cano = scale_anything(points_d_cano, (-self.radius, self.radius), (0, 1))
                        # points_d_sdf = self.network(self.encoding(points_d_cano.view(-1, 3)))[...,0].view(6, *points.shape[:-1]).float()
                        # grad_cano = torch.stack([
                        #     0.5 * (points_d_sdf[0] - points_d_sdf[1]) / eps,
                        #     0.5 * (points_d_sdf[2] - points_d_sdf[3]) / eps,
                        #     0.5 * (points_d_sdf[4] - points_d_sdf[5]) / eps,
                        # ], dim=-1)
                        #
                        # if not T_inv is None:
                        #     grad = ( torch.inverse(T_inv[..., :3, :3]) @ grad_cano[..., None]).squeeze(-1)
                        # else:
                        #     grad = grad_cano
                        # pass

        rv = [sdf]
        if with_grad:
            rv.append(grad)
            rv.append(grad_cano)
        if with_feature:
            rv.append(feature)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points, T_inv=None):
        if not T_inv is None:
            points = (T_inv[..., :3, :3] @ points[..., None]).squeeze(-1) + T_inv[..., :3, 3]

        points = contract_to_unisphere(points, self.radius, self.contraction_type)  # points normalized to (0, 1)
        sdf = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[..., 0]
        if 'sdf_activation' in self.config:
            sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        if not T_inv is None:
            pass
        return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)
        update_module_step(self.network, epoch, global_step)
