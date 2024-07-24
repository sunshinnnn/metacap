import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import chunk_batch, gradient, _unbatched_point_to_mesh_interp
from systems.utils import update_module_step
from tools.utils.mesh import save_ply
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, render_weight_from_alpha, accumulate_along_rays
from nerfacc.intersection import ray_aabb_intersect
from pytorch3d import ops
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from tools.extensions.libmesh.inside_mesh import check_mesh_contains
from tools.extensions.implicit_waterproofing import implicit_waterproofing
import trimesh
# from knn_cuda import KNN

class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)


@models.register('neus')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.geometry.contraction_type = ContractionType.AABB

        if self.config.learned_background:
            self.geometry_bg = models.make(self.config.geometry_bg.name, self.config.geometry_bg)
            self.texture_bg = models.make(self.config.texture_bg.name, self.config.texture_bg)
            self.geometry_bg.contraction_type = ContractionType.UN_BOUNDED_SPHERE
            self.near_plane_bg, self.far_plane_bg = 0.1, 1e3
            self.cone_angle_bg = 10**(math.log10(self.far_plane_bg) / self.config.num_samples_per_ray_bg) - 1.
            self.render_step_size_bg = 0.01            

        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius,
                                                            self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=self.config.grid_resolution,
                contraction_type=ContractionType.AABB
            )

            self.occupancy_grid_template = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=self.config.grid_resolution,
                contraction_type=ContractionType.AABB
            )

            if self.config.learned_background:
                self.occupancy_grid_bg = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=self.config.grid_resolution,
                    contraction_type=ContractionType.UN_BOUNDED_SPHERE
                )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray

        self.rotate_template = False
        self.train_meta = False
        self.initialized = False
        self.vertices = None
        self.vertices_template = None
        self.face_vertices = None
        self.face_featOnehot = None
        self.face_T_inv_flat= None
        self.T_inv = None
        self.deformer='ddc'
        self.threshold_smpl = 0.075
        self.threshold_ddc = 0.05
        self.threshold_rigid = 0.05
        self.threshold_outer = 0.05

        self.deformLabels = None
        self.faces = None
        self.validInOut_grid = None
        # self.knn = KNN(k=1, transpose_mode=True)

    
    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        if self.config.learned_background:
            update_module_step(self.geometry_bg, epoch, global_step)
            update_module_step(self.texture_bg, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
        
        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size_bg) based on taylor series
            return density[...,None] * self.render_step_size_bg

    def get_alpha(self, sdf, normal, dirs, dists):
        # so maybe here?
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def forward_bg_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, _ = self.geometry_bg(positions)
            return density[...,None]            

        _, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
        # if the ray intersects with the bounding box, start from the farther intersection point
        # otherwise start from self.far_plane_bg
        # note that in nerfacc t_max is set to 1e10 if there is no intersection
        near_plane = torch.where(t_max > 1e9, self.near_plane_bg, t_max)
        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=None,
                grid=self.occupancy_grid_bg if self.config.grid_prune else None,
                sigma_fn=sigma_fn,
                near_plane=near_plane, far_plane=self.far_plane_bg,
                render_step_size=self.render_step_size_bg,
                stratified=self.randomized,
                cone_angle=self.cone_angle_bg,
                alpha_thre=0.0
            )       
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints  
        intervals = t_ends - t_starts

        density, feature = self.geometry_bg(positions) 
        rgb = self.texture_bg(feature, t_dirs)

        weights = render_weight_from_density(t_starts, t_ends, density[...,None], ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)       

        out = {
            'comp_rgb': comp_rgb,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': intervals.view(-1),
                'ray_indices': ray_indices.view(-1)
            })

        return out

    # def forward_(self, rays, T_inv = None, vertices = None, stage = None, global_step = -1, rays_template = None):
    def forward_(self, rays, stage=None, global_step=-1, rays_template=None, background_color=None,):
        # TODO: refine here
        # print('\nstage: {}\n'.format(stage))
        # if stage == 'validation':
        #     import pdb
        #     pdb.set_trace()
        # print(T_inv.shape)
        if background_color is None:
            background_color = self.background_color
        # pdb.set_trace()

        n_rays = rays.shape[0]
        n_rays2 = rays.shape[0]
        if rays.shape[1] == 8:
            rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
            near, far = rays[:, 6], rays[:, 7]
        else:
            rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
            near, far = None, None

        def occ_eval_fn_smpl(x):
            shape = x.shape
            if self.deformer=='smpl':
                threshold = self.threshold_smpl #   0.075#0.1
            else:
                threshold = self.threshold_ddc #  0.05
            # threshold = 0.1
            # import pdb
            # pdb.set_trace()
            x = x.reshape(-1,3)[None]
            with torch.no_grad():
                dist_sq, idx, neighbors = ops.knn_points(x, self.vertices.float(), K=1)
                # dist_sq, idx = self.knn(x, self.vertices.float())
                valid = dist_sq < threshold ** 2

            alpha = torch.zeros([shape[0],1]).to(x.device)
            alpha[valid.reshape(-1)] = 1.0
            return alpha

        def occ_eval_fn_smpl_pre(x):
            shape = x.shape
            if self.deformer=='smpl':
                threshold = self.threshold_smpl #   0.075#0.1
            else:
                threshold = self.threshold_ddc #  0.05
            # threshold = 0.1
            x = x.reshape(-1,3)[None]
            with torch.no_grad():
                dist_sq, idx, neighbors = ops.knn_points(x, self.vertices.float(), K=1)
                valid = dist_sq < threshold ** 2

            alpha = torch.zeros([shape[0],1]).to(x.device)
            alpha[valid.reshape(-1)] = 1.0
            return alpha

        def occ_eval_fn_smpl_500(x):
            shape = x.shape
            if self.deformer=='smpl':
                threshold = self.threshold_smpl #   0.075#0.1
            else:
                threshold = self.threshold_ddc #  0.05
            x = x.reshape(-1,3)[None]
            with torch.no_grad():
                dist_sq, idx, neighbors = ops.knn_points(x, self.vertices.float(), K=1)
            cano_smpl_trimesh = trimesh.Trimesh( self.vertices.data.cpu().numpy().reshape(-1,3), self.faces)
            validInOut = self.validInOut_grid.to(self.rank)
            dist_sq = dist_sq.reshape(-1)
            dist_sq[validInOut.to(self.rank)] *= -1

            threshold_outer = self.threshold_outer / 2
            valid0 = (dist_sq > 0) & (dist_sq < threshold_outer ** 2)
            valid1 = (dist_sq < 0) & (dist_sq > - threshold ** 2)
            valid = valid0 | valid1
            alpha = torch.zeros([shape[0],1]).to(x.device)
            alpha[valid.reshape(-1)] = 1.0
            return alpha

        def occ_eval_fn_smpl_template(x):
            shape = x.shape
            if self.deformer=='smpl':
                threshold = self.threshold_smpl #   0.075#0.1
            else:
                threshold = self.threshold_ddc #  0.05
            x = x.reshape(-1,3)[None]
            with torch.no_grad():
                dist_sq, idx, neighbors = ops.knn_points(x, self.vertices_template.float().to(self.rank), K=1)
                valid = dist_sq < threshold ** 2
            alpha = torch.zeros([shape[0],1]).to(x.device)
            alpha[valid.reshape(-1)] = 1.0
            return alpha

        def occ_eval_fn_smpl_template_500(x):
            shape = x.shape
            if self.deformer=='smpl':
                threshold = self.threshold_smpl #   0.075#0.1
            else:
                threshold = self.threshold_ddc #  0.05
            x = x.reshape(-1,3)[None]
            with torch.no_grad():
                dist_sq, idx, neighbors = ops.knn_points(x, self.vertices_template.float(), K=1)
            cano_smpl_trimesh = trimesh.Trimesh( self.vertices_template.data.cpu().numpy().reshape(-1,3), self.faces)
            validInOut = implicit_waterproofing(cano_smpl_trimesh, x.data.cpu().numpy().reshape(-1,3) )[0]
            validInOut = torch.Tensor(validInOut).reshape(-1) > 0
            dist_sq = dist_sq.reshape(-1)
            dist_sq[validInOut.to(self.rank)] *= -1
            threshold_outer = self.threshold_outer / 2
            valid0 = (dist_sq > 0) & (dist_sq < threshold_outer ** 2)
            valid1 = (dist_sq < 0) & (dist_sq > - threshold ** 2)
            valid = valid0 | valid1

            alpha = torch.zeros([shape[0],1]).to(x.device)
            alpha[valid.reshape(-1)] = 1.0
            return alpha

        if not self.initialized or self.train_meta:
            if self.config.grid_pre:
                self.occupancy_grid._binary = self.validInOut_grid.reshape([self.config.grid_resolution,self.config.grid_resolution,self.config.grid_resolution])
                self.initialized = True
            else:
                if stage == 'train':
                    self.occupancy_grid._binary = torch.zeros_like(self.occupancy_grid._binary)
                    if global_step >= self.config.decay_step and self.train_meta:
                        self.occupancy_grid.every_n_step(step=1, occ_eval_fn=occ_eval_fn_smpl_500, occ_thre=self.config.get('grid_prune_occ_thre', 0.01), n=1)
                    else:
                        self.occupancy_grid.every_n_step(step=1, occ_eval_fn=occ_eval_fn_smpl, occ_thre=self.config.get('grid_prune_occ_thre', 0.01),n=1)
                    if not rays_template is None:
                        self.occupancy_grid_template._binary = torch.zeros_like(self.occupancy_grid_template._binary)
                        if global_step >= self.config.decay_step and self.train_meta:
                            # self.occupancy_grid_template.every_n_step(step=1, occ_eval_fn=occ_eval_fn_smpl_template_500,
                            #                                                       occ_thre=self.config.get('grid_prune_occ_thre', 0.01),
                            #                                                       n=1)
                            self.occupancy_grid_template.every_n_step(step=1, occ_eval_fn=occ_eval_fn_smpl_template,
                                                                                  occ_thre=self.config.get('grid_prune_occ_thre', 0.01),
                                                                                  n=1)
                        else:
                            self.occupancy_grid_template.every_n_step(step=1, occ_eval_fn=occ_eval_fn_smpl_template, occ_thre=self.config.get('grid_prune_occ_thre', 0.01), n=1)
                    self.initialized = True
                else:
                    self.occupancy_grid._binary = torch.zeros_like(self.occupancy_grid._binary)
                    self.occupancy_grid._update(step=1, occ_eval_fn=occ_eval_fn_smpl, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))
                    if not rays_template is None:
                        self.occupancy_grid_template._binary = torch.zeros_like(self.occupancy_grid_template._binary)
                        self.occupancy_grid_template._update(step=1, occ_eval_fn=occ_eval_fn_smpl_template, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))

                    # self.occupancy_grid_template2._update(step=1, occ_eval_fn=occ_eval_fn_smpl_template, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))
                    self.initialized = True
            # print(stage, ' ', global_step)
        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(  # N points, N depths=(t_starts + t_ends) / 2.
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=None,
                near_plane=near, far_plane=far,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts

        if not rays_template is None:
            rays_o_template, rays_d_template = rays_template[:, 0:3], rays_template[:, 3:6] # both (N_rays, 3)
            # near, far = None, None
            with torch.no_grad():
                ray_indices_template, t_starts_template, t_ends_template = ray_marching(
                    rays_o_template, rays_d_template,
                    scene_aabb=self.scene_aabb,
                    grid=self.occupancy_grid_template,
                    alpha_fn=None,
                    near_plane=None, far_plane=None,
                    render_step_size=self.render_step_size,
                    stratified=self.randomized,
                    cone_angle=0.0,
                    alpha_thre=0.0
                )
            ray_indices_template = ray_indices_template.long()
            t_origins_template = rays_o_template[ray_indices_template] #
            t_dirs_template = rays_d_template[ray_indices_template]
            midpoints_template = (t_starts_template + t_ends_template) / 2.
            positions_template = t_origins_template + t_dirs_template * midpoints_template
            dists_template = t_ends_template - t_starts_template
            n_rays_template = rays_template.shape[0]
        # pdb.set_trace()
        if not rays_template is None:
            if positions_template.shape[0] == 0:
                opacity_template = torch.zeros([n_rays_template, 1]).to(rays_o.device)
                comp_rgb_template = torch.zeros([n_rays_template, 3]).to(rays_o.device)
                if not self.training:
                    depth_template = torch.zeros([n_rays_template, 1]).to(rays_o.device)
                    comp_normal_template = torch.zeros([n_rays_template, 3]).to(rays_o.device)
                # sdf_grad_template = torch.zeros_like(positions_template)
                # sdf_cano_template = torch.zeros_like(opacity_template)
            else:
                T_inv_template = torch.eye(4).repeat(positions_template.shape[0], 1, 1).to(positions.device)
                t_dirs_cano_template = t_dirs_template.clone()
                # sdf_cano_template, sdf_grad_template, sdf_grad_cano_template, feature_cano_template = self.geometry(positions_template, T_inv_template, with_grad=True, with_feature=True)  # cano
                sdf_cano_template, sdf_grad_template, feature_cano_template = self.geometry(positions_template, T_inv_template, with_grad=True, with_feature=True)  # cano
                sdf_grad_cano_template = sdf_grad_template
                normal_cano_template = F.normalize(sdf_grad_cano_template, p=2, dim=-1)
                normal_template = F.normalize(sdf_grad_template, p=2, dim=-1)
                alpha_template = self.get_alpha(sdf_cano_template, normal_template, t_dirs_template, dists_template)[..., None]
                rgb_template = self.texture(feature_cano_template, t_dirs_cano_template, normal_cano_template)

                weights_template = render_weight_from_alpha(alpha_template, ray_indices=ray_indices_template, n_rays=n_rays_template)
                opacity_template = accumulate_along_rays(weights_template, ray_indices_template, values=None, n_rays=n_rays_template)
                comp_rgb_template = accumulate_along_rays(weights_template, ray_indices_template, values=rgb_template, n_rays=n_rays_template)
                if not self.training:
                    depth_template = accumulate_along_rays(weights_template, ray_indices_template, values=midpoints_template, n_rays=n_rays_template)
                    comp_normal_template = accumulate_along_rays(weights_template, ray_indices_template, values=normal_template, n_rays=n_rays_template)
                    comp_normal_template = F.normalize(comp_normal_template, p=2, dim=-1)

        # print(positions.shape)
        if positions.shape[0] == 0:
            opacity = torch.zeros([n_rays,1]).to(rays_o.device)
            comp_rgb = torch.zeros([n_rays, 3]).to(rays_o.device)
            if not self.training:
                depth = torch.zeros([n_rays, 1]).to(rays_o.device)
                comp_normal = torch.zeros([n_rays, 3]).to(rays_o.device)
            # pass
        else:
            if not self.face_vertices is None:
                with torch.no_grad():
                    dist_, face_, type_ = point_to_mesh_distance(positions.float()[None], self.face_vertices[None])
                    T_inv, _  = _unbatched_point_to_mesh_interp(positions.float(), face_[0], type_[0], self.face_vertices, self.face_T_inv_flat)
                    T_inv = T_inv.reshape(-1,4,4)
                    if not self.face_featOnehot is None and self.config.geometry.xyz_encoding_config.include_feat:
                        featOnehot = self.face_featOnehot[face_[0]]
                    else:
                        featOnehot = None
                    R_temp = torch.inverse( T_inv[..., :3, :3] ).transpose(2,1)
                    t_dirs_cano = ( R_temp @ t_dirs[..., None]).squeeze(-1)

            # sdf_cano, sdf_grad_cano, feature_cano = self.geometry(positions_cano, T_inv=None, with_grad=True, with_feature=True) # cano
            # sdf_cano, sdf_grad, sdf_grad_cano, feature_cano = self.geometry(positions, T_inv = T_inv, featOnehot = featOnehot,  with_grad=True, with_feature=True) # cano
            # sdf_cano, sdf_grad_cano, feature_cano = self.geometry(positions, T_inv = T_inv, featOnehot = featOnehot,  with_grad=True, with_feature=True) # cano
            # sdf_grad = (torch.inverse(T_inv[..., :3, :3]) @ sdf_grad_cano[..., None]).squeeze(-1)
            sdf_cano, sdf_grad, feature_cano = self.geometry(positions, T_inv = T_inv, featOnehot = featOnehot,  with_grad=True, with_feature=True) # cano
            sdf_grad_cano = (T_inv[..., :3, :3] @ sdf_grad[..., None]).squeeze(-1)
            normal_cano = F.normalize(sdf_grad_cano, p=2, dim=-1)
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            alpha = self.get_alpha(sdf_cano, normal, t_dirs, dists)[..., None]
            rgb = self.texture(feature_cano, t_dirs_cano, normal_cano)
            weights = render_weight_from_alpha(alpha[:t_starts.shape[0]], ray_indices=ray_indices, n_rays=n_rays2)
            opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays2)
            comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb[:t_starts.shape[0]], n_rays=n_rays2)
            if not self.training:
                depth = accumulate_along_rays(weights, ray_indices, values=midpoints[:t_starts.shape[0]], n_rays=n_rays2)
                comp_normal = accumulate_along_rays(weights, ray_indices, values=normal[:t_starts.shape[0]], n_rays=n_rays2)
                comp_normal = F.normalize(comp_normal, p=2, dim=-1)

        if not rays_template is None:
            out = {
                'comp_normal': comp_normal[:rays.shape[0]] if not self.training else None,
                'depth': depth[:rays.shape[0]] if not self.training else None,
                'comp_rgb': comp_rgb[:rays.shape[0]],
                'opacity': opacity[:rays.shape[0]],
                'rays_valid': opacity[:rays.shape[0]] > 0 ,
                'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device),

                'comp_normal_template': comp_normal_template if not self.training else None,
                'depth_template': depth_template if not self.training else None,
                'comp_rgb_template': comp_rgb_template,
                'opacity_template': opacity_template,
                'rays_valid_template': opacity_template > 0,
                'num_samples_template': torch.as_tensor([len(t_starts_template)], dtype=torch.int32, device=rays.device)
            }
        else:
            out = {
                'comp_normal': comp_normal if not self.training else None,
                'depth': depth if not self.training else None,
                'comp_rgb': comp_rgb,
                'opacity': opacity,
                'rays_valid': opacity > 0 ,
                'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
            }

        if self.training:
            if not rays_template is None:
                out.update({
                    'positions': positions,
                    'sdf_samples': sdf_cano[:],
                    'sdf_grad_samples': sdf_grad[:],
                    # 'sdf_grad_samples_cano': sdf_grad_cano[:],
                    'weights': weights.view(-1),
                    'points': midpoints.view(-1),
                    'intervals': dists.view(-1),
                    'ray_indices': ray_indices.view(-1),

                    'sdf_samples_template': sdf_cano_template,
                    'sdf_grad_samples_template': sdf_grad_template,
                    # 'sdf_grad_samples_cano_template': sdf_grad_cano_template,
                })
            else:
                out.update({
                    'positions': positions,
                    'sdf_samples': sdf_cano,
                    # 'sdf_samples': sdf,
                    # 'sdf_grad_samples': sdf_grad[rays.shape[0]:],
                    # 'sdf_grad_samples_cano': sdf_grad_cano[rays.shape[0]:],
                    'sdf_grad_samples': sdf_grad[:],
                    # 'sdf_grad_samples_cano': sdf_grad_cano[:],
                    'weights': weights.view(-1),
                    'points': midpoints.view(-1),
                    'intervals': dists.view(-1),
                    'ray_indices': ray_indices.view(-1)
                })

        if not rays_template is None:
            if self.config.learned_background:
                out_bg = self.forward_bg_(rays)
            else:
                out_bg = {
                    # 'comp_rgb': self.background_color[None, :].expand(*comp_rgb[:rays.shape[0]].shape),
                    'comp_rgb': background_color[None, :].expand(*comp_rgb[:rays.shape[0]].shape),
                    'num_samples': torch.zeros_like(out['num_samples']),
                    'rays_valid': torch.zeros_like(out['rays_valid']),

                    # 'comp_rgb_template': self.background_color[None, :].expand(*comp_rgb[rays.shape[0]:].shape),
                    # 'comp_rgb_template': self.background_color[None, :].expand(*comp_rgb_template.shape),
                    'comp_rgb_template': background_color[None, :].expand(*comp_rgb_template.shape),
                    'num_samples_template': torch.zeros_like(out['num_samples_template']),
                    'rays_valid_template': torch.zeros_like(out['rays_valid_template'])
                }

            out_full = {
                'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
                'num_samples': out['num_samples'] + out_bg['num_samples'],
                'rays_valid': out['rays_valid'] | out_bg['rays_valid'],

                'comp_rgb_template': out['comp_rgb_template'] + out_bg['comp_rgb_template'] * (1.0 - out['opacity_template']),
                'num_samples_template': out['num_samples_template'] + out_bg['num_samples_template'],
                'rays_valid_template': out['rays_valid_template'] | out_bg['rays_valid_template']
            }

        else:
            # pdb.set_trace()
            # print(background_color.shape)
            if self.config.learned_background:
                out_bg = self.forward_bg_(rays)
            else:
                out_bg = {
                    # 'comp_rgb': self.background_color[None,:].expand(*comp_rgb.shape),
                    'comp_rgb': background_color[None, :].expand(*comp_rgb.shape),
                    'num_samples': torch.zeros_like(out['num_samples']),
                    'rays_valid': torch.zeros_like(out['rays_valid'])
                }

            out_full = {
                'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
                'num_samples': out['num_samples'] + out_bg['num_samples'],
                'rays_valid': out['rays_valid'] | out_bg['rays_valid']
            }

        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
        }

    def forward_template(self, rays, background_color = None):
        n_rays = rays.shape[0]
        if rays.shape[1] == 8:
            rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
            near, far = rays[:, 6], rays[:, 7]
        else:
            rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
            near, far = None, None

        if background_color is None:
            background_color = self.background_color

        def occ_eval_fn_smpl_template(x):
            shape = x.shape
            if self.deformer=='smpl':
                threshold = self.threshold_smpl #   0.075#0.1
            else:
                threshold = self.threshold_ddc #  0.05

            x = x.reshape(-1, 3)[None]
            with torch.no_grad():
                dist_sq, idx, neighbors = ops.knn_points(x, self.vertices_template.float().to(self.rank), K=1)
                valid = dist_sq < threshold ** 2

            alpha = torch.zeros([shape[0], 1]).to(x.device)
            alpha[valid.reshape(-1)] = 1.0
            return alpha

        if not self.initialized or self.train_meta:
            # self.occupancy_grid_template = OccupancyGrid(
            #     roi_aabb=self.scene_aabb,
            #     # resolution=64,
            #     resolution=self.config.grid_resolution,
            #     # resolution=128, #improve psnr
            #     contraction_type=ContractionType.AABB
            # )
            self.occupancy_grid._update(step=1, occ_eval_fn=occ_eval_fn_smpl_template,
                                        occ_thre=self.config.get('grid_prune_occ_thre', 0.01))
            self.initialized = True

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid,
                alpha_fn=None,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts
        T_inv = self.T_inv[0,:1].repeat(positions.shape[0],1,1)
        R_temp = torch.inverse(T_inv[..., :3, :3]).transpose(2, 1)
        t_dirs_cano = (R_temp @ t_dirs[..., None]).squeeze(-1)
        # sdf_cano, sdf_grad, sdf_grad_cano, feature_cano = self.geometry(positions, T_inv, with_grad=True, with_feature=True)  # cano
        sdf_cano, sdf_grad, feature_cano = self.geometry(positions, T_inv, with_grad=True, with_feature=True)  # cano
        sdf_grad_cano = (T_inv[..., :3, :3] @ sdf_grad[..., None]).squeeze(-1)
        normal_cano = F.normalize(sdf_grad_cano, p=2, dim=-1)
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf_cano, normal, t_dirs, dists)[..., None]
        rgb = self.texture(feature_cano, t_dirs_cano, normal_cano)
        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)

        if not self.training:
            depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
            comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
            comp_normal = F.normalize(comp_normal, p=2, dim=-1)

        out = {
            'comp_normal': comp_normal if not self.training else None,
            'depth': depth if not self.training else None,

            'comp_rgb': comp_rgb,
            'opacity': opacity,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }
        if self.config.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                # 'comp_rgb': self.background_color[None,:].expand(*comp_rgb.shape),
                'comp_rgb': background_color[None, :].expand(*comp_rgb.shape),
                'num_samples': torch.zeros_like(out['num_samples']),
                'rays_valid': torch.zeros_like(out['rays_valid'])
            }

        out_full = {
            'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
            'num_samples': out['num_samples'] + out_bg['num_samples'],
            'rays_valid': out['rays_valid'] | out_bg['rays_valid']
        }

        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
        }


    def forward(self, rays, T_inv = None, vertices = None, stage = None, global_step = -1, rays_template = None, background_color=None):
        # print(" global_step:  ", global_step)
        if self.training:
            out = self.forward_(rays, background_color=background_color, stage=stage, global_step = global_step, rays_template = rays_template)
        else:
            if not self.rotate_template:
                out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays, stage)
            else:
                out = chunk_batch(self.forward_template, self.config.ray_chunk, True, rays,)
        return {
            **out,
            'inv_s': self.variance.inv_s
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config, valid=None, featOnehot= None, bounds = None , validInOut = None, T_inv = None, meshPath = None):
        mesh, level = self.geometry.isosurface(valid = valid, featOnehot = featOnehot, bounds = bounds, validInOut = validInOut, T_inv = T_inv, meshPath = meshPath)
        if export_config.export_vertex_color:
            if not T_inv is None:
                with torch.no_grad():
                    dist_, face_, type_ = point_to_mesh_distance(mesh['v_pos'].to(self.rank).float()[None], self.face_vertices[None])
                    T_inv, _  = _unbatched_point_to_mesh_interp(mesh['v_pos'].to(self.rank).float(), face_[0], type_[0], self.face_vertices, self.face_T_inv_flat)
                    T_inv_temp = T_inv.reshape(-1,4,4)
                    if not self.face_featOnehot is None and self.config.geometry.xyz_encoding_config.include_feat:
                        featOnehot = self.face_featOnehot[face_[0]]
                    else:
                        featOnehot = None
                # _, sdf_grad, sdf_grad_cano, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), T_inv= T_inv_temp, featOnehot = featOnehot, with_grad=True, with_feature=True)
                # _, sdf_grad_cano, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), T_inv= T_inv_temp, featOnehot = featOnehot, with_grad=True, with_feature=True)
                _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), T_inv= T_inv_temp, featOnehot = featOnehot, with_grad=True, with_feature=True)
                sdf_grad_cano = (T_inv_temp[..., :3, :3] @ sdf_grad[..., None]).squeeze(-1)
            else:
                # _, sdf_grad, sdf_grad_cano, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), T_inv= None, featOnehot = featOnehot,  with_grad=True, with_feature=True)
                # _, sdf_grad_cano, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), T_inv= None, featOnehot = featOnehot,  with_grad=True, with_feature=True)
                _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), T_inv= None, featOnehot = featOnehot,  with_grad=True, with_feature=True)
                sdf_grad_cano = sdf_grad
            normal = F.normalize(sdf_grad_cano, p=2, dim=-1)
            rgb = self.texture(feature, -normal, normal) # set the viewing directions to the normal to get "albedo"
            mesh['v_rgb'] = rgb.cpu()
        return mesh, level
