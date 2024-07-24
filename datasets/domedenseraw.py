"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-05-17
"""
# Load system modules
import os
import os.path as osp
import pdb
import sys
import warnings
warnings.filterwarnings("ignore")

# Setup Python OpenGL platform
os.environ['PYOPENGL_PLATFORM'] = 'egl' #use for visibility map computation

# Load basic modules for array, image, mesh
import cv2, trimesh
import json, math, glob, time, datasets, random
import numpy as np
from tqdm import tqdm
from PIL import Image


# Load PyTorch and Lightning for deep learning
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl

# Load libraries for 3D and related operations
from pytorch3d import ops
from pytorch3d.structures import Meshes
from kaolin.metrics.trianglemesh import point_to_mesh_distance # from kaolin.ops.mesh import uniform_laplacian
from tools.pyrender_tools import set_vertex_colors
from models.ray_utils import get_ray_directions, get_ray_directions_batch, unproject_points, get_rays, transform_rays_w2s, transform_rays_w2s_image, get_rays_image
from models.utils import _unbatched_point_to_mesh_interp
from tools.utils.misc import get_rank
from tools.extensions.libmesh.inside_mesh import check_mesh_contains
from tools.extensions.implicit_waterproofing import implicit_waterproofing
from tools.mesh_tools import save_ply, uniform_laplacian, compute_normal_torch
from tools.rotation_tools import fromTransformation2VectorTorch, fromVector2TransformationTorch

## Load template model
from tools.smplx import SMPL
from tools.skel_tools import load_model, load_smpl_param, load_ddc_param

## Load some useful tools
from tools.pyrender_tools import get_extrinc_from_sphere, Renderer
from tools.omni_tools import resizeImg
from tools.config_tools import load_config_with_default
from tools.cam_tools import load_camera_param


class DomeDenseRawDatasetBase():
    def setup(self, config, split):
        """
        param:
          config: Configuration object for dataset
          split: The split of the dataset (train, val, test)

        return:
          None
        """
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True
        self.initialized = False


        root = self.config.dataset.data_dir
        self.motion_dir = osp.join(root, 'motions')
        os.makedirs(self.config.save_dir, exist_ok=True)
        print("+ split:", self.split)
        print("+ data_dir:", root)
        print("+ save_dir:", self.config.save_dir)
        print()

        self.T_inv_world = None
        self.smpl_name = self.config.dataset.smpl_name #  "smpl_params_smplx.npz"
        self.ddc_name = self.config.dataset.ddc_name #  "smpl_params_smplx.npz"
        self.smpl_gender = self.config.dataset.smpl_gender
        self.smpl_dir = self.config.dataset.smpl_dir
        self.k = 1
        self.threshold_smpl = self.config.dataset.threshold_smpl
        self.threshold_ddc = self.config.dataset.threshold_ddc
        self.threshold_rigid = self.config.dataset.threshold_rigid
        self.threshold_outer = self.config.dataset.threshold_outer
        if self.config.dataset.deformer == 'smpl':
            self.threshold = self.threshold_smpl #0.1
        self.frame = self.config.dataset.frame
        self.camera_scale = self.config.dataset.camera_scale
        self.train_num_rays = self.config.model.max_train_num_rays

        if self.config.dataset.rotate and self.split=='test':
            # self.img_lists = sorted(glob.glob( osp.join(self.config.dataset.rotate_dir, "*.png")) )
            smpl_param_path = f"{self.motion_dir}/{self.smpl_name}"
            smpl_params = load_smpl_param(smpl_param_path, returnTensor=False, frames=[self.frame])

            H = W = self.config.dataset.rotate_w
            K_ = np.eye(3)
            K_[0, 0] = K_[1, 1] = 960 * H / 540
            K_[0, 2] = W / 2
            K_[1, 2] = H / 2

            Eall, Kall = [], []
            elevations = self.config.dataset.rotate_ele
            rotate_step = self.config.dataset.rotate_step
            rotate_shift = self.config.dataset.rotate_shift

            azimuths = []
            for idx in range(len(elevations)):
                azimuths += [ (np.arange(0, 360, len(elevations) * rotate_step) - rotate_step * idx).tolist()]

            for i in range(len(elevations)):
                for j in range(len(azimuths[i])):
                    if self.config.dataset.rotate_template:
                        E_ = get_extrinc_from_sphere(distance=4, elevation=elevations[i], azimuth=azimuths[i][j] * self.config.dataset.rotate_scale - rotate_shift)  # E c2w
                    else:
                        E_ = get_extrinc_from_sphere(distance=4, elevation=elevations[i], azimuth=azimuths[i][j] * self.config.dataset.rotate_scale - rotate_shift, t_shift=smpl_params["transl"][0] - [0, 0.2, 0])  # E c2w
                        # E_ = get_extrinc_from_sphere(distance=4, elevation=elevations[i], azimuth=azimuths[i][j] * self.config.dataset.rotate_scale - rotate_shift, t_shift=[2.73866439, 0.99124293, -0.05214555])  # E c2w

                    Eall.append(E_)
                    Kall.append(K_)

            Es = np.stack(Eall)
            Ks = np.stack(Kall)
            np.savez(osp.join(self.config.save_dir, 'testCam360.npz'), Ks = Ks, Esc2w = Es, Esw2c = np.linalg.inv(Es), H = H, W = W)

            Ec2w = torch.Tensor(Es)
            Ew2c = torch.inverse(Ec2w)
            Ks = torch.Tensor(Ks)
            self.img_lists = np.arange(0,Ks.shape[0]).tolist()
        else:
            # pdb.set_trace()
            self.img_lists = sorted(glob.glob(f"{root}/recon_neus2/imgs/{str(self.frame).zfill(6)}/*.png"))
            self.cam_path = sorted(glob.glob(f"{root}/recon_neus2/imgs/{str(self.frame).zfill(6)}/*.json"))[0]
            if self.config.dataset.with_depth and self.split=='train':
                self.dep_lists = sorted(glob.glob(f"{root}/recon_neus2/depths/{str(self.frame).zfill(6)}/*.png"))
            else:
                self.dep_lists = []

            Ks, Es, H, W = load_camera_param(self.cam_path, self.camera_scale)  # T, C, 4, 4
            Ks = Ks[:,:3,:3]
            Ew2c = torch.Tensor(Es)
            Ec2w = torch.inverse(Ew2c)
            Ks = torch.Tensor(Ks)

            if not self.config.dataset.active_camera_all:
                self.active_camera = list(self.config.dataset.active_camera)
            else:
                self.active_camera = []

            if not split=='test':
                if split=='val' and (self.config.trainer.val_check_interval==10 or self.config.trainer.val_check_interval==1):
                    self.active_camera_test = self.config.dataset.active_camera_test
                    if len(self.active_camera_test) > 0:
                        self.img_lists = [self.img_lists[cIdx] for cIdx in self.active_camera_test]
                        Ks = Ks[self.active_camera_test]
                        Ew2c = Ew2c[self.active_camera_test]
                        Ec2w = Ec2w[self.active_camera_test]
                else:
                    if len(self.active_camera) > 0:
                        self.img_lists = [self.img_lists[cIdx] for cIdx in self.active_camera]
                        if self.config.dataset.with_depth and self.split == 'train':
                            self.dep_lists = [self.dep_lists[cIdx] for cIdx in self.active_camera]
                        Ks = Ks[self.active_camera]
                        Ew2c = Ew2c[self.active_camera]
                        Ec2w = Ec2w[self.active_camera]
            elif split=='test':
                self.active_camera_test = self.config.dataset.active_camera_test
                if len(self.active_camera_test) > 0:
                    self.img_lists = [self.img_lists[cIdx] for cIdx in self.active_camera_test]
                    Ks = Ks[self.active_camera_test]
                    Ew2c = Ew2c[self.active_camera_test]
                    Ec2w = Ec2w[self.active_camera_test]

        Ks[:,  0, 0] *= 1 / self.config.dataset.img_downscale
        Ks[:,  1, 1] *= 1 / self.config.dataset.img_downscale
        Ks[:,  0, 2] *= 1 / self.config.dataset.img_downscale
        Ks[:,  1, 2] *= 1 / self.config.dataset.img_downscale
        self.Ks = Ks #[:,:,:3,:3]
        self.Ew2c = Ew2c
        self.Ec2w = Ec2w

        if 'img_wh' in self.config.dataset:
            w, h = self.config.dataset.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config.dataset:
            w, h = W // self.config.dataset.img_downscale, H // self.config.dataset.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")

        self.w, self.h, self.img_wh = w, h, (w, h)
        self.frameNum = len(self.img_lists)

        if self.config.dataset.deformer =='smpl':
            smpl_param_path = f"{self.motion_dir}/{self.smpl_name}"
            print("+ deformer:", self.config.dataset.deformer)
            print("+ smpl_param_path:", smpl_param_path)
            print()

            smpl_params = load_smpl_param(smpl_param_path, returnTensor=True, frames=[self.frame])
            self.body_model = SMPL(self.smpl_dir, gender=self.smpl_gender, batch_size = 1)
            self.faces = self.body_model.faces.astype(np.int64)
            self.faces_subdiv = self.faces.copy()
            self.smplroot = torch.Tensor([[-0.0022, -0.2408,  0.0286]])
            dist = torch.bmm(Ew2c[:,:3,:3], (smpl_params["transl"][:,:,None] + self.smplroot[:,:,None] ).repeat(Ew2c.shape[0],1,1) ) + Ew2c[:,:3,3:] - self.smplroot[:,:,None]
            dist = torch.linalg.norm(dist[:,:,0], ord = 2, dim = 1, keepdims= True )
            self.near = dist - 2
            self.far = dist + 2

            self.prepare_deformer_smpl(smpl_params)

        elif self.config.dataset.deformer =='ddc':
            ddc_param_path = f"{self.motion_dir}/{self.ddc_name}"
            print("+ deformer:", self.config.dataset.deformer)
            print("+ ddc_param_path:", ddc_param_path)
            print()

            ddc_params = load_ddc_param(ddc_param_path, returnTensor=True, frames= self.frame)
            self.cfgs = load_config_with_default(default_path=self.config.dataset.default_path, path=self.config.dataset.config_path, log=False)
            self.eg = load_model(self.cfgs, useCuda=False, device=None)
            self.faces = self.eg.character.faces
            # self.faces_subdiv = self.faces.copy()

            self.featuresInit = F.one_hot(torch.arange(0, 7))
            labels = np.array( self.eg.character.vertexLabels ).astype(np.int64)
            uq_labels = list(set(labels))
            labelsNew = labels.copy()
            for idx in range(len(uq_labels)):
                labelsNew[np.where(labelsNew == uq_labels[idx])] = idx
            self.labels = torch.Tensor(labels).to(torch.long)
            self.labelsNew = torch.Tensor(labelsNew).to(torch.long)
            self.deformLabels = (self.labels == 5) | (self.labels == 9)
            self.featOnehot = self.featuresInit[self.labelsNew]

            # pdb.set_trace()
            self.ddcroot =  torch.Tensor([[0.0, 0.4, 0.069]])
            dist = torch.bmm(Ew2c[:,:3,:3], (ddc_params["motion"][:,:3,None] + self.ddcroot[:,:,None] ).repeat(Ew2c.shape[0],1,1)  ) + Ew2c[:,:3,3:] - self.ddcroot[:,:,None]
            dist = torch.linalg.norm(dist[:,:,0], ord = 2, dim = 1, keepdims= True )
            self.near = dist - 2
            self.far = dist + 2

            self.prepare_deformer_ddc(ddc_params)


        if self.config.dataset.compute_occlusion and self.split=='train':
            camTemplatePath = osp.join(self.config.dataset.occlusion_template_dir, 'testCam360.npz')
            datazTemp = dict(np.load(camTemplatePath))
            print("+ load occlusion_template from:",camTemplatePath)
            print()
            if 'H' in datazTemp.keys():
                h_template, w_template = int(datazTemp['H']), int(datazTemp['W'])
            else:
                h_template, w_template = 540, 540
            self.h_template, self.w_template = h_template, w_template
            Ks_template = datazTemp['Ks']
            Es_template = datazTemp['Esc2w']
            Ec2w_template = torch.Tensor(Es_template)
            Ew2c_template = torch.inverse(Ec2w_template)
            Ks_template = torch.Tensor(Ks_template)

            directions_template = get_ray_directions_batch(w_template, h_template, Ks_template[:1]) # 1, H, W, 3 # Fixed a stupid bug here, orz
            rays_o, rays_d = get_rays_image(directions_template, Ec2w_template, keepdim=False)

            os.makedirs(osp.join(self.config.save_dir, 'occ_imgs'),exist_ok=True)
            render = Renderer(height=h_template, width=w_template, camera_type=1, use_ground=False, use_axis=False)
            render.add_mesh(self.template_occlusion_mesh)

            mask3Ls = []
            mask2Ls = []
            maskLs = []
            imgs_template = []
            kernel = np.ones((3,3), dtype=np.uint8)
            for idx in range(len(Ec2w_template)):
                img_temp_read = cv2.imread(osp.join(self.config.dataset.occlusion_template_dir, 'mask_imgs', '{}.png'.format(str(idx).zfill(3))), -1)
                # if self.config.dataset.img_downscale != 1.0:
                #     img_temp_read = resizeImg(img_temp_read, scale = 1 / self.config.dataset.img_downscale)
                img_temp = img_temp_read[:, :, :3][:, :, ::-1]
                img_temp = (img_temp[..., :3] / 255).astype(np.float32)
                mask_temp = img_temp_read[:, :, 3]
                # img_temp = np.where(np.stack([mask_temp,mask_temp,mask_temp], axis = -1), img_temp, 0.0)
                imgs_template.append(img_temp)
                K_ = Ks_template[0]
                # K_[:2,:] /= downscale
                E_ = Ec2w_template[idx].data.cpu().numpy()
                # E_[:3,:] /= downscale
                render.set_camera(K_, E_)
                color, depth = render.render(albedo=True)
                color = color[:, :, :3]
                mask = np.where(depth>0, 255,0).astype('uint8')
                mask2 = np.where(color.sum(2) == 0, 255, 0).astype('uint8')
                mask2 = cv2.dilate(mask2,kernel,1)
                mask2Ls.append(mask2)
                maskLs.append(mask)
                mask3Ls.append(mask_temp)

                cv2.imwrite(osp.join(self.config.save_dir , 'occ_imgs', '%03d.png'%(idx)), color[:,:,::-1])
                cv2.imwrite(osp.join(self.config.save_dir, 'occ_imgs', '%03d_input.png' % (idx)), (img_temp * 255.0).astype('uint8')[:,:,::-1] )
            mask3Ls = torch.Tensor(np.stack(mask3Ls, 0)) / 255.0
            mask2Ls = torch.Tensor(np.stack(mask2Ls, 0)) / 255.0
            maskLs = torch.Tensor(np.stack(maskLs, 0)) / 255.0
            imgs_template = torch.Tensor(np.stack(imgs_template, 0))

            self.rays_template = torch.cat([rays_o, rays_d, torch.zeros(rays_o.shape[0], 1), torch.ones(rays_o.shape[0], 1) * 10.0], 1)[mask2Ls.reshape(-1)>0]#.to(self.rank)
            self.all_images_template = imgs_template.reshape(-1, 3)[mask2Ls.reshape(-1) > 0].float()#.to(self.rank)
            self.all_fg_masks_template = mask3Ls.reshape(-1)[mask2Ls.reshape(-1) > 0].float()#.to(self.rank)


        if split=='val':
            if split == 'val' and (self.config.trainer.val_check_interval == 10 or self.config.trainer.val_check_interval==1):
                pass
            else:
                self.img_lists = self.img_lists[:1]
                Ks= Ks[:1]
                Ec2w = Ec2w[:1]
                self.w2s = self.w2s[:1]
                self.near = self.near[:1]
                self.far = self.far[:1]
                self.T_inv = self.T_inv[:1]
        elif split=='test':
            self.img_lists = self.img_lists[::self.config.dataset.test_interval]
            Ks = Ks[::self.config.dataset.test_interval]
            Ec2w = Ec2w[::self.config.dataset.test_interval]
            self.w2s = self.w2s[::self.config.dataset.test_interval]
            self.near = self.near[::self.config.dataset.test_interval]
            self.far = self.far[::self.config.dataset.test_interval]
            self.T_inv = self.T_inv[::self.config.dataset.test_interval]

        if self.config.export.save_mesh:
            self.prepare_cano_pts()
            print("+ save cano mesh: {}".format(True))
        else:
            print("+ save cano mesh: {}".format(False))

        if not split == 'train':
            if self.config.export.save_mesh:
                self.prepare_world_pts()
                print("+ save world mesh: {}".format(True))
            else:
                print("+ save world mesh: {}".format(False))
        else:
            print("+ save world mesh: {}".format(False))
        if self.config.model.grid_pre:
            self.prepare_occ_pts()
            print("+ pre-compute occupancy gird: {}".format(True))
        else:
            print("+ pre-compute occupancy gird: {}".format(False))
        print()

        self.directions = get_ray_directions_batch(w, h, Ks)#.to(self.rank)
        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
        kernel_size = 64
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.maskList = []
        self.maskXList = []
        self.maskYList = []
        self.maskNumList = []
        kernel_size2 = 3
        kernel2 = np.ones((kernel_size2, kernel_size2), np.uint8)

        if self.config.dataset.with_depth and self.split == 'train':
            self.all_depths = []
        for idx, frame in enumerate(range(len(Ec2w))):
            c2w = Ec2w[idx, :3, :4]
            self.all_c2w.append(c2w)
            if self.config.dataset.rotate and self.split == 'test':
                # temp_img = resizeImg(temp_img, h = self.h, w = self.w)
                temp_img = np.zeros([self.h, self.w, 4])
                temp_img = np.ones_like(temp_img) * 255
            else:
                temp_img = cv2.imread(self.img_lists[idx], -1 )

            if self.config.dataset.img_downscale != 1.0:
                temp_img = resizeImg(temp_img, scale=1 / self.config.dataset.img_downscale)

            os.makedirs(osp.join(self.config.save_dir, 'input_{}'.format(self.split) ), exist_ok=True)
            cv2.imwrite(osp.join(self.config.save_dir, 'input_{}'.format(self.split),  '%03d.jpg' % (idx)), temp_img)
            img = temp_img[:,:,:3][:,:,::-1]
            msk = temp_img[:,:, 3] / 255
            img = (img[..., :3] / 255).astype(np.float32)
            msk = msk.astype(np.float32)
            if self.config.dataset.erode_mask:
                msk = cv2.erode(msk, kernel2)

            msk_d = cv2.dilate(msk, kernel)
            self.maskList.append(msk_d)
            xx, yy = np.where(msk_d)
            self.maskXList.append(xx)
            self.maskYList.append(yy)
            self.maskNumList.append(len(xx))

            if self.config.dataset.with_depth and self.split=='train':
                temp_depth = cv2.imread(self.dep_lists[idx], -1 )[:,:,::-1]
                temp_depth = temp_depth / 255.0 * 10.0

            img = torch.Tensor(img)
            msk = torch.Tensor(msk)

            self.all_fg_masks.append(msk) # (h, w)
            self.all_images.append(img)
            if self.config.dataset.with_depth and self.split == 'train':
                self.all_depths.append(torch.Tensor(temp_depth))

        self.maskList = np.stack(self.maskList, 0 )
        self.maskXList = np.array(self.maskXList)
        self.maskYList = np.array(self.maskYList)
        self.maskNumList = np.array(self.maskNumList)
        maxTemp = np.max(self.maskNumList)
        for idx in range(self.maskXList.shape[0]):
            self.maskXList[idx] = np.concatenate([self.maskXList[idx], np.ones(  maxTemp - len(self.maskXList[idx]) )] , 0)
            self.maskYList[idx] = np.concatenate([self.maskYList[idx], np.ones(  maxTemp - len(self.maskYList[idx]) )] , 0)
        self.maskXList = np.stack(self.maskXList, 0)
        self.maskYList = np.stack(self.maskYList, 0)

        self.w2s = self.w2s#.to(self.rank)
        self.near = self.near#.to(self.rank)
        self.far = self.far#.to(self.rank)
        self.T_inv = self.T_inv.to(self.rank)
        self.vertices = self.vertices.to(self.rank)
        self.vs_template = self.vs_template.to(self.rank)
        self.face_vertices = self.face_vertices.to(self.rank)
        self.face_T_inv_flat = self.face_T_inv_flat.to(self.rank)
        if not self.T_inv_world is None:
            self.T_inv_world = self.T_inv_world.to(self.rank)

        if self.config.dataset.rotate_template and split =='test':
            T_inv_temp = torch.eye(4).repeat(1,self.T_inv.shape[1],1,1)
            self.T_inv = T_inv_temp.to(self.rank)
            # self.T_inv_world = self.T_inv.clone()

            w2s_temp = torch.eye(4) #.to(self.rank)
            self.w2s = w2s_temp.reshape(*self.w2s.shape)

            self.vertices = self.vs_template.clone()
            self.face_vertices = self.vertices[0, self.facesTorch].to(self.rank)
            self.face_T_inv_flat = self.T_inv[0, self.facesTorch].reshape(self.face_vertices.shape[0], self.face_vertices.shape[1], -1).to(self.rank)

        # self.all_c2w, self.all_images, self.all_fg_masks = \
        #     torch.stack(self.all_c2w, dim=0).float(), \ #.to(self.rank), \
        #     torch.stack(self.all_images, dim=0).float(), \#.to(self.rank), \
        #     torch.stack(self.all_fg_masks, dim=0).float() #.to(self.rank)

        self.all_c2w = torch.stack(self.all_c2w, dim=0).float()#.to(self.rank)
        self.all_images = torch.stack(self.all_images, dim=0).float()#.to(self.rank)
        self.all_fg_masks = torch.stack(self.all_fg_masks, dim=0).float()#.to(self.rank)

        self.rotate_template = self.config.dataset.rotate_template

        if self.config.dataset.with_depth and self.split == 'train':
            self.all_depths = torch.stack(self.all_depths, dim=0).float() #.to(self.rank)


    def get_bbox_from_smpl(self, vs, factor=1.2):
        """
        This function calculates the bounding box of a 3D model based on its vertices.
        param:
            vs: tensor of shape (1, N, 3) representing the vertices coordinates of the 3D model.
            factor: scaling factor to adjust the size of the bounding box. Default value is 1.2.
        return:
             tensor of shape (2, 3) representing the minimum and maximum coordinates of the bounding box.
        """
        assert vs.shape[0] == 1
        min_vert = vs.min(dim=1).values
        max_vert = vs.max(dim=1).values

        c = (max_vert + min_vert) / 2
        s = (max_vert - min_vert) / 2
        s = s.max(dim=-1).values * factor

        min_vert = c - s[:, None]
        max_vert = c + s[:, None]
        return torch.cat([min_vert, max_vert], dim=0)


    def initialize_smpl(self, smpl_params, device, batch = None):
        """
        This function initializes the SMPL model by converting it to the canonical space and setting the necessary variables for further calculations.
        param:
            smpl_params: A dictionary containing the SMPL parameters including "betas" and "body_pose" arrays.
            device: The device on which the calculations will be performed.
            batch: The number of batches for the calculations. If not provided, it will be determined from the size of the "betas" array in smpl_params.
        return:
            None
        """
        # convert to canonical space: deformed => T pose => Template pose
        if not batch is None:
            batch_size = batch
        else:
            batch_size = smpl_params["betas"].shape[0]
        if self.config.dataset.cano_motion == 'template':
            body_pose_t = torch.zeros((batch_size, 69), device=device)
            body_pose_t[:, 2] = torch.pi / 6
            body_pose_t[:, 5] = -torch.pi / 6
        elif self.config.dataset.cano_motion == 'world':
            body_pose_t = smpl_params["body_pose"]
        smpl_outputs = self.body_model(betas=smpl_params["betas"], body_pose=body_pose_t)
        self.bbox = self.get_bbox_from_smpl(smpl_outputs.vertices[0:1].detach())
        self.T_template = smpl_outputs.T
        self.vs_template = smpl_outputs.vertices[:1]
        self.pose_offset_t = smpl_outputs.pose_offsets[:1]
        self.shape_offset_t = smpl_outputs.shape_offsets[:1]

    def prepare_deformer_smpl(self, smpl_params):
        """
        This function prepares the deformer for SMPL meshes. It takes in SMPL parameters, initializes the body model
        if it is not already initialized, and applies the necessary transformations to generate the deformed mesh.

        param:
             smpl_params: Dictionary containing SMPL parameters including betas, body_pose, global_orient, and transl.
        return:
              None
        """
        device = smpl_params["betas"].device
        if next(self.body_model.parameters()).device != device:
            self.body_model = self.body_model.to(device)
            self.body_model.eval()

        if not self.initialized:
            self.initialize_smpl(smpl_params, device, smpl_params["body_pose"].shape[0])
            self.initialized = True

        smpl_outputs = self.body_model(betas=smpl_params["betas"],
                                       body_pose=smpl_params["body_pose"],
                                       global_orient=smpl_params["global_orient"],
                                       transl=smpl_params["transl"])
        s2w = smpl_outputs.A[:, 0]
        w2s = torch.inverse(s2w)

        T_inv = torch.inverse(smpl_outputs.T.float()).clone() @ s2w[:, None]
        T_inv[..., :3, 3] += self.pose_offset_t - smpl_outputs.pose_offsets
        T_inv[..., :3, 3] += self.shape_offset_t - smpl_outputs.shape_offsets
        T_inv = self.T_template @ T_inv
        self.T_inv = T_inv.detach()

        self.vertices = (smpl_outputs.vertices @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[:, None, :3, 3]
        self.vertices = self.vertices[:1]
        self.w2s = w2s

        if self.config.dataset.subdiv>0:
            mesh_pytorch3d = Meshes(verts=[self.vertices[0]], faces=[torch.Tensor(self.faces)])
            mesh_pytorch3d_subdiv, T_inv_subdiv = ops.SubdivideMeshes()(meshes=mesh_pytorch3d, feats=self.T_inv.reshape(-1, 16))
            mesh_pytorch3d_subdiv, T_inv_subdiv = ops.SubdivideMeshes()(meshes=mesh_pytorch3d_subdiv, feats=T_inv_subdiv.reshape(-1, 16))
            mesh_pytorch3d_subdiv, T_inv_subdiv = ops.SubdivideMeshes()(meshes=mesh_pytorch3d_subdiv, feats=T_inv_subdiv.reshape(-1, 16))
            self.vertices = mesh_pytorch3d_subdiv.verts_list()[0][None]
            self.faces_subdiv = mesh_pytorch3d_subdiv.faces_list()[0].data.cpu().numpy()
            self.T_inv = T_inv_subdiv.reshape(-1,4,4)[None]

        self.vertices_world = torch.matmul(s2w[None, :, :3, :3], self.vertices.clone().reshape(1, -1, 3, 1) * 1.0)[:, :, :, 0] + s2w[None, :, :3, 3]
        self.facesTorch = torch.Tensor(self.faces).to(torch.long)
        self.face_vertices = self.vertices[0, self.facesTorch].to(self.rank)
        self.face_T_inv_flat = self.T_inv[0, self.facesTorch].reshape(self.face_vertices.shape[0], self.face_vertices.shape[1], -1).to(self.rank)

        save_ply(osp.join(self.config.save_dir, 'debug_fk_template.ply'), self.vertices.data.cpu().numpy().reshape(-1, 3), self.faces_subdiv)
        save_ply(osp.join(self.config.save_dir, 'debug_template.ply'), self.vs_template.data.cpu().numpy().reshape(-1, 3), self.faces)
        save_ply(osp.join(self.config.save_dir, 'debug_fk_template_world.ply'), self.vertices_world.data.cpu().numpy().reshape(-1, 3), self.faces_subdiv)
        np.savez(osp.join(self.config.save_dir, 'T_w2s.npz'), T_w2s=w2s.data.cpu().numpy().reshape(4, 4))


    def initialize_ddc(self, ddc_params, device, batch = None):
        """
        This function initializes the DDC template by converting the input motion to the canonical space and calculating the template pose.

        param:
            ddc_params: A dictionary containing the necessary parameters for initializing DDC.
            device: The device on which the calculations will be performed.
            batch: The batch size (default = None).
        return:
            None
        """
        if not batch is None:
            batch_size = batch
        else:
            batch_size = 1

        if self.config.dataset.cano_motion =='template':
            motion_init = self.eg.character.motion_base.clone()
            motion_init[:, :6] = 0.0
            skinRs, skinTs = self.eg.updateNode(motion_init)
            self.vs_template, self.T_template, Tdeform_ = self.eg.forwardF(skinRs=skinRs, skinTs=skinTs, returnTransformation=True)

        elif self.config.dataset.cano_motion =='world':
            motion_init = ddc_params["motion"].clone()
            motion_init[:, :6] = 0.0
            skinRs, skinTs = self.eg.updateNode(motion_init)
            # self.vs_template, self.T_template, Tdeform_ = self.eg.forwardF(skinRs = skinRs, skinTs = skinTs, returnTransformation=True)
            deltaRs, deltaTs = ddc_params["deltaR"], ddc_params["deltaT"]
            self.vs_template, self.T_template, Tdeform_ = self.eg.forwardF(deltaRs=deltaRs, deltaTs=deltaTs * 1000.0, skinRs = skinRs, skinTs = skinTs,
                                                      displacements = ddc_params["displacement"] * 1000.0, returnTransformation=True)
        self.T_template = self.T_template @ Tdeform_
        self.vs_template /= 1000.0
        self.bbox = self.get_bbox_from_smpl(self.vs_template.detach())
        self.s2w_init = self.eg.character.jointTransformations[:, 0]

    def prepare_deformer_ddc(self, ddc_params):
        """
        This function prepares the deformer for DDC meshes. It takes in DDC parameters, initializes the body model if
        it is not already initialized, and applies the necessary transformations to generate the deformed mesh.

        param:
            ddc_params: dictionary containing the parameters for preparing the deformer DDC.
        return:
            None
        """
        device = ddc_params["motion"].device
        if self.eg.device != device:
            self.eg.device = device

        if not self.initialized:
            self.initialize_ddc(ddc_params, device, ddc_params["motion"].shape[0])
            self.initialized = True

        skinRs, skinTs = self.eg.updateNode(ddc_params["motion"])
        deltaRs, deltaTs = ddc_params["deltaR"], ddc_params["deltaT"]
        # vertsF, Tskin, Tdeform = self.eg.forwardF(deltaRs=deltaRs, deltaTs=deltaTs * 1000.0, skinRs = skinRs, skinTs = skinTs, returnTransformation=True)
        vertsF, Tskin, Tdeform = self.eg.forwardF(deltaRs=deltaRs, deltaTs=deltaTs * 1000.0, skinRs = skinRs, skinTs = skinTs,
                                                  displacements = ddc_params["displacement"] * 1000.0, returnTransformation=True)

        s2w = self.eg.character.jointTransformations[:,0]
        s2w = s2w @ torch.inverse(self.s2w_init)
        w2s = torch.inverse(s2w)
        self.w2s = w2s

        self.Tskin_inv = torch.inverse(Tskin) @ s2w[:, None]
        self.Tdeform_inv = torch.inverse(Tdeform)

        laplacian_matrix = uniform_laplacian(vertsF.shape[1], torch.Tensor(self.eg.character.faces).to(torch.long), self.eg.character.labelMatrix)
        Tfull =  w2s[:, None] @ Tskin @ Tdeform

        if self.config.dataset.smoothDQ:
            Tfull_DQ = fromTransformation2VectorTorch(Tfull.reshape(-1, 4, 4))
            Tfull_DQ += torch.matmul(laplacian_matrix, Tfull_DQ) * 0.3
            Tfull_DQ += torch.matmul(laplacian_matrix, Tfull_DQ) * 0.3
            Tfull_DQ += torch.matmul(laplacian_matrix, Tfull_DQ) * 0.3
            Tfull4 = fromVector2TransformationTorch(Tfull_DQ).reshape(-1, 4, 4)
        else:
            Tfull4 = Tfull.reshape(-1, 4, 4)

        self.T_inv = self.T_template @ torch.inverse(Tfull4)[None]

        T_temp = torch.inverse(self.T_inv)
        self.vertices = torch.matmul(T_temp[:, :, :3, :3], self.vs_template.reshape(1, -1, 3, 1) * 1000.0)[:, :, :, 0] + T_temp[:, :, :3, 3]
        self.vertices_world = torch.matmul(s2w[None, :, :3, :3], self.vertices.clone().reshape(1, -1, 3, 1) * 1.0)[:, :, :, 0] + s2w[None, :, :3, 3]
        self.vs_template_world = (torch.matmul(s2w[None, :, :3, :3], self.vs_template.clone().reshape(1, -1, 3, 1) * 1000.0)[:, :, :, 0] + s2w[None, :, :3, 3]) / 1000.0
        np.savez(osp.join(self.config.save_dir, 'T_w2s.npz'), T_w2s = w2s.data.cpu().numpy().reshape(4, 4))

        self.vertices /= 1000.0
        self.vertices_old = self.vertices.clone()
        self.vertices_world /= 1000.0
        self.T_inv[:,:,:3,3] /= 1000.0
        self.w2s[:,:3,3] /= 1000.0

        self.facesTorch = torch.Tensor(self.faces).to(torch.long)
        temp = self.labelsNew[self.facesTorch]
        labelsFace = torch.mode(temp)[0]
        self.face_featOnehot = self.featuresInit[labelsFace].to(self.rank)
        self.face_vertices = self.vertices[0, self.facesTorch].to(self.rank)
        self.face_T_inv_flat = self.T_inv[0, self.facesTorch].reshape(self.face_vertices.shape[0], self.face_vertices.shape[1], -1).to(self.rank)

        if self.config.dataset.subdiv>0:
            mesh_pytorch3d = Meshes(verts=[self.vertices[0]], faces=[torch.Tensor(self.faces)])
            mesh_pytorch3d_subdiv, T_inv_subdiv = ops.SubdivideMeshes()(meshes=mesh_pytorch3d, feats=self.T_inv.reshape(-1, 16))
            mesh_pytorch3d_subdiv, T_inv_subdiv = ops.SubdivideMeshes()(meshes=mesh_pytorch3d_subdiv, feats=T_inv_subdiv.reshape(-1, 16))
            mesh_pytorch3d_subdiv, T_inv_subdiv = ops.SubdivideMeshes()(meshes=mesh_pytorch3d_subdiv, feats=T_inv_subdiv.reshape(-1, 16))
            self.vertices = mesh_pytorch3d_subdiv.verts_list()[0][None]
            self.faces_subdiv = mesh_pytorch3d_subdiv.faces_list()[0].data.cpu().numpy()
            self.T_inv = T_inv_subdiv.reshape(-1,4,4)[None]
        else:
            self.faces_subdiv = self.faces

        if self.config.dataset.compute_occlusion and self.split=='train':
            downscale = 1
            render = Renderer(height=self.h//downscale, width=self.w//downscale, camera_type=1, use_ground=False, use_axis=False)
            mesh = trimesh.Trimesh(vertices = self.vertices_world.data.cpu().numpy().reshape(-1,3), faces = self.faces_subdiv)
            render.add_mesh(mesh)
            pointUnprojList = []
            rayoList = []
            for idx in range(len(self.Ec2w)):
                K_ = self.Ks[idx].data.cpu().numpy()
                K_[:2,:] /= downscale
                E_ = self.Ec2w[idx].data.cpu().numpy()
                render.set_camera(K_, E_)
                color, depth = render.render()
                color = color[:, :, :3]
                mask = np.where(depth>0,255,0)
                # render.del_mesh()
                pointsUnproj = unproject_points(depth, mask, H = self.h//downscale, W = self.w//downscale, K= K_, E = E_)
                save_ply(osp.join(self.config.save_dir, 'debug_unproj_{}.ply'.format(idx)), pointsUnproj)
                pointUnprojList.append(pointsUnproj)
                rayoList.append(np.tile(E_[:3,3:].reshape(-1,3),(pointsUnproj.shape[0],1)))
                cv2.imwrite(osp.join(self.config.save_dir , '%03d.png'%(idx)),color[:,:,::-1])

            pointUnprojList = torch.Tensor(np.concatenate(pointUnprojList, 0))[None]
            rayoList = torch.Tensor(np.concatenate(rayoList, 0))[None]
            rayList = (pointUnprojList - rayoList)[0]
            rayList /= torch.linalg.norm(rayList, ord=2, dim=1, keepdims = True)
            normal_world  = compute_normal_torch(self.vertices_world, self.faces)[0]

            K = 1
            threshold = 0.004
            with torch.no_grad():
                sdf, idx, neighbors = ops.knn_points(self.vertices_world, pointUnprojList, K=K)
                valid = sdf < threshold ** 2
                valid = valid[:,:,0].reshape(-1)

            valid_temp = torch.where(valid, 0.0, 1.0).reshape(-1,1)
            valid_temp += torch.matmul(laplacian_matrix, valid_temp)
            valid_temp += torch.matmul(laplacian_matrix, valid_temp)

            valid_final = (valid_temp==0).reshape(-1)
            valid_hand = (torch.Tensor(self.eg.character.vertexLabels)==14)
            # valid_final = torch.where( torch.logical_and(valid_hand, valid_final), True, valid_final )
            valid_final = torch.where( valid_hand, True, valid_final )

            colors = np.zeros([valid_final.shape[0],3], 'uint8')
            colors = np.where(np.stack([valid_final.data.cpu().numpy()>0, valid_final.data.cpu().numpy()>0, valid_final.data.cpu().numpy()>0], axis = -1),[255,0,0], colors)

            if self.split=='train':
                save_ply(osp.join(  self.config.save_dir , 'debug_fk_template.ply'  ), self.vertices.data.cpu().numpy().reshape(-1,3) , self.faces_subdiv, colors = colors)
                save_ply(osp.join(  self.config.save_dir , 'debug_template.ply'  ), self.vs_template.data.cpu().numpy().reshape(-1,3) , self.faces, colors = colors)
                save_ply(osp.join(  self.config.save_dir , 'debug_fk_template_world.ply'  ), self.vertices_world.data.cpu().numpy().reshape(-1,3) , self.faces_subdiv, colors = colors)
            self.template_occlusion_mesh = trimesh.Trimesh(vertices = self.vs_template.data.cpu().numpy().reshape(-1,3), faces = self.faces)
            set_vertex_colors(self.template_occlusion_mesh, colors )
            self.occlusion_rate = (1 - valid_final.sum() / valid_final.shape[0])
            self.train_num_rays_template = int(self.train_num_rays * self.occlusion_rate)

        elif self.config.dataset.compute_occlusion and self.split=='val':
            pass
        else:
            if self.split == 'train':
                save_ply(osp.join(self.config.save_dir, 'debug_fk_template.ply'),
                         self.vertices.data.cpu().numpy().reshape(-1, 3), self.faces_subdiv)
                save_ply(osp.join(self.config.save_dir, 'debug_template.ply'),
                         self.vs_template.data.cpu().numpy().reshape(-1, 3), self.faces)
                save_ply(osp.join(self.config.save_dir, 'debug_fk_template_world.ply'),
                         self.vertices_world.data.cpu().numpy().reshape(-1,3), self.faces_subdiv)
                save_ply(osp.join(self.config.save_dir, 'debug_template_world.ply'),
                         self.vs_template_world.data.cpu().numpy().reshape(-1,3), self.faces)

    def prepare_cano_pts(self):
        """
        Prepare canonical points for the template.

        return:
            None
        """
        cano_vertices = self.vs_template[0].to(self.rank)
        min_xyz = torch.min(cano_vertices, axis = 0)[0]
        max_xyz = torch.max(cano_vertices, axis = 0)[0]
        min_xyz[:2] -= 0.05
        max_xyz[:2] += 0.05
        min_xyz[2] -= 0.15
        max_xyz[2] += 0.15
        self.cano_bounds = torch.stack([min_xyz, max_xyz], axis = 0).to(torch.float32).to(self.rank)
        print('+ Canonical volume len: {}'.format(self.cano_bounds[1] - self.cano_bounds[0]))

        self.resolutions = torch.Tensor(self.config.model.geometry.isosurface.resolution).reshape(-1,3).to(self.rank)
        x_coords = torch.linspace(0, 1, steps = self.config.model.geometry.isosurface.resolution[0], dtype = torch.float32, device = self.rank).detach()
        y_coords = torch.linspace(0, 1, steps = self.config.model.geometry.isosurface.resolution[1], dtype = torch.float32, device = self.rank).detach()
        z_coords = torch.linspace(0, 1, steps = self.config.model.geometry.isosurface.resolution[2], dtype = torch.float32, device = self.rank).detach()
        xv, yv, zv = torch.meshgrid(x_coords, y_coords, z_coords)  # print(xv.shape) # (256, 256, 256)
        xv = torch.reshape(xv, (-1, 1))  # print(xv.shape) # (256*256*256, 1)
        yv = torch.reshape(yv, (-1, 1))
        zv = torch.reshape(zv, (-1, 1))
        pts = torch.cat([xv, yv, zv], dim = -1)

        pts = pts * (self.cano_bounds[1] - self.cano_bounds[0]) + self.cano_bounds[0]
        if self.config.dataset.deformer == 'smpl':
            threshold = self.threshold_smpl #0.1
        else:
            threshold = self.threshold_ddc

        with torch.no_grad():
            self.sdf, idx, neighbors = ops.knn_points(pts[None], cano_vertices[None], K=1)

        print('+ Start check contains..')
        st = time.time()
        cano_smpl_trimesh = trimesh.Trimesh(cano_vertices.data.cpu().numpy(), self.faces)
        if not self.config.dataset.loose:
            self.validInOut = check_mesh_contains(cano_smpl_trimesh, pts.data.cpu().numpy())[0]
        else:
            self.validInOut = implicit_waterproofing(cano_smpl_trimesh, pts.data.cpu().numpy())[0]
        self.validInOut = torch.Tensor(self.validInOut).reshape(-1) > 0
        print('+ Time for check contains: ', time.time() - st)

        self.sdf = self.sdf.reshape(-1)
        self.sdf[self.validInOut.to(self.rank)] *= -1

        threshold_outer = self.threshold_outer
        valid0 = (self.sdf > 0) & (self.sdf < threshold_outer ** 2)
        valid1 = (self.sdf < 0) & (self.sdf > - threshold ** 2)
        valid = valid0 | valid1
        self.valid = valid.reshape(-1)
        print('+ Valid cano point number: '.format(valid.sum()))
        print()

        del pts
        # from skimage.measure import marching_cubes
        # verts, faces, normals, values = marching_cubes(self.sdf.data.cpu().numpy().reshape(self.config.model.geometry.isosurface.resolution), 0)
        # save_ply(r'./template_sdf.ply', verts, faces)

    def prepare_world_pts(self):
        """
        Prepare the world points for the template.

        return:
            None
        """
        world_vertices = self.vertices[0].to(self.rank)
        min_xyz = torch.min(world_vertices, axis=0)[0]
        max_xyz = torch.max(world_vertices, axis=0)[0]
        min_xyz[:2] -= 0.05
        max_xyz[:2] += 0.05
        min_xyz[2] -= 0.15
        max_xyz[2] += 0.15
        self.world_bounds = torch.stack([min_xyz, max_xyz], axis=0).to(torch.float32).to(self.rank)
        print('+ World volume len: {}'.format(self.cano_bounds[1] - self.cano_bounds[0]))

        self.resolutions_world = torch.Tensor(self.config.model.geometry.isosurface.resolution).reshape(-1, 3).to(self.rank)
        x_coords = torch.linspace(0, 1, steps=self.config.model.geometry.isosurface.resolution[0],
                                  dtype=torch.float32, device=self.rank).detach()
        y_coords = torch.linspace(0, 1, steps=self.config.model.geometry.isosurface.resolution[1],
                                  dtype=torch.float32, device=self.rank).detach()
        z_coords = torch.linspace(0, 1, steps=self.config.model.geometry.isosurface.resolution[2],
                                  dtype=torch.float32, device=self.rank).detach()
        xv, yv, zv = torch.meshgrid(x_coords, y_coords, z_coords)
        xv = torch.reshape(xv, (-1, 1))
        yv = torch.reshape(yv, (-1, 1))
        zv = torch.reshape(zv, (-1, 1))
        pts = torch.cat([xv, yv, zv], dim=-1)

        # pts = pts * torch.from_numpy(self.cano_bounds[1] - self.cano_bounds[0]).to(pts) + torch.from_numpy(self.cano_bounds[0]).to(pts)
        pts = pts * (self.world_bounds[1] - self.world_bounds[0]) + self.world_bounds[0]

        if self.config.dataset.deformer == 'smpl':
            threshold = self.threshold_smpl#0.1
        else:
            threshold = self.threshold_ddc

        print('+ Start check contains..')
        st = time.time()
        world_smpl_trimesh = trimesh.Trimesh(world_vertices.data.cpu().numpy(), self.faces_subdiv)
        if not self.config.dataset.loose:
            self.validInOut_world = check_mesh_contains(world_smpl_trimesh, pts.data.cpu().numpy())[0]
        else:
            self.validInOut_world = implicit_waterproofing(world_smpl_trimesh, pts.data.cpu().numpy())[0]
        self.validInOut_world = torch.Tensor(self.validInOut_world).reshape(-1) > 0
        print('+ Time for check contains: ', time.time() - st)

        with torch.no_grad():
            sdf, idx, neighbors = ops.knn_points(pts[None], world_vertices[None], K=1)
            sdf = sdf.reshape(-1)
            sdf[self.validInOut_world.to(self.rank)] *= -1

            threshold_outer = self.threshold_outer
            valid0 = (sdf > 0) & (sdf < threshold_outer ** 2)
            valid1 = (sdf < 0) & (sdf > - threshold ** 2)
            valid = valid0 | valid1
            # self.valid = valid.reshape(-1)
            self.valid_world = valid.reshape(-1)
            print('+ Valid world point number: '.format(valid.sum()))
            print()

            self.infer_pts = pts.float()[self.valid_world].float()
            dist_, face_, type_ = point_to_mesh_distance(self.infer_pts[None], self.face_vertices[None])
            T_inv, _ = _unbatched_point_to_mesh_interp(self.infer_pts, face_[0], type_[0], self.face_vertices, self.face_T_inv_flat)

        self.T_inv_world = T_inv.reshape(-1, 4, 4)
        if self.config.model.geometry.xyz_encoding_config.include_feat:
            self.featOnehot_world = self.face_featOnehot[face_[0]]
        else:
            self.featOnehot_world = None

        del pts
        del self.infer_pts

    def prepare_occ_pts(self):
        #TODO: handle additional canonical template to support occlusion handling
        """
        Prepare the world points for the template.

        return:
            None
        """
        world_vertices = self.vertices[0].to(self.rank)
        occ_bounds = torch.as_tensor([-self.config.model.radius, -self.config.model.radius, -self.config.model.radius,
                                            self.config.model.radius, self.config.model.radius, self.config.model.radius], \
                                            dtype=torch.float32).to(self.rank).reshape(2,3)
        print('+ Occ volume len: {}'.format(occ_bounds[1] - occ_bounds[0]))


        x_coords = torch.linspace(0, 1, steps=self.config.model.grid_resolution,
                                  dtype=torch.float32, device=self.rank).detach()
        y_coords = torch.linspace(0, 1, steps=self.config.model.grid_resolution,
                                  dtype=torch.float32, device=self.rank).detach()
        z_coords = torch.linspace(0, 1, steps=self.config.model.grid_resolution,
                                  dtype=torch.float32, device=self.rank).detach()
        xv, yv, zv = torch.meshgrid(x_coords, y_coords, z_coords)
        xv = torch.reshape(xv, (-1, 1))
        yv = torch.reshape(yv, (-1, 1))
        zv = torch.reshape(zv, (-1, 1))
        pts = torch.cat([xv, yv, zv], dim=-1)

        pts = pts * (occ_bounds[1] - occ_bounds[0]) + occ_bounds[0]

        if self.config.dataset.deformer == 'smpl':
            threshold = self.threshold_smpl#0.1
        else:
            threshold = self.threshold_ddc

        with torch.no_grad():
            sdf, idx, neighbors = ops.knn_points(pts[None], world_vertices[None], K=1)
            valid = sdf < threshold ** 2

            self.valid_occ = valid.reshape(-1)
            print('+ Valid occ point number: '.format(valid.sum()))
            print()
        del pts

class DomeDenseRawDataset(Dataset, DomeDenseRawDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        # print(index)
        # print(index.shape)
        # exit()
        c2w = self.all_c2w[[index]]
        if self.directions.ndim == 3:  # (H, W, 3)
            directions = self.directions
        elif self.directions.ndim == 4:  # (N, H, W, 3)
            directions = self.directions[index].reshape(-1, 3)
        rays_o, rays_d = get_rays(directions, c2w, keepdim=False)

        rgb = self.all_images[[index]].view(-1, self.all_images.shape[-1])
        fg_mask = self.all_fg_masks[[index]].view(-1)

        w2s = self.w2s[:]
        rays_o, rays_d = transform_rays_w2s(rays_o, rays_d, w2s)

        near = self.near[[index]]
        far = self.far[[index]]
        near = near.repeat(rays_o.shape[0], 1)
        far = far.repeat(rays_o.shape[0], 1)
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1), near, far], dim=-1)

        background_color = torch.ones((3,), dtype=torch.float32)

        if self.apply_mask:
            rgb = rgb * fg_mask[..., None] + background_color * (1 - fg_mask[..., None])

        return {
            'index': index,
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask,
            'background_color': background_color,
        }


class DomeDenseRawIterableDataset(IterableDataset, DomeDenseRawDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:

            index = torch.randint(0, len(self.all_images), size=(self.train_num_rays,))
            c2w = self.all_c2w[index]
            x = torch.randint(0, self.w, size=(self.train_num_rays,))
            y = torch.randint(0, self.h, size=(self.train_num_rays,))
            if self.directions.ndim == 3: # (H, W, 3)
                directions = self.directions[y, x]
            elif self.directions.ndim == 4: # (N, H, W, 3)
                directions = self.directions[index, y, x]

            rgb = self.all_images[index, y, x].view(-1, self.all_images.shape[-1]) #.to(self.rank)
            fg_mask = self.all_fg_masks[index, y, x].view(-1) #.to(self.rank)
            rays_o, rays_d = get_rays(directions, c2w, keepdim=False)

            w2s = self.w2s[:]
            rays_o, rays_d = transform_rays_w2s(rays_o, rays_d, w2s)
            # w2s = self.w2s[:]
            # rays_o, rays_d = transform_rays_w2s(rays_o, rays_d, self.w2s)

            if self.config.dataset.with_depth:
                near = self.all_depths[index, y,x, 0:1] + self.dataset.depth_shift
                far = self.all_depths[index, y, x, 1:2] - self.dataset.depth_shift
            else:
                near = self.near[index]
                far = self.far[index]

            rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1), near, far], dim=-1)

            if self.config.dataset.compute_occlusion:
                index_template = torch.randint(0, len(self.all_images_template), size=(self.train_num_rays_template,)) #, device=self.all_images_template.device)
                rgb_template = self.all_images_template[index_template]
                fg_mask_template = self.all_fg_masks_template[index_template]
                rays_template = self.rays_template[index_template]
            # else:
            #     rgb_template = None
            #     fg_mask_template = None
            #     rays_template = None

            if self.config.model.background_color == 'white':
                background_color = torch.ones((3,), dtype=torch.float32)
            elif self.config.model.background_color == 'random':
                background_color = torch.rand((3,), dtype=torch.float32) # affect a lot !
            else:
                raise NotImplementedError

            if self.apply_mask:
                rgb = rgb * fg_mask[..., None] + background_color * (1 - fg_mask[..., None])

                if self.config.dataset.compute_occlusion:
                    rgb_template = rgb_template * fg_mask_template[..., None] + background_color * (
                                1 - fg_mask_template[..., None])

            if self.config.dataset.compute_occlusion:
                yield {
                    'index': index,
                    'rays': rays,
                    'rgb': rgb,
                    'fg_mask': fg_mask,
                    'rgb_template': rgb_template,
                    'fg_mask_template': fg_mask_template,
                    'rays_template': rays_template,
                    'background_color': background_color,
                }
            else:
                yield {
                    'index': index,
                    'rays': rays,
                    'rgb': rgb,
                    'fg_mask': fg_mask,
                    'background_color': background_color,
                }
            # yield {}


@datasets.register('domedenseraw')
class DomeDenseRawDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = DomeDenseRawIterableDataset(self.config, self.config.dataset.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = DomeDenseRawDataset(self.config, self.config.dataset.val_split)
        if stage in [None, 'test']:
            self.test_dataset = DomeDenseRawDataset(self.config, self.config.dataset.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = DomeDenseRawDataset(self.config, self.config.dataset.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None

        def worker_init_fn(worker_id):  # set numpy's random seed
            # seed = torch.initial_seed()
            # seed = seed % (2 ** 32)
            seed = self.config.seed
            # np.random.seed(seed + worker_id)
            random.seed(seed + worker_id)
            np.random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)

        return DataLoader(
            dataset, 
            num_workers=os.cpu_count() //2,
            # num_workers=14,
            batch_size=batch_size,
            pin_memory=True,
            worker_init_fn=worker_init_fn if dataset.split == 'train' else None,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
