"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-07-06
"""
# Load system modules
import os
import os.path as osp
import sys
import warnings
warnings.filterwarnings("ignore")
## Setup Python OpenGL platform
# os.environ['PYOPENGL_PLATFORM'] = 'egl' #use for visibility map computation

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
from models.ray_utils import get_ray_directions, get_ray_directions_batch, get_ray_directions_batch_naive
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


class DomeDenseRawSeqMetaDatasetBase():
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
        self.smpl_name = self.config.dataset.smpl_name  # "smpl_params_smplx.npz"
        self.ddc_name = self.config.dataset.ddc_name  # "smpl_params_smplx.npz"
        self.smpl_gender = self.config.dataset.smpl_gender
        self.smpl_dir = self.config.dataset.smpl_dir
        self.k = 1
        self.threshold_smpl = self.config.dataset.threshold_smpl
        self.threshold_ddc = self.config.dataset.threshold_ddc
        self.threshold_rigid = self.config.dataset.threshold_rigid
        self.threshold_outer = self.config.dataset.threshold_outer
        if self.config.dataset.deformer == 'smpl':
            self.threshold = self.threshold_smpl#0.1

        self.frames = list(self.config.dataset.frames)
        if isinstance(self.config.dataset.frames_path, str):
            fileList =os.listdir(self.config.dataset.frames_path)
            fileList = [ int(item.split('_')[0])  for item in fileList]
            fileList = sorted(fileList)
            random.shuffle(fileList)
            self.frames = fileList
        self.frameNum = len(self.frames)
        self.camera_scale = self.config.dataset.camera_scale
        print("+ Frame Number is: ", self.frameNum)
        print(self.frames)

        self.img_lists = []
        self.cam_path_lists = []
        self.dep_lists = []
        for frame in self.frames:
            self.img_lists.append( sorted(glob.glob(f"{root}/recon_neus2/imgs/{str(frame).zfill(6)}/*.png")) )
            self.cam_path_lists.append(sorted(glob.glob(f"{root}/recon_neus2/imgs/{str(frame).zfill(6)}/*.json"))[0])
            if self.config.dataset.with_depth and self.split=='train':
                self.dep_lists.append( sorted(glob.glob(f"{root}/recon_neus2/depths/{str(frame).zfill(6)}/*.png")) )

        self.motion_dir = osp.join(root, 'motions')
        self.img_lists = np.array(self.img_lists)
        if self.config.dataset.with_depth and self.split == 'train':
            self.dep_lists = np.array(self.dep_lists)

        Ks, Es, H, W = load_camera_param(self.cam_path_lists, self.camera_scale) # T, C, 4, 4
        Ew2c = torch.Tensor(Es)
        Ec2w = torch.inverse(Ew2c)

        if self.split == 'train':
            Ks[:, :, 0, 0] *= 1 / self.config.dataset.img_downscale
            Ks[:, :, 1, 1] *= 1 / self.config.dataset.img_downscale
            Ks[:, :, 0, 2] *= 1 / self.config.dataset.img_downscale
            Ks[:, :, 1, 2] *= 1 / self.config.dataset.img_downscale


        if not self.config.dataset.active_camera_all:
            self.active_camera = list(self.config.dataset.active_camera)
        else:
            self.active_camera = []

        if len(self.active_camera) >0 and not split=='test':
            img_lists_temp = []
            if self.config.dataset.with_depth and self.split == 'train':
                dep_lists_temp = []
            for idx in range(len(self.img_lists)):
                img_lists_temp.append( [ self.img_lists[idx][cIdx]   for cIdx in self.active_camera] )
                if self.config.dataset.with_depth and self.split == 'train':
                    dep_lists_temp.append([self.dep_lists[idx][cIdx] for cIdx in self.active_camera])

            self.img_lists = np.array(img_lists_temp)
            if self.config.dataset.with_depth and self.split == 'train':
                self.dep_lists = np.array(dep_lists_temp)

            Ks = Ks[:,self.active_camera]
            Ew2c = Ew2c[:,self.active_camera]
            Ec2w = Ec2w[:,self.active_camera]

        if self.config.dataset.deformer =='smpl':
            smpl_param_path = f"{self.motion_dir}/{self.smpl_name}"
            print("+ deformer:", self.config.dataset.deformer)
            print("+ smpl_param_path:", smpl_param_path)
            print()

            smpl_params = load_smpl_param(smpl_param_path, returnTensor=True, frames = self.frames)
            self.body_model = SMPL(self.smpl_dir, gender=self.smpl_gender, batch_size = len(self.frames))
            self.faces = self.body_model.faces.astype(np.int64)
            self.faces_subdiv = self.faces.copy()
            self.smplroot = torch.Tensor([[-0.0022, -0.2408,  0.0286]])
            dist = torch.matmul(Ew2c[:,:,:3,:3], (smpl_params["transl"] + self.smplroot )[:,None,:,None] ) + Ew2c[:, :, :3, 3:] - self.smplroot[None, :, :, None]
            dist = torch.linalg.norm(dist[:, :, :, 0], ord = 2, dim = 2, keepdims= True )
            self.near = dist - 2 # self.near = dist - 1    T,C,1
            self.far = dist + 2 # self.far = dist + 1

            self.prepare_deformer_smpl(smpl_params)

        elif self.config.dataset.deformer =='ddc':
            ddc_param_path = f"{self.motion_dir}/{self.ddc_name}"
            print("+ deformer:", self.config.dataset.deformer)
            print("+ ddc_param_path:", ddc_param_path)
            print()

            ddc_params = load_ddc_param(ddc_param_path, returnTensor=True, frames= self.frames)
            self.cfgs = load_config_with_default(default_path=self.config.dataset.default_path, path=self.config.dataset.config_path, log=False)
            self.egInit = load_model(self.cfgs, useCuda=False, device=None)
            self.eg = load_model(self.cfgs, useCuda=False, device=None)
            self.faces = self.eg.character.faces
            self.faces_subdiv = self.faces.copy()

            self.featuresInit = F.one_hot(torch.arange(0, 7))
            labels = np.array( self.eg.character.vertexLabels ).astype(np.int64)
            uq_labels = list(set(labels))
            labelsNew = labels.copy()
            for idx in range(len(uq_labels)):
                labelsNew[np.where(labelsNew == uq_labels[idx])] = idx
            self.labels = torch.Tensor(labels).to(torch.long)
            self.labelsNew = torch.Tensor(labelsNew).to(torch.long)
            self.deformLabels = (self.labels == 5) | (self.labels == 9)

            self.ddcroot =  torch.Tensor([[0.0, 0.4, 0.069]])
            dist = torch.matmul(Ew2c[:,:,:3,:3], (ddc_params["motion"][:,:3] + self.ddcroot )[:,None, :,None]  ) + Ew2c[:, :,:3,3:] - self.ddcroot[None, :, :, None]
            dist = torch.linalg.norm(dist[:, :, :, 0], ord = 2, dim = 2, keepdims= True )
            self.near = dist - 2 # self.near = dist - 1    T,C,1
            self.far = dist + 2 # self.far = dist + 1

            self.prepare_deformer_ddc(ddc_params)

        if split=='val':
            self.img_lists = self.img_lists[:1, :1]
            Ks= Ks[:1, :1]
            Ec2w = Ec2w[:1, :1]
            self.w2s = self.w2s[:1]
            self.near = self.near[:1, :1]
            self.far = self.far[:1, :1]

        elif split=='test':
            self.img_lists = self.img_lists[:1, :4]
            Ks = Ks[:1, ::4]
            Ec2w = Ec2w[:1, ::4]
            self.w2s = self.w2s[:1]
            self.near = self.near[:1, ::4]
            self.far = self.far[:1, ::4]
            # self.T_inv = self.T_inv[:, ::4]

        if split == 'train':
            self.prepare_grid_pts()
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
        print()

        if 'img_wh' in self.config.dataset:
            w, h = self.config.dataset.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config.dataset:
            if self.split == 'train':
                w, h = int(W // self.config.dataset.img_downscale), int(H // self.config.dataset.img_downscale)
            else:
                w, h = W, H
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")

        self.w, self.h, self.img_wh = w, h, (w, h)
        self.directions  = get_ray_directions_batch_naive(w, h).to(self.rank)
        self.all_images, self.all_fg_masks = [], []
        self.all_depths = []
        self.all_c2w = Ec2w
        self.Ks = torch.Tensor(Ks).to(self.rank)

        if self.config.dataset.preload:
            for fIdx in range(Ec2w.shape[0]):
                all_images_temp, all_fg_masks_temp = [], []
                if self.config.dataset.with_depth and self.split == 'train':
                    all_depths_temp = []
                for cIdx in range(Ec2w.shape[1]):
                    print("{}/{}".format(fIdx,cIdx), flush=True)
                    temp_img = cv2.imread(self.img_lists[fIdx][cIdx], -1)
                    if self.config.dataset.img_downscale != 1.0 and self.split == 'train':
                        temp_img = resizeImg(temp_img, scale = 1 / self.config.dataset.img_downscale)

                    img = temp_img[:, :, :3][:, :, ::-1]
                    if self.config.dataset.blur:
                        # imgBlurDir = makePath(osp.join(self.config.save_dir, 'imgBlur') )
                        for i in range(self.config.dataset.blurNum):
                            img = cv2.GaussianBlur(img, (5,5), 0)
                        # cv2.imwrite( osp.join(imgBlurDir, '{}_{}.jpg'.format(fIdx, cIdx)), img[:,:,::-1])
                        pass
                    msk = temp_img[:, :, 3] / 255
                    img = (img[..., :3] / 255).astype(np.float32)
                    msk = msk.astype(np.float32)
                    img = torch.Tensor(img)
                    msk = torch.Tensor(msk)
                    all_fg_masks_temp.append(msk) # (h, w)
                    all_images_temp.append(img)

                    if self.config.dataset.with_depth and self.split == 'train':
                        temp_depth = cv2.imread(self.dep_lists[fIdx][cIdx] , -1)[:, :, ::-1]
                        temp_depth = temp_depth / 255.0 * 10.0
                        if self.config.dataset.img_downscale != 1.0 and self.split == 'train':
                            temp_depth = resizeImg(temp_depth, scale=1 / self.config.dataset.img_downscale)
                        all_depths_temp.append(torch.Tensor(temp_depth))

                self.all_images.append(torch.stack(all_images_temp, dim=0))
                self.all_fg_masks.append(torch.stack(all_fg_masks_temp, dim=0))
                if self.config.dataset.with_depth and self.split == 'train':
                    self.all_depths.append(torch.stack(all_depths_temp, dim=0))

            self.all_c2w, self.all_images, self.all_fg_masks = \
                self.all_c2w.float().to(self.rank), \
                torch.stack(self.all_images, dim=0).float(), \
                torch.stack(self.all_fg_masks, dim=0).float()
            if self.config.dataset.with_depth and self.split == 'train':
                self.all_depths = torch.stack(self.all_depths, dim=0).float()
        else:
            self.all_c2w = self.all_c2w.float().to(self.rank)

        self.w2s = self.w2s.to(self.rank)
        self.near = self.near.to(self.rank)
        self.far = self.far.to(self.rank)

        self.T_inv = self.T_inv.to(self.rank)
        self.vertices = self.vertices.to(self.rank)
        self.face_vertices = self.face_vertices.to(self.rank)
        self.face_T_inv_flat = self.face_T_inv_flat.to(self.rank)

        if not self.T_inv_world is None:
            self.T_inv_world = self.T_inv_world.to(self.rank)

    def get_bbox_from_smpl(self, vs, factor=1.2):
        """
        This function calculates the bounding box of a 3D model based on its vertices.
        param:
            vs: tensor of shape (T, N, 3) representing the vertices coordinates of the 3D model.
            factor: scaling factor to adjust the size of the bounding box. Default value is 1.2.
        return:
             tensor of shape (2, 3) representing the minimum and maximum coordinates of the bounding box.
        """
        # T, N, 3
        min_vert = (vs[:].min(dim=1).values).min(dim=0, keepdim=True).values
        max_vert = (vs[:].max(dim=1).values).max(dim=0, keepdim=True).values
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
        if not batch is None:
            batch_size = batch
        else:
            batch_size = smpl_params["betas"].shape[0]
        body_pose_t = torch.zeros((batch_size, 69), device=device)
        body_pose_t[:, 2] = torch.pi / 6
        body_pose_t[:, 5] = -torch.pi / 6
        smpl_outputs = self.body_model(betas=smpl_params["betas"], body_pose=body_pose_t)
        self.bbox = self.get_bbox_from_smpl(smpl_outputs.vertices[0:1].detach()) # T,3
        self.T_template = smpl_outputs.T[:1]                                     # 1,6890,4,4
        self.vs_template = smpl_outputs.vertices[:1]                             # 1,6890,3
        self.pose_offset_t = smpl_outputs.pose_offsets[:1]                       # 1,6890,3
        self.shape_offset_t = smpl_outputs.shape_offsets[:1]                     # 1,6890,3

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
        s2w = smpl_outputs.A[:, 0] # T,4,4
        w2s = torch.inverse(s2w)   # T,4,4

        T_inv = torch.inverse(smpl_outputs.T.float()).clone() @ s2w[:, None]  #T,6890,4,4 @ T,1,4,4
        T_inv[..., :3, 3] += self.pose_offset_t - smpl_outputs.pose_offsets   #T,6890,3 + 1,6890,3
        T_inv[..., :3, 3] += self.shape_offset_t - smpl_outputs.shape_offsets #T,6890,3 + 1,6890,3
        T_inv = self.T_template @ T_inv  # T, 6890, 4, 4 @ 1, 6890, 4, 4 -> T,6890,4,4
        self.T_inv = T_inv.detach()
        self.vertices = (smpl_outputs.vertices @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[:, None, :3, 3] #T,6890,3 @ T,3,3 + T,1,3
        self.w2s = w2s
        self.vertices_world = torch.matmul(s2w[:, None, :3, :3], self.vertices.clone().reshape(-1, 6890, 3, 1) * 1.0)[:, :, :, 0] + s2w[:, None, :3, 3]

        self.facesTorch = torch.Tensor(self.faces).to(torch.long)   # F,3
        self.face_vertices = self.vertices[:, self.facesTorch].to(self.rank) #T,F,3,3
        self.face_T_inv_flat = self.T_inv[:, self.facesTorch].reshape(self.face_vertices.shape[0], self.face_vertices.shape[1], self.face_vertices.shape[2], -1).to(self.rank) # T,F,3,4,4

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

        os.makedirs(self.config.save_dir, exist_ok= True)
        for i in range(len(self.frames)):
            save_ply(osp.join(  self.config.save_dir , 'debug_fk_template_{}.ply'.format(i)), self.vertices[i].data.cpu().numpy().reshape(-1,3) , self.faces_subdiv)
        save_ply(osp.join(  self.config.save_dir , 'debug_template.ply'  ), self.vs_template[0].data.cpu().numpy().reshape(-1,3) , self.faces)


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
            motion_init = self.egInit.character.motion_base.clone()
            motion_init[:, :6] = 0.0
            skinRs, skinTs = self.eg.updateNode(motion_init)
            self.vs_template, self.T_template, Tdeform_ = self.eg.forwardF(skinRs=skinRs, skinTs=skinTs, returnTransformation=True)
        elif self.config.dataset.cano_motion =='world':
            motion_init = ddc_params["motion"].clone()
            motion_init[:, :6] = 0.0
            skinRs, skinTs = self.eg.updateNode(motion_init)
            deltaRs, deltaTs = ddc_params["deltaR"], ddc_params["deltaT"]
            self.vs_template, self.T_template, Tdeform_ = self.eg.forwardF(deltaRs=deltaRs, deltaTs=deltaTs * 1000.0, skinRs = skinRs, skinTs = skinTs,
                                                      displacements = ddc_params["displacement"] * 1000.0, returnTransformation=True)
        self.T_template = self.T_template @ Tdeform_ # T,N,4,4
        self.vs_template /= 1000.0                   # T,N,3
        self.bbox = self.get_bbox_from_smpl(self.vs_template.detach()) # T, 2, 3
        self.s2w_init = self.eg.character.jointTransformations[:, 0] # T,4,4

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

        vertsFList = []
        TskinList = []
        TdeformList = []
        s2wList = []

        lst = range(len(self.frames))
        lst_chunks = np.array_split(lst, (len(lst) // 50) + 1)
        for chunk in lst_chunks:
            skinRs, skinTs = self.eg.updateNode(ddc_params["motion"][chunk])
            deltaRs, deltaTs = ddc_params["deltaR"][chunk], ddc_params["deltaT"][chunk]
            displacements = ddc_params["displacement"][chunk]
            vertsF, Tskin, Tdeform = self.eg.forwardF(deltaRs=deltaRs, deltaTs=deltaTs * 1000.0, skinRs = skinRs, skinTs = skinTs, displacements = displacements * 1000.0, returnTransformation=True)

            s2w = self.eg.character.jointTransformations[:,0]
            vertsFList.append(vertsF)
            TskinList.append(Tskin)
            TdeformList.append(Tdeform)
            s2wList.append(s2w)

        vertsF = torch.cat(vertsFList, 0)
        Tskin = torch.cat(TskinList, 0)
        Tdeform = torch.cat(TdeformList, 0)
        s2w = torch.cat(s2wList, 0)

        s2w = s2w @ torch.inverse(self.s2w_init) # T,4,4
        w2s = torch.inverse(s2w)
        self.w2s = w2s             # T,4,4

        self.Tskin_inv = torch.inverse(Tskin) @ s2w[:, None]  # T,N,4,4 @ T,1,4,4 -> T, N, 4, 4
        self.Tdeform_inv = torch.inverse(Tdeform)        # T, N, 4, 4

        laplacian_matrix = uniform_laplacian(vertsF.shape[1], torch.Tensor(self.eg.character.faces).to(torch.long), self.eg.character.labelMatrix)
        laplacian_matrix = laplacian_matrix[None] # 1, N, N
        Tfull =  w2s[:, None] @ Tskin @ Tdeform          # T, N, 4, 4
        Tfull_DQ = fromTransformation2VectorTorch(Tfull) # T, N, 8
        if self.config.dataset.smoothDQ:
            Tfull_DQ += torch.matmul(laplacian_matrix, Tfull_DQ) * 0.3 # T, N, 8 * 0.3
            Tfull_DQ += torch.matmul(laplacian_matrix, Tfull_DQ) * 0.3 # T, N, 8
            Tfull_DQ += torch.matmul(laplacian_matrix, Tfull_DQ) * 0.3 # T, N, 8
            Tfull4 = fromVector2TransformationTorch(Tfull_DQ)# T, N, 4, 4
        else:
            Tfull4 = Tfull

        self.T_inv = self.T_template @ torch.inverse(Tfull4) # T, N, 4, 4    #[None]
        T_temp = torch.inverse(self.T_inv)  # T, N, 4, 4
        self.vertices = torch.matmul(T_temp[:, :, :3, :3], self.vs_template[:,:,:,None] * 1000.0)[:, :, :, 0] + T_temp[:, :, :3, 3]
        self.vertices /= 1000.0
        self.vertices_old = self.vertices.clone() # T,N,3
        self.T_inv[:,:,:3,3] /= 1000.0      # T, N, 4, 4
        self.w2s[:,:3,3] /= 1000.0      # T, 4, 4

        self.facesTorch = torch.Tensor(self.faces).to(torch.long)   # F,3
        self.face_vertices = self.vertices[:, self.facesTorch].to(self.rank) #T,F,3,3
        self.face_T_inv_flat = self.T_inv[:, self.facesTorch].reshape(self.face_vertices.shape[0], self.face_vertices.shape[1],\
                                                                      self.face_vertices.shape[2], -1).to(self.rank)

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

        os.makedirs(self.config.save_dir, exist_ok= True)
        for i in range(len(self.frames)):
            save_ply(osp.join(  self.config.save_dir , 'debug_fk_template_{}.ply'.format(i)), self.vertices[i].data.cpu().numpy().reshape(-1,3) , self.faces_subdiv)
        save_ply(osp.join(  self.config.save_dir , 'debug_template.ply'  ), self.vs_template[0].data.cpu().numpy().reshape(-1,3) , self.faces)

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

        # pts = pts * torch.from_numpy(self.cano_bounds[1] - self.cano_bounds[0]).to(pts) + torch.from_numpy(self.cano_bounds[0]).to(pts)
        pts = pts * (self.cano_bounds[1] - self.cano_bounds[0]) + self.cano_bounds[0]

        if self.config.dataset.deformer == 'smpl':
            # threshold = 0.075#0.1
            threshold = self.threshold_smpl#0.1
        else:
            # threshold = 0.05
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
        print('+ Canonical volume len: {}'.format(self.cano_bounds[1] - self.cano_bounds[0]))

        self.resolutions_world = torch.Tensor(self.config.model.geometry.isosurface.resolution).reshape(-1, 3).to(self.rank)
        x_coords = torch.linspace(0, 1, steps=self.config.model.geometry.isosurface.resolution[0],
                                  dtype=torch.float32, device=self.rank).detach()
        y_coords = torch.linspace(0, 1, steps=self.config.model.geometry.isosurface.resolution[1],
                                  dtype=torch.float32, device=self.rank).detach()
        z_coords = torch.linspace(0, 1, steps=self.config.model.geometry.isosurface.resolution[2],
                                  dtype=torch.float32, device=self.rank).detach()
        xv, yv, zv = torch.meshgrid(x_coords, y_coords, z_coords)  # print(xv.shape) # (256, 256, 256)
        xv = torch.reshape(xv, (-1, 1))  # print(xv.shape) # (256*256*256, 1)
        yv = torch.reshape(yv, (-1, 1))
        zv = torch.reshape(zv, (-1, 1))
        pts = torch.cat([xv, yv, zv], dim=-1)

        pts = pts * (self.world_bounds[1] - self.world_bounds[0]) + self.world_bounds[0]

        if self.config.dataset.deformer == 'smpl':
            # threshold = 0.075#0.1
            threshold = self.threshold_smpl#0.1
        else:
            # threshold = 0.05
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
            self.valid_world = valid.reshape(-1)

            print('+ Valid world point number: '.format(valid.sum()))
            self.infer_pts = pts.float()[self.valid_world].float()
            dist_, face_, type_ = point_to_mesh_distance(self.infer_pts[None], self.face_vertices[:1])
            T_inv, _ = _unbatched_point_to_mesh_interp(self.infer_pts, face_[0], type_[0], self.face_vertices[0], self.face_T_inv_flat[0])

        self.T_inv_world = T_inv.reshape(-1, 4, 4)
        del pts
        del self.infer_pts


    def prepare_grid_pts(self):
        """
        Prepares grid points within the specified radius for the given model.

        return:
            None
        """
        grid_bounds = torch.as_tensor([-self.config.model.radius, -self.config.model.radius, -self.config.model.radius,
                                        self.config.model.radius, self.config.model.radius, self.config.model.radius], dtype=torch.float32).to(self.rank)
        x_coords = torch.linspace(0, 1, steps=self.config.model.grid_resolution,
                                  dtype=torch.float32, device=self.rank).detach()
        y_coords = torch.linspace(0, 1, steps=self.config.model.grid_resolution,
                                  dtype=torch.float32, device=self.rank).detach()
        z_coords = torch.linspace(0, 1, steps=self.config.model.grid_resolution,
                                  dtype=torch.float32, device=self.rank).detach()
        xv, yv, zv = torch.meshgrid(x_coords, y_coords, z_coords)  # print(xv.shape) # (256, 256, 256)
        xv = torch.reshape(xv, (-1, 1))  # print(xv.shape) # (256*256*256, 1)
        yv = torch.reshape(yv, (-1, 1))
        zv = torch.reshape(zv, (-1, 1))
        pts = torch.cat([xv, yv, zv], dim=-1)
        pts = pts * (grid_bounds[1] - grid_bounds[0]) + grid_bounds[0]

        self.validInOut_grid = []
        for idx in tqdm(range(self.vertices.shape[0])):
            world_smpl_trimesh = trimesh.Trimesh(self.vertices[idx].to(self.rank).data.cpu().numpy(), self.faces_subdiv)
            if not self.config.dataset.loose:
                validInOut_grid = check_mesh_contains(world_smpl_trimesh, pts.data.cpu().numpy())[0]
            else:
                validInOut_grid = implicit_waterproofing(world_smpl_trimesh, pts.data.cpu().numpy())[0]
            self.validInOut_grid.append(torch.Tensor(validInOut_grid).reshape(-1) > 0)
        self.validInOut_grid = torch.stack(self.validInOut_grid, 0 )



class DomeDenseRawSeqMetaDataset(Dataset, DomeDenseRawSeqMetaDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        # return len(self.all_images)
        return len(self.all_c2w)
    
    def __getitem__(self, index):
        fIdx = index % len(self.frames)
        cIdx = index // len(self.frames)
        return {
            'index': index,
            'fIdx': fIdx,
            'cIdx': cIdx,
        }


class DomeDenseRawSeqMetaIterableDataset(IterableDataset, DomeDenseRawSeqMetaDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('domedenserawseq_meta')
class DomeDenseRawSeqMetaDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = DomeDenseRawSeqMetaIterableDataset(self.config, self.config.dataset.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = DomeDenseRawSeqMetaDataset(self.config, self.config.dataset.val_split)
        if stage in [None, 'test']:
            self.test_dataset = DomeDenseRawSeqMetaDataset(self.config, self.config.dataset.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = DomeDenseRawSeqMetaDataset(self.config, self.config.dataset.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
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
