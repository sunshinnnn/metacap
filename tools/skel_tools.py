"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-01-27
"""

import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from .rotation_tools import *
from .pyrender_tools import create_point, create_skeleton
from .mesh_tools import load_obj_mesh, save_ply
from .omni_tools import to_cpu, to_tensorFloat, to_tensor, checkPlatformDir, get_palette
from .torch3d_transforms import *
from tqdm import tqdm
import trimesh
from pytorch3d import ops

def load_ddc_param(path, returnTensor=False, frames = None):
    """
    Load DDC parameters from a given file path.

    param:
     path: The path to the parameter file.
     returnTensor: True to return parameters as PyTorch tensors, False to return as numpy arrays. Default is False.
     frames: A list of frame indices to load parameters for. Default is None, which loads parameters for all frames.
    return:
     A dictionary containing motion, deltaR, deltaT, and displacement parameters.

    """
    try:
        params = dict(np.load(str(path)))
    except FileNotFoundError:
        print(f"File not found at {path}")
        return None

    if isinstance(frames, int):
        frames = [frames]
    if frames is None:  # If frames is None, load parameters for all frames
        frames = params['frameList']
    assert isinstance(frames, list)

    idxList = []
    for frame in frames:
        try:
            idx = list(params['frameList']).index(frame)
            idxList.append(idx)
        except ValueError:
            print(f"Frame {frame} not found in parameter file.")
            continue

    motion = params["motionList"].astype(np.float32)[idxList]
    deltaR = params["deltaRList"].astype(np.float32)[idxList]
    deltaT = params["deltaTList"].astype(np.float32)[idxList]
    displacement = params["displacementList"].astype(np.float32)[idxList]

    # Check for tuple type and get first item
    if isinstance(motion, tuple):
        motion = motion[0]
    if isinstance(deltaR, tuple):
        deltaR = deltaR[0]
    if isinstance(deltaT, tuple):
        deltaT = deltaT[0]
    if isinstance(displacement, tuple):
        displacement = displacement[0]
    if returnTensor:
        motion = torch.Tensor(motion)
        deltaR = torch.Tensor(deltaR)
        deltaT = torch.Tensor(deltaT)
        displacement = torch.Tensor(displacement)
    return {
        "motion": motion,
        "deltaR": deltaR,
        "deltaT": deltaT,
        "displacement": displacement,
    }


def load_smpl_param(path, returnTensor = False, frames = None):
    """
    Load SMPL parameters from a given path.

    param:
     path: The path to the SMPL parameter file.
     returnTensor: Flag indicating whether to return the parameters as torch.Tensor objects. Default is False.
     frames: The frames to load from the parameters. This must be a list of frame indices. Default is None.

    return:
     A dictionary containing the SMPL parameters.
    """
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

    N = smpl_params["body_pose"].shape[0]
    if frames is None:
        betas = smpl_params["betas"].astype(np.float32).reshape(-1, 10)
        body_pose = smpl_params["body_pose"].astype(np.float32)
        global_orient = smpl_params["global_orient"].astype(np.float32)
        transl =  smpl_params["transl"].astype(np.float32)
        # print(betas.shape[0])
        if betas.shape[0] != N:
            betas = np.repeat(betas, N, 0 )
    else:
        assert isinstance(frames, list)
        betas = smpl_params["betas"].astype(np.float32).reshape(-1, 10)
        if betas.shape[0] != N:
            betas = np.repeat(betas, N, 0)
        betas = betas[frames]
        body_pose = smpl_params["body_pose"].astype(np.float32)[frames]
        global_orient = smpl_params["global_orient"].astype(np.float32)[frames]
        transl =  smpl_params["transl"].astype(np.float32)[frames]
    if isinstance(betas, tuple):
        betas = betas[0]
    if isinstance(body_pose, tuple):
        body_pose = body_pose[0]
    if isinstance(global_orient, tuple):
        global_orient = global_orient[0]
    if isinstance(transl, tuple):
        transl = transl[0]
    if returnTensor:
        betas = torch.Tensor(betas)
        body_pose = torch.Tensor(body_pose)
        global_orient = torch.Tensor(global_orient)
        transl = torch.Tensor(transl)
    return {
        "betas": betas,
        "body_pose": body_pose,
        "global_orient": global_orient,
        "transl": transl,
    }


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def load_model(cfgs, useCuda=True, device=None):
    if device is None:
        if useCuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    charPath = checkPlatformDir(cfgs.characterPath)
    graphPath = checkPlatformDir(cfgs.graphPath)
    connectionPath = checkPlatformDir(cfgs.connectionPath)

    computeConnectionFlag = cfgs.compute_connection
    useDQ = cfgs.useDQ
    verbose = cfgs.verbose
    segWeightsFlag = cfgs.load_seg_weights
    computeAdjacencyFlag = cfgs.compute_adjacency

    sk = SkinnedCharacter(charPath=charPath, useDQ=useDQ, verbose=verbose, device=device, \
                          segWeightsFlag=segWeightsFlag, computeAdjacencyFlag = computeAdjacencyFlag)
    eg = EmbeddedGraph(character=sk, graphPath=graphPath, segWeightsFlag = segWeightsFlag, computeConnectionFlag=computeConnectionFlag, connectionPath=connectionPath, useDQ=useDQ, verbose=verbose, device=device)

    return eg


class EmbeddedGraph():
    def __init__(self, character=None, graphPath=None, segWeightsFlag = False, computeConnectionFlag = False, connectionPath=None, useDQ = True, verbose=False, device='cpu'):
        self.character = character
        self.graphPath = graphPath
        self.connectionPath = connectionPath
        self.verbose = verbose
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.useDQ = useDQ
        assert self.useDQ == self.character.useDQ
        assert self.device == self.character.device

        self.segWeightsFlag = segWeightsFlag
        self.computeConnectionFlag = computeConnectionFlag
        self.numVert = self.character.numVert
        self.loadMeshG()
        self.setupNodeGraph()
        self.setupConnection()
        self.normalizeWeights()    #==> add into self.setupConnection()
        self.computeConnectionsNr()
        self.EGNodeToBaseMeshVertices = self.nodeIdxs

        self.vertsOfGraphOnBaseMesh = self.character.verts0[self.nodeIdxs.astype('int').tolist()]
        # skinRs, skinTs = self.updateNode(self.character.motion_base)
        # self.

    def setupConnection(self):
        if self.computeConnectionFlag:
            if self.verbose:
                print('Compute Connections through computeConnectionFlag')
            self.computeConnections()
        else:
            fname = osp.join(osp.dirname(self.graphPath),
                             osp.basename(self.graphPath).split('.')[0] + '_connection.npz')
            if not self.connectionPath is None and os.path.isfile(self.connectionPath):
                if self.verbose:
                    print('Load Connecttions from ', self.connectionPath)
                dataz = np.load(self.connectionPath, allow_pickle=True)
                self.connectionIdxs = list(dataz['connectionIdxs'])
                self.connectionWeights = list(dataz['connectionWeights'])
            # elif self.connectionPath is None and os.path.isfile(fname):
            elif os.path.isfile(fname):
                self.connectionPath = fname
                dataz = np.load(self.connectionPath, allow_pickle=True)
                self.connectionIdxs = list(dataz['connectionIdxs'])
                self.connectionWeights = list(dataz['connectionWeights'])
            # elif self.connectionPath is None and not os.path.isfile(fname):
            elif not os.path.isfile(fname):
                if self.verbose:
                    print('Compute Connections through no file found')
                self.computeConnections()
            else:
                raise NotImplementedError("Not included in the if condition?? What's up?")


    def loadMeshG(self):
        # load mesh
        self.vertsG0, self.facesG = load_obj_mesh(self.graphPath)
        # self.facesG = self.facesG.astype(np.int32)
        self.vertsG0 = torch.from_numpy(self.vertsG0).float().to(self.device)
        self.vertsG0Source = self.vertsG0.clone()
        self.numVertG = self.vertsG0.shape[0]
        self.numFaceG = self.facesG.shape[0]
        if self.verbose:
            print('-- loaded graph mesh has {} verts {} faces'.format(self.numVertG, self.numFaceG))

        self.computeNeighboursG()
        if self.segWeightsFlag:
            self.loadSegmentationWeightsG()
        self.computeAdjacency()
        nrAllConnections = 0
        for i in range(self.numVertG):
            nrAllConnections += len(self.neighboursG[i])
        self.numEdgeG = int(nrAllConnections / 2)
        self.numNeighboursG = np.zeros(self.numVertG)
        self.neighbourOffsetsG = np.zeros(self.numVertG + 1)
        self.neighbourIdxsG = np.zeros(2 * self.numEdgeG)
        count = 0
        offset = 0
        self.neighbourOffsetsG[0] = 0
        for i in range(self.numVertG):
            pass
            valance = len(self.neighboursG[i])
            self.numNeighboursG[count] = valance
            for j in range(valance):
                self.neighbourIdxsG[offset] = self.neighboursG[i][j]
                offset += 1
            self.neighbourOffsetsG[count + 1] = offset
            count += 1

    def loadSegmentationWeightsG(self):
        verts0 = self.character.verts0.data.cpu().numpy()
        vertsG0 = self.vertsG0.data.cpu().numpy()

        from scipy.spatial import KDTree
        kdtree = KDTree(verts0)
        idxls = []
        dls = []
        for idx in range(len(vertsG0)):
            d, i = kdtree.query(vertsG0[idx])
            dls.append(d)
            idxls.append(i)
        self.vertexWeightsG = np.array(self.character.vertexWeights)[idxls].tolist()
        self.vertexLabelsG = np.array(self.character.vertexLabels)[idxls].tolist()
        # palettes = get_palette(20)
        # save_ply(r'D:\06_Exps\test.ply',vertsG0,self.facesG,palettes[self.vertexLabelsG] )



    def computeAdjacency(self):
        self.adjacency = np.zeros((self.numVertG, self.numVertG), dtype=np.float32)
        self.compressedAdjacency = [[] for _ in range(self.numVertG)]
        self.numberOfEdges = 0
        self.numberOfNeigbours = np.zeros((self.numVertG))
        for i in range(self.numFaceG):
            v0 = self.facesG[i, 0]
            v1 = self.facesG[i, 1]
            v2 = self.facesG[i, 2]
            self.adjacency[v0, v1] = 1
            self.adjacency[v0, v2] = 1
            self.adjacency[v1, v0] = 1
            self.adjacency[v1, v2] = 1
            self.adjacency[v2, v0] = 1
            self.adjacency[v2, v1] = 1

            # v0
            if v1 + 1 not in self.compressedAdjacency[v0]:
                self.compressedAdjacency[v0].append(v1 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v0] = self.numberOfNeigbours[v0] + 1
            if v2 + 1 not in self.compressedAdjacency[v0]:
                self.compressedAdjacency[v0].append(v2 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v0] = self.numberOfNeigbours[v0] + 1
            # v1
            if v0 + 1 not in self.compressedAdjacency[v1]:
                self.compressedAdjacency[v1].append(v0 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v1] = self.numberOfNeigbours[v1] + 1
            if v2 + 1 not in self.compressedAdjacency[v1]:
                self.compressedAdjacency[v1].append(v2 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v1] = self.numberOfNeigbours[v1] + 1
            # v2
            if v0 + 1 not in self.compressedAdjacency[v2]:
                self.compressedAdjacency[v2].append(v0 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v2] = self.numberOfNeigbours[v2] + 1
            if v1 + 1 not in self.compressedAdjacency[v2]:
                self.compressedAdjacency[v2].append(v1 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v2] = self.numberOfNeigbours[v2] + 1
        self.compressedAdjacency = np.asarray(self.compressedAdjacency, dtype=object)
        self.maximumNumNeighbours = int(np.amax(self.numberOfNeigbours))
        self.laplacian = - self.adjacency
        for i in range(0, self.numVertG):
            self.laplacian[i, i] = self.numberOfNeigbours[i]

        # if self.segWeightsFlag:
        if False:
            # row weight
            if self.verbose:
                print('     ++ Compute row weights')
            self.rowWeight = np.zeros((self.numVert), dtype=np.float32)

            for i in range(0, self.numVert):
                self.rowWeight[i] = 0.0
                for j in range(0, len(self.compressedAdjacency[i])):
                    nIdx = self.compressedAdjacency[i][j] - 1
                    self.rowWeight[i] = self.rowWeight[i] + (self.vertexWeights[i] + self.vertexWeights[nIdx]) / 2.0
                self.rowWeight[i] = self.rowWeight[i] / float(self.numberOfNeigbours[i])

            # laplacian weighted
            if self.verbose:
                print('     ++ Compute laplacian weights')
            self.adjacencyWeights = np.zeros((self.numVert, self.numVert))
            for i in range(self.numFace):
                v0 = self.faces[i, 0]
                v1 = self.faces[i, 1]
                v2 = self.faces[i, 2]

                self.adjacencyWeights[v0, v1] = (self.vertexWeights[v0] + self.vertexWeights[v1]) / 2.0
                self.adjacencyWeights[v0, v2] = (self.vertexWeights[v0] + self.vertexWeights[v2]) / 2.0
                self.adjacencyWeights[v1, v0] = (self.vertexWeights[v1] + self.vertexWeights[v0]) / 2.0
                self.adjacencyWeights[v1, v2] = (self.vertexWeights[v1] + self.vertexWeights[v2]) / 2.0
                self.adjacencyWeights[v2, v0] = (self.vertexWeights[v2] + self.vertexWeights[v0]) / 2.0
                self.adjacencyWeights[v2, v1] = (self.vertexWeights[v2] + self.vertexWeights[v1]) / 2.0

    def setupNodeGraph(self):
        highestHighResVert, lowestHighResVert = torch.argmax(self.character.verts0[:, 1]), torch.argmin(self.character.verts0[:, 1])
        maxY, minY = self.character.verts0[:, 1][highestHighResVert], self.character.verts0[:, 1][lowestHighResVert]
        # neighbours = character.neighbours
        # distanceHighestVert, distanceLowestVert = character.computeGeodesicDistance(highestHighResVert, lowestHighResVert)
        distanceHighestVert = self.character.computeGeodesicDistance(highestHighResVert)
        distanceLowestVert = self.character.computeGeodesicDistance(lowestHighResVert)

        highestLowResVert, lowestLowResVert = torch.argmax(self.vertsG0[:, 1]), torch.argmin(self.vertsG0[:, 1])
        maxYLow, minYLow = self.vertsG0[:, 1][highestLowResVert], self.vertsG0[:, 1][lowestLowResVert]
        # distanceHighestVertLow, distanceLowestVertLow = self.computeGeodesicDistanceG(highestLowResVert, lowestLowResVert)
        distanceHighestVertLow = self.computeGeodesicDistanceG(highestLowResVert)
        distanceLowestVertLow = self.computeGeodesicDistanceG(lowestLowResVert)

        # # fill embedded nodes array
        # if max(distanceHighestVert) == 1000000 or max(distanceLowestVert) == 1000000 or \
        #         max(distanceHighestVertLow) == 1000000 or max(distanceLowestVertLow) == 1000000:
        #     print('[ERROR] You need to handle vert connect to mesh.')
        # else:
        # self.nodeRs = np.tile(np.eye(3)[None],(self.numVertG,1,1))
        # self.nodeRs = np.zeros((self.numVertG, 3))
        # self.nodeTs = np.zeros((self.numVertG, 3))
        self.nodeIdxs = np.ones(self.numVertG) * -1  # the closest idx on the high res mesh
        self.nodeNeighbours = []
        self.nodeRadius = np.zeros(self.numVertG)
        for i in range(self.numVertG):
            self.nodeNeighbours.append([])
        alreadyUsed = np.zeros(self.character.numVert)
        for i in range(self.numVertG):
            vertG_temp = self.vertsG0[i]
            norms_temp = torch.linalg.norm(self.character.verts0 - vertG_temp, axis=1)
            closestVertexDistance = 100000
            closestVertexId = -1
            # for j in range(self.character.numVert):
            #     graphConnectedToHigh = distanceHighestVertLow[i] != 1000000
            #     graphConnectedToLow = distanceLowestVertLow[i] != 1000000
            #     meshConnectedToHigh = distanceHighestVert[j] != 1000000
            #     meshConnectedToLow = distanceLowestVert[j] != 1000000
            #     distance = norms_temp[j]
            #     if distance< closestVertexDistance and not alreadyUsed[closestVertexId] and\
            #             (graphConnectedToHigh==meshConnectedToHigh) and (graphConnectedToLow==meshConnectedToLow):
            #         closestVertexId = j
            #         closestVertexDistance = distance

            closestVertexId = torch.argmin(norms_temp)
            closestVertexDistance = norms_temp[closestVertexId]

            graphConnectedToHigh = distanceHighestVertLow[i] != 1000000
            graphConnectedToLow = distanceLowestVertLow[i] != 1000000
            meshConnectedToHigh = distanceHighestVert[closestVertexId] != 1000000
            meshConnectedToLow = distanceLowestVert[closestVertexId] != 1000000

            # print(i)
            while not((graphConnectedToHigh ==meshConnectedToHigh) and (graphConnectedToLow==meshConnectedToLow)):
                norms_temp[closestVertexId] = 100000.0
                closestVertexId = torch.argmin(norms_temp)
                closestVertexDistance = norms_temp[closestVertexId]
                meshConnectedToHigh = distanceHighestVert[closestVertexId] != 1000000
                meshConnectedToLow = distanceLowestVert[closestVertexId] != 1000000

            while alreadyUsed[closestVertexId]:
                norms_temp[closestVertexId] = 100000.0
                closestVertexId = torch.argmin(norms_temp)
                closestVertexDistance = norms_temp[closestVertexId]

            assert (closestVertexId != -1)
            alreadyUsed[closestVertexId] = True

            self.nodeIdxs[i] = closestVertexId
            # add the neighbourhood information
            neighbourOffset = self.neighbourOffsetsG[i]
            numNeighbour = self.numNeighboursG[i]
            for j in range(int(numNeighbour)):
                # print(int(neighbourOffset+ j))
                neighbourIdx = self.neighbourIdxsG[int(neighbourOffset + j)]
                self.nodeNeighbours[i].append(neighbourIdx)
                # self.nodeNeighbours[i] = list(set(self.nodeNeighbours[i]))

                # closestVertexId = torch.argmin(norms_temp)
                # closestVertexDistance = norms_temp[closestVertexId]
                #
                # graphConnectedToHigh = distanceHighestVertLow[i] != 1000000
                # graphConnectedToLow = distanceLowestVertLow[i] != 1000000
                # meshConnectedToHigh = distanceHighestVert[closestVertexId] != 1000000
                # meshConnectedToLow = distanceLowestVert[closestVertexId] != 1000000
                #
                # if distanceHighestVert[closestVertexId]==1000000 or distanceLowestVert[closestVertexId]==1000000 or \
                #     distanceHighestVert[closestVertexId] == 1000000 or distanceHighestVert[closestVertexId]==1000000:
                #
                # while alreadyUsed[closestVertexId]:
                #     norms_temp[closestVertexId] = 100000.0
                #     closestVertexId = torch.argmin(norms_temp)
                #     closestVertexDistance = norms_temp[closestVertexId]
                # alreadyUsed[closestVertexId] = True
                # self.nodeIdxs[i] = closestVertexId
                # # add the neighbourhood information
                # neighbourOffset = self.neighbourOffsetsG[i]
                # numNeighbour = self.numNeighboursG[i]
                # for j in range(int(numNeighbour)):
                #     # print(int(neighbourOffset+ j))
                #     neighbourIdx = self.neighbourIdxsG[int(neighbourOffset + j)]
                #     self.nodeNeighbours[i].append(neighbourIdx)
                # # self.nodeNeighbours[i] = list(set(self.nodeNeighbours[i]))

    def computeNeighboursG(self):
        self.neighboursG = []
        for i in range(self.numVertG):
            self.neighboursG.append([])
        for i in range(self.numFaceG):
            self.neighboursG[self.facesG[i, 0]].append(self.facesG[i, 1])
            self.neighboursG[self.facesG[i, 0]].append(self.facesG[i, 2])
            self.neighboursG[self.facesG[i, 1]].append(self.facesG[i, 0])
            self.neighboursG[self.facesG[i, 1]].append(self.facesG[i, 2])
            self.neighboursG[self.facesG[i, 2]].append(self.facesG[i, 0])
            self.neighboursG[self.facesG[i, 2]].append(self.facesG[i, 1])
        for i in range(self.numVertG):
            self.neighboursG[i] = list(set(self.neighboursG[i]))

    def computeGeodesicDistanceG(self, idx=-1):
        distances = np.ones(self.numVertG) * 1000000
        distances[idx] = 0

        import queue
        Q = queue.Queue()
        Q.put(idx)
        while not Q.empty():
            v = Q.get()
            for j in range(len(self.neighboursG[v])):
                # print('v:{},j:{},value:{}'.format(v,j,self.neighbours[v][j]))
                # print(distancesH.shape)
                if distances[self.neighboursG[v][j]] == 1000000:
                    Q.put(self.neighboursG[v][j])
                    distances[self.neighboursG[v][j]] = distances[v] + 1
        return distances

    def computeConnections(self):
        """
            Super Slow, so save the results of connection into a file.
        """
        # Estimate distance of each vertex from each embedded node
        # and estimate radius of each embedded node
        distance = []
        # distanceG = []

        for i in range(self.numVertG):
            distance.append([])
            # distanceG.append([])
        # radius = longest distance to neighbouring nodes (walking on the undecimated mesh)
        # || one step -> radius 1 ||  two steps -> radius 2 and so on

        # nb.jit(nopython=True,parallel=True)
        for i in tqdm(range(self.numVertG)):
            # print(i)
            distance[i] = self.character.computeGeodesicDistance(int(self.nodeIdxs[i]))  # distance in high mesh
            # distance[i] = self.character.computeGeodesicDistance2(int(self.nodeIdxs[i])) # distance in high mesh
            for j in range(len(self.nodeNeighbours[i])):
                # print(i,j)
                # print(self.nodeNeighbours[i][j])
                # print(self.nodeIdxs[int(self.nodeNeighbours[i][j])])
                # print(int(self.nodeIdxs[int(self.nodeNeighbours[i][j])]))
                d = distance[i][
                    int(self.nodeIdxs[int(self.nodeNeighbours[i][j])])]  # each neighbour on graph->high mesh-> distance
                if d > self.nodeRadius[i]:
                    self.nodeRadius[i] = d
            self.nodeRadius[i] = np.floor(self.nodeRadius[i] / 2)
            if self.nodeRadius[i] < 3:
                self.nodeRadius[i] = 3

        self.connectionIdxs = []
        self.connectionWeights = []
        unconnectedVertices = 0
        for i in range(self.character.numVert):
            self.connectionIdxs.append([])
            self.connectionWeights.append([])
        for v in range(self.character.numVert):
            for i in range(self.numVertG):
                d = distance[i][v] / self.nodeRadius[i]
                if d <= 1:
                    self.connectionIdxs[v].append(i)
                    self.connectionWeights[v].append(np.exp(-0.5 * d * d))
            if len(self.connectionIdxs[v]) == 0:
                unconnectedVertices += 1
                dmin = 1000000.0
                imin = -1
                for i in range(self.numVertG):
                    d = distance[i][v]
                    if d < dmin:
                        dmin = d
                        imin = i
                self.connectionIdxs[v] = [imin]
                self.connectionWeights[v] = [1.0]

        fname = osp.join(osp.dirname(self.graphPath), osp.basename(self.graphPath).split('.')[0] + '_connection.npz')
        if not os.path.isfile(fname):
            print("Save connections into: ", fname)
            np.savez(fname, connectionIdxs=np.asarray(self.connectionIdxs, dtype=object),
                     connectionWeights=np.asarray(self.connectionWeights, dtype=object))

        if unconnectedVertices > 0:
            print("There are {} vertices where no closest graph node was found!".format(unconnectedVertices))
        else:
            print("All vertices found closest graph node")

    def normalizeWeights(self):
        print('normalize Weights')
        for v in range(self.character.numVert):
            K = len(self.connectionIdxs[v])
            w = 0
            for i in range(K):
                w += self.connectionWeights[v][i]
            for i in range(K):
                self.connectionWeights[v][i] /= w

        self.connectionMat = torch.zeros((self.character.numVert, self.numVertG)).float().to(self.device)
        self.vertsOfBaseMeshExpand = torch.zeros([self.character.numVert, self.numVertG, 3]).float().to(self.device)
        for i in range(self.character.numVert):
            for j in range(len(self.connectionIdxs[i])):
                idxG = self.connectionIdxs[i][j]
                weightG = self.connectionWeights[i][j]
                self.connectionMat[i, idxG] = weightG
                self.vertsOfBaseMeshExpand[i, idxG] = self.vertsG0[idxG]

    def computeConnectionsNr(self):
        print('compute ConnectionsNr')
        # number of node to vertex connections
        self.nodeToVertexConnectionsNr = 0
        for i in range(self.character.numVert):
            self.nodeToVertexConnectionsNr += len(self.connectionIdxs[i])

        # number of node to node connections
        self.nodeToNodeConnectionsNr = 0
        for i in range(self.numVertG):
            self.nodeToNodeConnectionsNr += len(self.nodeNeighbours[i])

    def updateSkinMesh(self):
        pass

    def updateNode(self, dofParams, useDQ = None, returnT = False):
        """
            This function is consistent with pose2embededgraph operator.
            Input: dofParams[B,D]
            Output: skinRs[B,K,3], skinTs[B,K,3] (skin transformation)
        """

        if not useDQ is None:
            useDQS_m = useDQ
        else:
            useDQS_m = self.useDQ
        self.character.update(dofParams)
        self.B = B = dofParams.shape[0]

        skinRs = torch.zeros((self.B, self.numVertG, 3)).float().to(self.device)
        skinTs = torch.zeros((self.B, self.numVertG, 3)).float().to(self.device)


        if not returnT:
            if not useDQS_m:
                raise NotImplementedError("We haven't implemented LBS for node skin update yet.")
            else:
                # self.character.DQTransformations = torch.zeros((self.B, self.character.skinWeight.shape[1], 8)).float().to(self.device)
                # for i in range(self.character.skinWeight.shape[1]):
                #     self.character.DQTransformations[:, i] = fromTransformation2VectorTorch(self.character.jointTransformations[:, i])
                nodeDQTransformations = fromTransformation2VectorTorch(self.character.jointTransformations.reshape(B * self.character.skinWeight.shape[1],4,4)).float().to(self.device).reshape(B, self.character.skinWeight.shape[1], 8)
                #B,29,8

                nodeIdxsList = list(self.nodeIdxs.astype('int'))
                nodeSkinWeight = self.character.skinWeight[nodeIdxsList,:]
                # nodeDQTransformations = self.character.DQTransformations
                # nodeDQSigns = torch.ones((self.B, nodeSkinWeight.shape[1], 1)).float().to(self.device)
                # nodeVertTransformations = torch.zeros((self.B, self.numVertG, 4, 4)).float().to(self.device)
                # for i in range(nodeSkinWeight.shape[1]):
                #     nodeDQSigns[:, i] = torch.where(torch.einsum('bm,bm->b', nodeDQTransformations[:, 0, :4],
                #                                                   nodeDQTransformations[:, i, :4]).reshape(self.B, 1) > 0, 1, -1)

                # nodeDQSigns = torch.where(torch.einsum('bm,bm->b', nodeDQTransformations[:, 0, :4],
                #                                              nodeDQTransformations[:, i, :4]).reshape(self.B, 1) > 0, 1, -1)

                # nodeDQSigns = torch.ones((self.B, nodeSkinWeight.shape[1], 1)).float().to(self.device)
                # temp = (nodeDQTransformations[:,0:1,:4].clone()  * nodeDQTransformations[:,:,:4].clone()).sum(-1)[:,:,None]
                # temp = (nodeDQTransformations[:, :, :4].clone() * nodeDQTransformations[:, :, :4].clone().transpose(1,2)).sum(-1)[:, :, None]

                # temp = torch.einsum('bne,bem->bnm', nodeDQTransformations[:,:,:4].clone(), nodeDQTransformations[:, :, :4].clone().transpose(1,2))
                # nodeDQSigns = torch.where( temp>0, 1, -1 ).float().to(self.device)

                _, indices = torch.max(nodeSkinWeight, 1)
                nodeDQTransformationsInit = nodeDQTransformations[:, indices, :4].clone()
                temp = torch.einsum('bne,bem->bnm', nodeDQTransformationsInit, nodeDQTransformations[:, :, :4].clone().transpose(1,2))
                nodeDQSigns = torch.where(temp > 0, 1, -1).float().to(self.device)

                # nodeDQSigns = nodeDQSigns[:, indices] * nodeDQSigns.transpose(1,2)

                # for i in range(nodeSkinWeight.shape[1]):
                #     nodeDQTransformations[:, i] *= nodeDQSigns[:, i]
                # nodeDQTransformations *= nodeDQSigns
                nodeVertDQTransformations = torch.matmul(nodeSkinWeight.repeat(self.B, 1, 1) * nodeDQSigns, nodeDQTransformations)
                # for i in range(nodeSkinWeight.shape[0]):


                G = self.numVertG
                nodeVertTransformations = fromVector2TransformationTorch(nodeVertDQTransformations.view(-1,8)).float().to(self.device) #B*G,8 -> B*G,3,3
                skinRs = matrix_to_euler_angles(nodeVertTransformations[:,:3,:3]).reshape(B,G,3) #B*G,3,3 -> B*G,3 ->
                # for i in range(self.numVertG):
                #     nodeVertTransformations[:, i] = fromVector2TransformationTorch(nodeVertDQTransformations[:, i])
                #     skinRs[:, i] = matrix_to_euler_angles(nodeVertTransformations[:, i, :3, :3])
                    # skinTs[:, i] = nodeVertTransformations[:, i, :3, 3]
                skinTs = nodeVertTransformations.view(B,G,4,4)[:,:,:3, 3]
            return skinRs, skinTs
        else:
            if not useDQS_m:
                # nodeVertTransformations = torch.matmul(self.character.skinWeight.repeat(self.B, 1, 1), self.character.jointTransformations)
                # nodeVertTransformations = torch.matmul(self.character.skinWeight.repeat(self.B, 1, 1), self.character.jointTransformations)
                nodeVertTransformations = torch.einsum('vj,bjmn->bvmn', self.character.skinWeight, self.character.jointTransformations)

            else:
                nodeDQTransformations = fromTransformation2VectorTorch(
                    self.character.jointTransformations.reshape(B * self.character.skinWeight.shape[1], 4, 4)).float().to(
                    self.device).reshape(B, self.character.skinWeight.shape[1], 8)
                # B,29,8

                # nodeIdxsList = list(self.nodeIdxs.astype('int'))
                # nodeSkinWeight = self.character.skinWeight[nodeIdxsList, :]
                nodeSkinWeight = self.character.skinWeight[:, :]
                # nodeDQTransformations = self.character.DQTransformations
                # nodeDQSigns = torch.ones((self.B, nodeSkinWeight.shape[1], 1)).float().to(self.device)
                # nodeVertTransformations = torch.zeros((self.B, self.numVertG, 4, 4)).float().to(self.device)
                # for i in range(nodeSkinWeight.shape[1]):
                #     nodeDQSigns[:, i] = torch.where(torch.einsum('bm,bm->b', nodeDQTransformations[:, 0, :4],
                #                                                   nodeDQTransformations[:, i, :4]).reshape(self.B, 1) > 0, 1, -1)

                # nodeDQSigns = torch.where(torch.einsum('bm,bm->b', nodeDQTransformations[:, 0, :4],
                #                                              nodeDQTransformations[:, i, :4]).reshape(self.B, 1) > 0, 1, -1)

                # nodeDQSigns = torch.ones((self.B, nodeSkinWeight.shape[1], 1)).float().to(self.device)
                # temp = (nodeDQTransformations[:,0:1,:4].clone()  * nodeDQTransformations[:,:,:4].clone()).sum(-1)[:,:,None]
                # temp = (nodeDQTransformations[:, :, :4].clone() * nodeDQTransformations[:, :, :4].clone().transpose(1,2)).sum(-1)[:, :, None]

                # temp = torch.einsum('bne,bem->bnm', nodeDQTransformations[:,:,:4].clone(), nodeDQTransformations[:, :, :4].clone().transpose(1,2))
                # nodeDQSigns = torch.where( temp>0, 1, -1 ).float().to(self.device)

                _, indices = torch.max(nodeSkinWeight, 1)
                nodeDQTransformationsInit = nodeDQTransformations[:, indices, :4].clone()
                temp = torch.einsum('bne,bem->bnm', nodeDQTransformationsInit,
                                    nodeDQTransformations[:, :, :4].clone().transpose(1, 2))
                nodeDQSigns = torch.where(temp > 0, 1, -1).float().to(self.device)

                # nodeDQSigns = nodeDQSigns[:, indices] * nodeDQSigns.transpose(1,2)

                # for i in range(nodeSkinWeight.shape[1]):
                #     nodeDQTransformations[:, i] *= nodeDQSigns[:, i]
                # nodeDQTransformations *= nodeDQSigns
                nodeVertDQTransformations = torch.matmul(nodeSkinWeight.repeat(self.B, 1, 1) * nodeDQSigns,
                                                         nodeDQTransformations)
                # for i in range(nodeSkinWeight.shape[0]):

                G = self.numVertG
                nodeVertTransformations = fromVector2TransformationTorch(nodeVertDQTransformations.view(-1, 8)).float().to(
                    self.device)  # B*G,8 -> B*G,3,3

            return nodeVertTransformations.reshape(B,-1,4,4)


    def forwardG(self, deltaRs = None, deltaTs = None, skinRs = None, skinTs = None, B =1):
        """
            Forward of Graph Mesh
            Deformation and Skinning
        """
        if deltaRs is None and deltaTs is None and skinRs is None and skinTs is None:
            B = B
        else:
            if not deltaRs is None:
                B = deltaRs.shape[0]
            if not deltaTs is None:
                B = deltaTs.shape[0]
            if not skinRs is None:
                B = skinRs.shape[0]
            if not skinTs is None:
                B = skinTs.shape[0]
        if deltaRs is None:
            deltaRs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if deltaTs is None:
            deltaTs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if skinRs is None:
            skinRs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if skinTs is None:
            skinTs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        vertsOfGraphOnBaseMesh = self.character.verts0[self.nodeIdxs.astype('int').tolist()]

        vertsOfGraphOnBaseMeshDeformed = vertsOfGraphOnBaseMesh.repeat(B,1,1) + deltaTs #B,NG,3
        R_skin_s = euler_angles_to_matrix(skinRs) #B,NG,3,3
        t_skin_s = skinTs #B,NG,3
        vertsOfGraphOnBaseMeshDeformedSkined = torch.matmul(R_skin_s, vertsOfGraphOnBaseMeshDeformed.reshape(B,self.numVertG, 3, 1))[:, :, :, 0] + \
                                 t_skin_s
        return vertsOfGraphOnBaseMeshDeformedSkined

    def inverseG(self, vertsOfGraphOnBaseMeshDeformedSkined, deltaRs = None, deltaTs = None, skinRs = None, skinTs = None, B =1):
        """
            Forward of Graph Mesh
            Deformation and Skinning
        """
        if deltaRs is None and deltaTs is None and skinRs is None and skinTs is None:
            B = B
        else:
            if not deltaRs is None:
                B = deltaRs.shape[0]
            if not deltaTs is None:
                B = deltaTs.shape[0]
            if not skinRs is None:
                B = skinRs.shape[0]
            if not skinTs is None:
                B = skinTs.shape[0]
        if deltaRs is None:
            deltaRs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if deltaTs is None:
            deltaTs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if skinRs is None:
            skinRs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if skinTs is None:
            skinTs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        # vertsOfGraphOnBaseMesh = self.character.verts0[self.nodeIdxs.astype('int').tolist()]

        R_skin_s = euler_angles_to_matrix(skinRs)  # B,NG,3,3
        R_skin_s_inv = R_skin_s.transpose(2,3)
        vertsOfGraphOnBaseMeshDeformed = torch.matmul(R_skin_s_inv, (vertsOfGraphOnBaseMeshDeformedSkined - skinTs).reshape(B, self.numVertG, 3, 1))[:, :, :, 0]
        vertsOfGraphOnBaseMeshRecovered = vertsOfGraphOnBaseMeshDeformed - deltaTs

        return vertsOfGraphOnBaseMeshRecovered


    def forwardF(self, deltaRs = None, deltaTs = None, skinRs = None, skinTs = None, displacements = None, Ts = None, B=1, returnNormal = False, returnTransformation = False, returnVertices = True):
        """
            Forward of Fine-Grained Mesh
            Deformation and Skinning
        """
        if deltaRs is None and deltaTs is None and skinRs is None and skinTs is None:
            B = B
        else:
            if not deltaRs is None:
                B = deltaRs.shape[0]
            if not deltaTs is None:
                B = deltaTs.shape[0]
            if not skinRs is None:
                B = skinRs.shape[0]
            if not skinTs is None:
                B = skinTs.shape[0]
        if deltaRs is None:
            deltaRs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if deltaTs is None:
            deltaTs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if skinRs is None:
            skinRs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if skinTs is None:
            skinTs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        # if displacements is None:
        #     displacements = torch.zeros([B, self.character.numVert, 3]).float().to(self.device)

        ## Old embedded deformation
        # vertsOfBaseMesh = self.character.verts0
        # # vertsOfGraphOnBaseMesh = self.character.verts0[self.nodeIdxs.astype('int').tolist()]
        # deltaRsMat = euler_angles_to_matrix(deltaRs) # B, K, 3, 3
        # deltaRsWeightedMat = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B,1,1), deltaRsMat)#B,F,3,3
        # idenditryMat = torch.eye(3).repeat(B,self.numVertG, 1, 1).float().to(self.device)
        # residual_gk = torch.einsum('bfcm,bcmn->bfcn', self.vertsOfBaseMeshExpand.repeat(B,1,1,1),
        #                         (idenditryMat - deltaRsMat).transpose(2,3))
        #
        # residualWeighted_gk = torch.einsum('bfcm, bfc->bfm', residual_gk, self.connectionMat.repeat(B,1,1))
        #
        # residual = deltaTs
        # residualWeighted = self.connectionMat.repeat(B,1,1).matmul(residual) # B,F,K  * B,K,3 = B, F, 3
        # vertsOfBaseMeshDeformed = torch.matmul(deltaRsWeightedMat, vertsOfBaseMesh.reshape(1, self.character.numVert, 3, 1))[
        #                           :, :, :, 0] + \
        #                           residualWeighted + residualWeighted_gk
        # # vertsOfBaseMeshDeformed = torch.matmul(deltaRsWeightedMat, vertsOfBaseMesh.reshape(1, self.character.numVert, 3, 1))[
        # #                           :, :, :, 0] + \
        # #                           residualWeighted
        #
        # # vertsOfBaseMeshDeformed = torch.matmul(deltaRsWeightedMat, vertsOfBaseMesh.reshape(1, self.character.numVert, 3, 1))[
        # #                           :, :, :, 0]
        #
        # # vertsOfBaseMeshDeformed = vertsOfBaseMesh.repeat(B,1,1)

        ## New embedded deformation
        vertsOfBaseMesh = self.character.verts0
        deltaRsMat = euler_angles_to_matrix(deltaRs) # B, K, 3, 3
        # deltaRsWeightedMat = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B,1,1), deltaRsMat)#B,F,3,3
        deltaRsWeightedMat = torch.einsum('bvk,bkmn->bvmn', self.connectionMat[None], deltaRsMat)#B,F,3,3
        # idenditryMat = torch.eye(3).repeat(B,self.numVertG, 1, 1).float().to(self.device)
        idenditryMat = torch.eye(3).repeat(1, 1, 1, 1).float().to(self.device)
        # residual_gk = torch.einsum('bfcm,bcmn->bfcn', self.vertsOfBaseMeshExpand.repeat(B,1,1,1),  (idenditryMat - deltaRsMat).transpose(2,3))
        
        # residual_gk = []
        # for i in tqdm(range(B)):
        #     residual_gk.append(torch.einsum('fcm,cmn->fcn', self.vertsOfBaseMeshExpand,  (idenditryMat - deltaRsMat).transpose(2,3)[i] ))
        # residual_gk  = torch.stack(residual_gk, dim=0)
        residual_gk = torch.einsum('bfcm,bcmn->bfcn', self.vertsOfBaseMeshExpand.repeat(1,1,1,1),  (idenditryMat - deltaRsMat).transpose(2,3))


        # residualWeighted_gk = torch.einsum('bfcm, bfc->bfm', residual_gk, self.connectionMat.repeat(B,1,1))
        residualWeighted_gk = torch.einsum('bfcm, bfc->bfm', residual_gk, self.connectionMat.repeat(1,1,1))


        # residual = deltaTs
        # residualWeighted = self.connectionMat.repeat(B,1,1).matmul(residual) # B,F,K  * B,K,3 = B, F, 3
        
        # residual = deltaTs
        residualWeighted = self.connectionMat.repeat(1,1,1).matmul(deltaTs) # B,F,K  * B,K,3 = B, F, 3

        transformDeformationMatWeighted = torch.eye(4).float().to(self.device).repeat(B, self.character.numVert, 1, 1)
        transformDeformationMatWeighted[:, :, :3, :3] = deltaRsWeightedMat
        transformDeformationMatWeighted[:, :, :3, 3] = residualWeighted + residualWeighted_gk
        # transformDeformationMatWeightedInv = torch.linalg.inv(transformDeformationMatWeighted)
        # vertsOfBaseMeshRecovered = torch.matmul(transformDeformationMatWeightedInv[:,:,:3,:3], (vertsOfBaseMeshDeformed).reshape(B, self.character.numVert, 3, 1))[:, :, :, 0] + \
        #                           transformDeformationMatWeightedInv[:,:,:3,3]
        if returnVertices:
            vertsOfBaseMeshDeformed = torch.matmul(transformDeformationMatWeighted[:,:,:3,:3], vertsOfBaseMesh.reshape(1, self.character.numVert, 3, 1))[
                                      :, :, :, 0] + \
                                      transformDeformationMatWeighted[:, :, :3, 3]

            if not displacements is None:
                vertsOfBaseMeshDeformed = vertsOfBaseMeshDeformed + displacements
                transformDisplacement = torch.eye(4).float().to(self.device).repeat(B, self.character.numVert, 1, 1)
                transformDisplacement[:, :, :3, 3] = displacements
                transformDeformationMatWeighted =   transformDisplacement @ transformDeformationMatWeighted
        ## Old Forward Kinematics
        # skinRsMat = euler_angles_to_matrix(skinRs)
        # # idenditryMat = np.tile(np.eye(3), (self.numVertG, 1, 1))
        # # residual = np.zeros([vertsOfGraphOnBaseMesh.shape[0], 3])
        # residual = skinTs
        # residualWeighted = self.connectionMat.repeat(B,1,1).matmul(residual) # B,F,K  * B,K,3 = B, F, 3
        # # residualWeighted = self.connectionMat.dot(residual)
        # # deltaRsWeightedMat = np.einsum('bk,kmn->bmn', self.connectionMat, deltaRsMat)
        # skinRsWeightedMat = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B,1,1), skinRsMat)#B,F,3,3
        # vertsOfBaseMeshDeformedSkined = torch.matmul(skinRsWeightedMat, vertsOfBaseMeshDeformed.reshape(B, self.character.numVert, 3, 1))[
        #                           :, :, :, 0] + \
        #                           residualWeighted
        # # vertsOfBaseMeshDeformedSkined = torch.matmul(skinRsWeightedMat, vertsOfBaseMeshDeformed.reshape(B, self.character.numVert, 3, 1))[
        # #                           :, :, :, 0]


        if Ts is None:
            ## New Forward Kinematics
            skinRsMat = euler_angles_to_matrix(skinRs) #B,G,3,3
            # residual = skinTs # B,G,3
            transformMat = torch.eye(4).float().to(self.device).repeat(B, self.numVertG, 1, 1)
            transformMat[:,:,:3,:3] = skinRsMat
            transformMat[:, :, :3, 3] = skinTs
            # transformMatWeighted = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B, 1, 1), transformMat)  # B, F, 4, 4
            # transformMatWeightedInv = torch.linalg.inv(transformMatWeighted)



            transformMatDQ = fromTransformation2VectorTorch(transformMat.reshape(-1,4,4)).reshape(B,-1,8)  # B*N, 8

            _, indices = torch.max(self.connectionMat, 1)
            nodeDQTransformationsInit = transformMatDQ[:, indices, :4].clone()
            temp = torch.einsum('bne,bem->bnm', nodeDQTransformationsInit,
                                transformMatDQ[:, :, :4].clone().transpose(1, 2))
            nodeDQSigns = torch.where(temp > 0, 1, -1).float().to(self.device)

            # nodeDQSigns = nodeDQSigns[:, indices] * nodeDQSigns.transpose(1,2)

            # for i in range(nodeSkinWeight.shape[1]):
            #     nodeDQTransformations[:, i] *= nodeDQSigns[:, i]
            # nodeDQTransformations *= nodeDQSigns
            # nodeVertDQTransformations = torch.matmul(nodeSkinWeight.repeat(self.B, 1, 1) * nodeDQSigns,
            #                                          nodeDQTransformations)

            transformMatDQWeighted = torch.einsum('bvk,bkm->bvm', self.connectionMat.repeat(B, 1, 1) * nodeDQSigns, transformMatDQ)  # B, F, 4, 4
            transformMatWeighted = fromVector2TransformationTorch(transformMatDQWeighted.reshape(-1,8)).reshape(B,-1,4,4)

        else:
            transformMatWeighted = Ts

        if returnVertices:
            vertsOfBaseMeshDeformedSkined = torch.matmul(transformMatWeighted[:,:,:3,:3], (vertsOfBaseMeshDeformed).reshape(B, self.character.numVert, 3, 1))[:, :, :, 0] + \
                                      transformMatWeighted[:,:,:3,3]
            # vertsOfBaseMeshDeformedSkined += displacements
        else:
            vertsOfBaseMeshDeformedSkined = None

        if returnNormal:
            normals = torch.zeros_like(vertsOfBaseMeshDeformedSkined)  # B,N,3
            tris = vertsOfBaseMeshDeformedSkined[:, self.character.faces]  # B,F,N,3
            n = torch.cross(tris[:, :, :, 1] - tris[:, :, :, 0], tris[:, :, :, 2] - tris[:, :, :, 0])
            n = torch.nn.functional.normalize(n, dim=-1)
            normals[:, self.character.faces[:, 0]] += n
            normals[:, self.character.faces[:, 1]] += n
            normals[:, self.character.faces[:, 2]] += n
            normals = torch.nn.functional.normalize(normals, dim=-1)
            return vertsOfBaseMeshDeformedSkined, normals

        if returnTransformation:
            return vertsOfBaseMeshDeformedSkined, transformMatWeighted, transformDeformationMatWeighted
        # else:
        return vertsOfBaseMeshDeformedSkined

    def inverseF(self, vertsOfBaseMeshDeformedSkined, deltaRs=None, deltaTs=None, skinRs=None, skinTs=None, B=1, returnNormal=False):
        """
            Forward of Fine-Grained Mesh
            Deformation and Skinning
        """
        if deltaRs is None and deltaTs is None and skinRs is None and skinTs is None:
            B = B
        else:
            if not deltaRs is None:
                B = deltaRs.shape[0]
            if not deltaTs is None:
                B = deltaTs.shape[0]
            if not skinRs is None:
                B = skinRs.shape[0]
            if not skinTs is None:
                B = skinTs.shape[0]
        if deltaRs is None:
            deltaRs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if deltaTs is None:
            deltaTs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if skinRs is None:
            skinRs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if skinTs is None:
            skinTs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)



        skinRsMat = euler_angles_to_matrix(skinRs) #B,G,3,3
        # residual = skinTs # B,G,3
        transformMat = torch.eye(4).float().to(self.device).repeat(B, self.numVertG, 1, 1)
        transformMat[:,:,:3,:3] = skinRsMat
        transformMat[:, :, :3, 3] = skinTs
        transformMatWeighted = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B, 1, 1), transformMat)  # B, F, 4, 4
        transformMatWeightedInv = torch.linalg.inv(transformMatWeighted)
        vertsOfBaseMeshDeformed = torch.matmul(transformMatWeightedInv[:,:,:3,:3], (vertsOfBaseMeshDeformedSkined).reshape(B, self.character.numVert, 3, 1))[:, :, :, 0] + \
                                  transformMatWeightedInv[:,:,:3,3]


        ## Inverse Embeded Graph Deformation
        # deltaRsMat = euler_angles_to_matrix(deltaRs)  # B, K, 3, 3
        # # residual = deltaTs
        # transformDeformationMat = torch.eye(4).float().to(self.device).repeat(B, self.numVertG, 1, 1)
        # transformDeformationMat[:,:,:3,:3] = deltaRsMat
        # transformDeformationMat[:, :, :3, 3] = deltaTs + self.vertsOfBaseMeshExpand +
        #
        # transformDeformationMatWeighted = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B, 1, 1), transformDeformationMat)  # B, F, 4, 4
        # transformDeformationMatWeightedInv = torch.linalg.inv(transformDeformationMatWeighted)



        vertsOfBaseMesh = self.character.verts0
        deltaRsMat = euler_angles_to_matrix(deltaRs) # B, K, 3, 3
        deltaRsWeightedMat = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B,1,1), deltaRsMat)#B,F,3,3
        idenditryMat = torch.eye(3).repeat(B,self.numVertG, 1, 1).float().to(self.device)
        residual_gk = torch.einsum('bfcm,bcmn->bfcn', self.vertsOfBaseMeshExpand.repeat(B,1,1,1),
                                (idenditryMat - deltaRsMat).transpose(2,3))

        residualWeighted_gk = torch.einsum('bfcm, bfc->bfm', residual_gk, self.connectionMat.repeat(B,1,1))

        residual = deltaTs
        residualWeighted = self.connectionMat.repeat(B,1,1).matmul(residual) # B,F,K  * B,K,3 = B, F, 3

        transformDeformationMatWeighted = torch.eye(4).float().to(self.device).repeat(B, self.character.numVert, 1, 1)
        transformDeformationMatWeighted[:, :, :3, :3] = deltaRsWeightedMat
        transformDeformationMatWeighted[:, :, :3, 3] = residualWeighted + residualWeighted_gk
        transformDeformationMatWeightedInv = torch.linalg.inv(transformDeformationMatWeighted)
        vertsOfBaseMeshRecovered = torch.matmul(transformDeformationMatWeightedInv[:,:,:3,:3], (vertsOfBaseMeshDeformed).reshape(B, self.character.numVert, 3, 1))[:, :, :, 0] + \
                                  transformDeformationMatWeightedInv[:,:,:3,3]



        # vertsOfBaseMeshDeformed = torch.matmul(deltaRsWeightedMat, vertsOfBaseMesh.reshape(1, self.character.numVert, 3, 1))[
        #                           :, :, :, 0] + \
        #                           residualWeighted + residualWeighted_gk




        # residualWeighted = self.connectionMat.repeat(B, 1, 1).matmul(residual)  # B,F,K  * B,K,3 = B, F, 3
        # skinRsWeightedMat = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B, 1, 1), skinRsMat)  # B,F,3,3
        # TransformMat = torch.eye(4).float().to(self.device).repeat(B, F, 1, 1)
        # TransformMatInv =
        # skinRsWeightedMatInv = skinRsWeightedMat.transpose(2, 3)
        # vertsOfBaseMeshDeformed = torch.matmul(skinRsWeightedMatInv, (vertsOfBaseMeshDeformedSkined - residualWeighted).reshape(B, self.character.numVert, 3, 1))[:, :, :, 0]

                                  # :, 0]
        # vertsOfBaseMeshDeformed = (vertsOfBaseMeshDeformedSkined - residualWeighted).reshape(B, self.character.numVert, 3, 1)[:, :, :, 0]




        # vertsOfBaseMeshDeformedSkined = torch.matmul(skinRsWeightedMat,
        #                                              vertsOfBaseMeshDeformed.reshape(B, self.character.numVert, 3,
        #                                                                              1))[
        #                                 :, :, :, 0] + \
        #                                 residualWeighted
        #
        # R_skin_s = euler_angles_to_matrix(skinRs)  # B,NG,3,3
        # R_skin_s_inv = R_skin_s.transpose(2,3)
        # vertsOfGraphOnBaseMeshDeformed = torch.matmul(R_skin_s_inv, (vertsOfGraphOnBaseMeshDeformedSkined - skinTs).reshape(B, self.numVertG, 3, 1))[:, :, :, 0]


        # return vertsOfBaseMeshDeformed
        return vertsOfBaseMeshRecovered

    def inverseFPts(self, pts, idx=None, deltaRs=None, deltaTs=None, skinRs=None, skinTs=None, B=1,
                 returnNormal=False):
        """
            Forward of Fine-Grained Mesh
            Deformation and Skinning
        """
        if deltaRs is None and deltaTs is None and skinRs is None and skinTs is None:
            B = B
        else:
            if not deltaRs is None:
                B = deltaRs.shape[0]
            if not deltaTs is None:
                B = deltaTs.shape[0]
            if not skinRs is None:
                B = skinRs.shape[0]
            if not skinTs is None:
                B = skinTs.shape[0]
        if deltaRs is None:
            deltaRs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if deltaTs is None:
            deltaTs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if skinRs is None:
            skinRs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)
        if skinTs is None:
            skinTs = torch.zeros([B, self.numVertG, 3]).float().to(self.device)

        skinRsMat = euler_angles_to_matrix(skinRs)  # B,G,3,3
        # residual = skinTs # B,G,3
        transformMat = torch.eye(4).float().to(self.device).repeat(B, self.numVertG, 1, 1)
        transformMat[:, :, :3, :3] = skinRsMat
        transformMat[:, :, :3, 3] = skinTs
        transformMatWeighted = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B, 1, 1),
                                            transformMat)  # B, F, 4, 4
        transformMatWeightedInv = torch.linalg.inv(transformMatWeighted)  # B, F, 4, 4
        # transformMatWeightedInv = transformMatWeightedInv[]
        if not idx is None:
            transformMatWeightedInv = batched_index_select(transformMatWeightedInv, dim=1, index=idx)
        vertsOfBaseMeshDeformed = torch.matmul(transformMatWeightedInv[:, :, :3, :3],
                                               (pts).reshape(B, -1, 3, 1))[:, :, :, 0] + \
                                  transformMatWeightedInv[:, :, :3, 3]

        ## Inverse Embeded Graph Deformation
        # deltaRsMat = euler_angles_to_matrix(deltaRs)  # B, K, 3, 3
        # # residual = deltaTs
        # transformDeformationMat = torch.eye(4).float().to(self.device).repeat(B, self.numVertG, 1, 1)
        # transformDeformationMat[:,:,:3,:3] = deltaRsMat
        # transformDeformationMat[:, :, :3, 3] = deltaTs + self.vertsOfBaseMeshExpand +
        #
        # transformDeformationMatWeighted = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B, 1, 1), transformDeformationMat)  # B, F, 4, 4
        # transformDeformationMatWeightedInv = torch.linalg.inv(transformDeformationMatWeighted)

        vertsOfBaseMesh = self.character.verts0
        deltaRsMat = euler_angles_to_matrix(deltaRs)  # B, K, 3, 3
        deltaRsWeightedMat = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B, 1, 1),
                                          deltaRsMat)  # B,F,3,3
        idenditryMat = torch.eye(3).repeat(B, self.numVertG, 1, 1).float().to(self.device)
        residual_gk = torch.einsum('bfcm,bcmn->bfcn', self.vertsOfBaseMeshExpand.repeat(B, 1, 1, 1),
                                   (idenditryMat - deltaRsMat).transpose(2, 3))

        residualWeighted_gk = torch.einsum('bfcm, bfc->bfm', residual_gk, self.connectionMat.repeat(B, 1, 1))

        residual = deltaTs
        residualWeighted = self.connectionMat.repeat(B, 1, 1).matmul(residual)  # B,F,K  * B,K,3 = B, F, 3

        transformDeformationMatWeighted = torch.eye(4).float().to(self.device).repeat(B, self.character.numVert, 1,
                                                                                      1)
        transformDeformationMatWeighted[:, :, :3, :3] = deltaRsWeightedMat
        transformDeformationMatWeighted[:, :, :3, 3] = residualWeighted + residualWeighted_gk
        transformDeformationMatWeightedInv = torch.linalg.inv(transformDeformationMatWeighted)
        if not idx is None:
            transformDeformationMatWeightedInv = batched_index_select(transformDeformationMatWeightedInv, dim=1, index=idx)
        vertsOfBaseMeshRecovered = torch.matmul(transformDeformationMatWeightedInv[:, :, :3, :3],
                                                (vertsOfBaseMeshDeformed).reshape(B, -1, 3, 1))[
                                   :, :, :, 0] + \
                                   transformDeformationMatWeightedInv[:, :, :3, 3]

        # vertsOfBaseMeshDeformed = torch.matmul(deltaRsWeightedMat, vertsOfBaseMesh.reshape(1, self.character.numVert, 3, 1))[
        #                           :, :, :, 0] + \
        #                           residualWeighted + residualWeighted_gk

        # residualWeighted = self.connectionMat.repeat(B, 1, 1).matmul(residual)  # B,F,K  * B,K,3 = B, F, 3
        # skinRsWeightedMat = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B, 1, 1), skinRsMat)  # B,F,3,3
        # TransformMat = torch.eye(4).float().to(self.device).repeat(B, F, 1, 1)
        # TransformMatInv =
        # skinRsWeightedMatInv = skinRsWeightedMat.transpose(2, 3)
        # vertsOfBaseMeshDeformed = torch.matmul(skinRsWeightedMatInv, (vertsOfBaseMeshDeformedSkined - residualWeighted).reshape(B, self.character.numVert, 3, 1))[:, :, :, 0]

        # :, 0]
        # vertsOfBaseMeshDeformed = (vertsOfBaseMeshDeformedSkined - residualWeighted).reshape(B, self.character.numVert, 3, 1)[:, :, :, 0]

        # vertsOfBaseMeshDeformedSkined = torch.matmul(skinRsWeightedMat,
        #                                              vertsOfBaseMeshDeformed.reshape(B, self.character.numVert, 3,
        #                                                                              1))[
        #                                 :, :, :, 0] + \
        #                                 residualWeighted
        #
        # R_skin_s = euler_angles_to_matrix(skinRs)  # B,NG,3,3
        # R_skin_s_inv = R_skin_s.transpose(2,3)
        # vertsOfGraphOnBaseMeshDeformed = torch.matmul(R_skin_s_inv, (vertsOfGraphOnBaseMeshDeformedSkined - skinTs).reshape(B, self.numVertG, 3, 1))[:, :, :, 0]

        # return vertsOfBaseMeshDeformed
        return vertsOfBaseMeshRecovered




        # vertsOfBaseMesh = self.character.verts0
        # vertsOfGraphOnBaseMesh = self.character.verts0[self.nodeIdxs.astype('int').tolist()]
        # deltaRsMat = euler_angles_to_matrix(deltaRs)  # B, K, 3, 3
        # deltaRsWeightedMat = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B, 1, 1),
        #                                   deltaRsMat)  # B,F,3,3
        # idenditryMat = torch.eye(3).repeat(B, self.numVertG, 1, 1).float().to(self.device)
        # residual_gk = torch.einsum('bfcm,bcmn->bfcn', self.vertsOfBaseMeshExpand.repeat(B, 1, 1, 1),
        #                            (idenditryMat - deltaRsMat).transpose(2, 3))
        #
        # residualWeighted_gk = torch.einsum('bfcm, bfc->bfm', residual_gk, self.connectionMat.repeat(B, 1, 1))
        #
        # residual = deltaTs
        # residualWeighted = self.connectionMat.repeat(B, 1, 1).matmul(residual)  # B,F,K  * B,K,3 = B, F, 3
        # vertsOfBaseMeshDeformed = torch.matmul(deltaRsWeightedMat,
        #                                        vertsOfBaseMesh.reshape(1, self.character.numVert, 3, 1))[
        #                           :, :, :, 0] + \
        #                           residualWeighted + residualWeighted_gk
        # # vertsOfBaseMeshDeformed = torch.matmul(deltaRsWeightedMat, vertsOfBaseMesh.reshape(1, self.character.numVert, 3, 1))[
        # #                           :, :, :, 0] + \
        # #                           residualWeighted
        #
        # # vertsOfBaseMeshDeformed = torch.matmul(deltaRsWeightedMat, vertsOfBaseMesh.reshape(1, self.character.numVert, 3, 1))[
        # #                           :, :, :, 0]
        #
        # # vertsOfBaseMeshDeformed = vertsOfBaseMesh.repeat(B,1,1)
        #
        # skinRsMat = euler_angles_to_matrix(skinRs)
        # # idenditryMat = np.tile(np.eye(3), (self.numVertG, 1, 1))
        # # residual = np.zeros([vertsOfGraphOnBaseMesh.shape[0], 3])
        # residual = skinTs
        # residualWeighted = self.connectionMat.repeat(B, 1, 1).matmul(residual)  # B,F,K  * B,K,3 = B, F, 3
        # # residualWeighted = self.connectionMat.dot(residual)
        # # deltaRsWeightedMat = np.einsum('bk,kmn->bmn', self.connectionMat, deltaRsMat)
        # skinRsWeightedMat = torch.einsum('bvk,bkmn->bvmn', self.connectionMat.repeat(B, 1, 1), skinRsMat)  # B,F,3,3
        # vertsOfBaseMeshDeformedSkined = torch.matmul(skinRsWeightedMat,
        #                                              vertsOfBaseMeshDeformed.reshape(B, self.character.numVert, 3,
        #                                                                              1))[
        #                                 :, :, :, 0] + \
        #                                 residualWeighted
        #
        # if returnNormal:
        #     normals = torch.zeros_like(vertsOfBaseMeshDeformedSkined)  # B,N,3
        #     tris = vertsOfBaseMeshDeformedSkined[:, self.character.faces]  # B,F,N,3
        #     n = torch.cross(tris[:, :, :, 1] - tris[:, :, :, 0], tris[:, :, :, 2] - tris[:, :, :, 0])
        #     n = torch.nn.functional.normalize(n, dim=-1)
        #     normals[:, self.character.faces[:, 0]] += n
        #     normals[:, self.character.faces[:, 1]] += n
        #     normals[:, self.character.faces[:, 2]] += n
        #     normals = torch.nn.functional.normalize(normals, dim=-1)
        #     return vertsOfBaseMeshDeformedSkined, normals
        # else:
        #     return vertsOfBaseMeshDeformedSkined










        # vertsOfGraphOnBaseMeshDeformed = vertsOfGraphOnBaseMesh.repeat(B,1,1) + deltaTs #B,NG,3
        # R_skin_s = euler_angles_to_matrix(skinRs) #B,NG,3,3
        # t_skin_s = skinTs #B,NG,3
        # vertsOfGraphOnBaseMeshDeformedSkined = torch.matmul(R_skin_s, vertsOfGraphOnBaseMeshDeformed.reshape(B,self.numVertG, 3, 1))[:, :, :, 0] + \
        #                          t_skin_s
        # return vertsOfGraphOnBaseMeshDeformedSkined

    def getMeshG(self, verts, idx = 0):
        vertsG = to_cpu(verts[idx])
        mesh = trimesh.Trimesh(vertices=vertsG, faces=self.facesG, process=False)
        return mesh

    def getMeshF(self, verts, idx=0):
        vertsF = to_cpu(verts[idx])
        mesh = trimesh.Trimesh(vertices=vertsF, faces=self.character.faces, process=False)
        return mesh

    # def getGraphMesh(self):
    #     vertsG = np.matmul(self.nodeVertTransformations[:, :3, :3], self.vertsG0.reshape(self.vertsG0.shape[0], 3, 1))[
    #              :, :, 0] + \
    #              self.nodeVertTransformations[:, :3, 3]
    #     mesh = trimesh.Trimesh(vertices=vertsG, faces=self.facesG)
    #     return mesh

    # def getDeformedGraphMesh(self, deltaRs=None, deltaTs=None, dofParams=None):
    #     if not dofParams is None:
    #         self.updateDof(dofParams)
    #     # self.nodeIdxs
    #     vertsOfGraphOnBaseMesh = self.character.verts0[self.nodeIdxs.astype('int').tolist()]
    #     if not deltaTs is None:
    #         vertsOfGraphOnBaseMesh += deltaTs
    #     R_skin_s = euler(self.nodeRs)
    #     t_skin_s = self.nodeTs
    #     vertsOfGraphOnBaseMesh = np.matmul(R_skin_s, vertsOfGraphOnBaseMesh.reshape(self.numVertG, 3, 1))[:, :, 0] + \
    #                              t_skin_s
    #     mesh = trimesh.Trimesh(vertices=vertsOfGraphOnBaseMesh, faces=self.facesG)
    #     return mesh

    def getDeformedMesh(self, deltaRs=None, deltaTs=None, dofParams=None):
        if not dofParams is None:
            self.updateDof(dofParams)
        if deltaRs is None:
            deltaRs = np.zeros([self.numVertG, 3])
        if deltaTs is None:
            deltaTs = np.zeros([self.numVertG, 3])

        vertsOfBaseMesh = self.character.verts0.copy()
        vertsOfGraphOnBaseMesh = self.character.verts0[self.nodeIdxs.astype('int').tolist()]
        deltaRsMat = euler(deltaRs)
        deltaRsWeightedMat = np.einsum('bk,kmn->bmn', self.connectionMat, deltaRsMat)
        idenditryMat = np.tile(np.eye(3), (self.numVertG, 1, 1))
        residual_gk = np.einsum('fcm,cmn->fcn', self.vertsOfBaseMeshExpand,
                                (idenditryMat - deltaRsMat).transpose(0, 2, 1))

        residualWeighted_gk = np.einsum('fcm,fc->fm', residual_gk, self.connectionMat)

        residual = deltaTs
        residualWeighted = self.connectionMat.dot(residual)
        vertsOfBaseMeshDeformed = np.matmul(deltaRsWeightedMat, vertsOfBaseMesh.reshape(self.character.numVert, 3, 1))[
                                  :, :, 0] + \
                                  residualWeighted + residualWeighted_gk

        deltaRsMat = euler(self.nodeRs)
        idenditryMat = np.tile(np.eye(3), (self.numVertG, 1, 1))
        residual = np.zeros([vertsOfGraphOnBaseMesh.shape[0], 3])
        residual = self.nodeTs
        residualWeighted = self.connectionMat.dot(residual)
        deltaRsWeightedMat = np.einsum('bk,kmn->bmn', self.connectionMat, deltaRsMat)
        vertsOfBaseMeshDeformedSkined = np.matmul(deltaRsWeightedMat,
                                                  vertsOfBaseMeshDeformed.reshape(self.character.numVert, 3, 1))[:, :,
                                        0] + \
                                        residualWeighted

        mesh = trimesh.Trimesh(vertices=vertsOfBaseMeshDeformedSkined, faces=self.character.faces)
        return mesh

class SkinnedCharacter(nn.Module):
    def __init__(self, charPath = None, skinPath=None, meshPath=None, motionPath=None, \
                 skeleton=None, skelPath=None, motionBasePath=None, useDQ = True, verbose=False, device='cpu',
                 segWeightsFlag=False, computeAdjacencyFlag=False,):
        super(SkinnedCharacter, self).__init__()
        self.verbose = verbose
        self.charPath = charPath
        self.motionPath = motionPath
        if not charPath is None:
            self.skelPath, self.meshPath, self.skinPath, self.motionBasePath = loadChar(charPath)
        else:
            self.skinPath, self.meshPath = skinPath, meshPath
            self.skelPath, self.motionBasePath = skelPath, motionBasePath
        if self.verbose:
            print("-- charPath: ", self.charPath)
            print("-- skinPath: ", self.skinPath)
            print("-- meshPath: ", self.meshPath)
            print("-- skelPath: ", self.skelPath)
            print("-- motionBasePath: ", self.motionBasePath)
            print("-- motionPath: ", self.motionPath)
        self.useDQ = useDQ

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.segWeightsFlag = segWeightsFlag
        self.computeAdjacencyFlag = computeAdjacencyFlag


        if skeleton is None:
            self.skeleton = Skeleton(self.skelPath, verbose=verbose, useDQ=useDQ, device=self.device)
        else:
            self.skeleton = skeleton
        assert self.skeleton.useDQ == useDQ
        assert self.skeleton.device == device

        self.numDof = self.skeleton.numDof
        # print("self.numDof: ", self.numDof)
        # exit()
        self.numJoint = self.skeleton.numJoint
        self.numMarker = self.skeleton.numMarker
        self.numMotion = -1

        self.skinWeight = None
        self.jointTransformations = None
        self.initialTransformations = None
        self.vertTransformations = None

        self.loadBaseMotion()
        self.loadSkin()
        self.loadMesh()
        if not motionPath is None:
            self.loadMotion()

        self.loadHandSeg()

    def loadBaseMotion(self):
        if self.verbose:
            print('-- update base motion to skeleton')
        self.motion_base = loadMotion(self.motionBasePath,returnIdx=False,returnTensor=True,device=self.device)
        self.skeleton.update(self.motion_base)

    def loadMotion(self):
        self.motionList = loadMotion(self.motionPath, returnIdx=False, returnTensor=True,device=self.device)
        self.numMotion = self.motionList.shape[0]
        if self.verbose:
            print('-- loaded motion has {} frames'.format(self.numMotion))

    def loadSkin(self):
        self.skinBoneNames, self.skinWeight, self.skinFirstIdxs = loadSkin(self.skinPath, returnTensor=True, device=self.device)
        self.skinBoneIdxs = []
        # print()
        for item in self.skinBoneNames:
            for i in np.flip(np.arange(0, len(self.skeleton.jointNames))):
                joint_name = self.skeleton.jointNames[i]
                # if item == 'GlobalScale':
                #     if joint_name[:len(item)] == item:
                #         self.skinBoneIdxs.append(i)
                #         break
                # else:
                #     # if joint_name[:len(item)] == item and not 'Scale' in item and not 'Roll' in item:
                #     if joint_name[:len(item)] == item and not 'Scale' in joint_name and not 'Roll' in joint_name:
                #         self.skinBoneIdxs.append(i)
                #         break

                if joint_name[:len(item)] == item:
                    if len(joint_name) > len(item):
                        if joint_name[len(item)] == '_':
                            self.skinBoneIdxs.append(i)
                            break
                    else:
                        self.skinBoneIdxs.append(i)
                        break

        self.numBone = len(self.skinBoneIdxs)
        if self.verbose:
            print("-- loaded skin with {} bones idxs: ".format(self.numBone), self.skinBoneIdxs)

        charJointtransformations = []
        charInitialtransformations = []
        for i in range(0, len(self.skinBoneIdxs)):
            sc = 1.0
            boneIdx = self.skinBoneIdxs[i]
            charJointtransformations.append(self.skeleton.jointGlobalTransformations[:, boneIdx])
            charInitialtransformations.append(torch.linalg.inv(self.skeleton.jointGlobalTransformations[:, boneIdx]))
        # self.jointTransformations = np.array(charJointtransformations)
        self.jointTransformations = torch.stack(charJointtransformations, dim=1)
        self.initialTransformations = torch.stack(charInitialtransformations, dim=1)
        # self.jointTransformations = torch.matmul(self.jointTransformations, self.initialTransformations)
        self.jointTransformations = torch.einsum('bdmn, bdnk -> bdmk', self.jointTransformations, self.initialTransformations)



    def loadHandSeg(self):
        # self.vertexLabels2 = []
        # self.vertexLabels2Nunmpy = np.array(self.vertexLabels2)
        # verts = self.verts0.data.cpu().numpy().reshape(-1, 3)
        # handIdxs = []
        # handIdxs.append( self.skeleton.jointNames.index('LeftHandEE') )
        # handIdxs.append( self.skeleton.jointNames.index('RightHandEE') )
        # handjoints = self.skeleton.jointGlobalTransformations[0, handIdxs,:3,3]
        # import pdb
        # pdb.set_trace()

        handIdxs = []
        handIdxs.append( self.skinBoneNames.index('LeftHandEE') )
        handIdxs.append( self.skinBoneNames.index('RightHandEE') )
        handIdxs.append( self.skinBoneNames.index('LeftHand') )
        handIdxs.append( self.skinBoneNames.index('RightHand') )
        hand_verts_flag = np.max(self.skinWeight.data.cpu().numpy()[:,handIdxs], 1) > 0.01  # (n_verts,)

        self.vertexLabels2Nunmpy = np.zeros(self.numVert)
        self.vertexLabels2Nunmpy[hand_verts_flag]  = 1
        # save_ply( '/CT/HOIMOCAP3/nobackup/t2.ply'  , verts[hand_verts_idx])
        # jointGlobalTransformations
        self.labelMatrix2 = np.zeros([self.numVert, self.numVert])
        # import pdb
        # pdb.set_trace()
        for i in range(self.numVert):
            self.labelMatrix2[i, np.where( self.vertexLabels2Nunmpy == self.vertexLabels2Nunmpy[i])[0]] = 1

    #     if hand_verts_flag[i]:
        #         # self.labelMatrix2[i, np.where( self.vertexLabels2Nunmpy == self.vertexLabels2Nunmpy[i])[0]] = 1
        #         self.labelMatrix2[i, i] = 1

    def loadMesh(self):
        # load mesh
        self.verts0, self.faces = load_obj_mesh(self.meshPath)
        # self.faces = self.faces.astype(np.int32)
        self.verts0 = to_tensorFloat(self.verts0).to(self.device)
        self.verts0Source = to_tensorFloat(self.verts0).to(self.device).clone()
        # self.faces = to_tensor(self.faces).to(self.device)
        self.numVert = self.verts0.shape[0]
        self.numFace = self.faces.shape[0]

        if self.verbose:
            print('-- loaded mesh has {} verts {} faces'.format(self.numVert, self.numFace))
        self.computeNeighbours()
        if self.segWeightsFlag:
            self.loadSegmentationWeights()
        self.computeAdjacency()
        nrAllConnections = 0
        for i in range(self.numVert):
            nrAllConnections += len(self.neighbours[i])
        self.numEdge = int(nrAllConnections / 2)

        # self.numNeighbours = np.zeros(self.numVert, dtype = torch.int).to(self.device)
        # self.neighbourOffsets = np.zeros(self.numVert + 1, dtype = torch.int).to(self.device)
        # self.neighbourIdxs = np.zeros(2 * self.numEdge, dtype = torch.int).to(self.device)
        self.numNeighbours = np.zeros(self.numVert)
        self.neighbourOffsets = np.zeros(self.numVert + 1)
        self.neighbourIdxs = np.zeros(2 * self.numEdge)
        count = 0
        offset = 0
        self.neighbourOffsets[0] = 0
        for i in range(self.numVert):
            pass
            valance = len(self.neighbours[i])
            self.numNeighbours[count] = valance
            for j in range(valance):
                self.neighbourIdxs[offset] = self.neighbours[i][j]
                offset += 1
            self.neighbourOffsets[count + 1] = offset
            count += 1

    def loadSegmentationWeights(self):

        self.vertexLabels = []
        self.vertexWeights = []
        self.legMask = []
        self.headMask = []

        # labels
        # segmentationFile = open(self.folderPath + 'segmentation.txt')
        # print(segmentationFile)
        with open(osp.join( osp.dirname(self.skinPath), 'segmentation.txt' ), 'r') as f:
            data = f.readlines()
            for line in data:
                # print(line, end='')
                splitted = line.split()
                if len(splitted) > 0:
                    self.vertexLabels.append(int(splitted[0]))

        if (len(self.vertexLabels) != self.numVert):
            print('VERTICES AND LABELS NOT THE SAME RANGE!')
            print(' Labels ' + str(len(self.vertexLabels)) + ' vs. Vertices ' + str(self.numberOfVertices))



        self.vertexLabelsNunmpy = np.array(self.vertexLabels)
        self.labelMatrix = np.zeros([self.numVert, self.numVert])
        for i in range(self.numVert):
            self.labelMatrix[i, np.where( self.vertexLabelsNunmpy == self.vertexLabelsNunmpy[i])[0]] = 1



        # weights
        for v in range(0, len(self.vertexLabels)):
            label = self.vertexLabels[v]

            if label == 16 or label == 17:
                self.legMask.append(0.0)
            else:
                self.legMask.append(1.0)

            # background / dress / coat / jumpsuit / skirt
            if (label == 0 or label == 6 or label == 7 or label == 10 or label == 12):
                self.vertexWeights.append(1.0)
            # upper clothes
            elif (label == 5):
                self.vertexWeights.append(1.0)
            # pants
            elif (label == 9):
                # self.vertexWeights.append(2.0)
                self.vertexWeights.append(2.0)
            # scarf / socks
            elif (label == 11 or label == 8):
                self.vertexWeights.append(1.0)
            # skins
            elif (label == 14 or label == 15):  # arm
                self.vertexWeights.append(5.0)
                # self.vertexWeights.append(0.0)
            elif (label == 16 or label == 17):  # leg
                self.vertexWeights.append(5.0)
            # shoes / glove / sunglasses / hat
            elif (label == 18 or label == 19 or label == 1 or label == 3 or label == 4):
                self.vertexWeights.append(5.0)
            # hat / hair / face
            elif (label == 2 or label == 13):
                self.vertexWeights.append(400.0)
                self.headMask.append(1.0)
            # else
            else:
                self.vertexWeights.append(1.0)
                if not self.verbose:
                    print('Vertex %d has no valid label', v)

    
    def computeAdjacency(self):
        self.adjacency = np.zeros((self.numVert, self.numVert), dtype=np.float32)
        self.compressedAdjacency = [[] for _ in range(self.numVert)]
        self.numberOfEdges = 0
        self.numberOfNeigbours = np.zeros((self.numVert))
        for i in range(self.numFace):
            v0 = self.faces[i, 0]
            v1 = self.faces[i, 1]
            v2 = self.faces[i, 2]
            self.adjacency[v0, v1] = 1
            self.adjacency[v0, v2] = 1
            self.adjacency[v1, v0] = 1
            self.adjacency[v1, v2] = 1
            self.adjacency[v2, v0] = 1
            self.adjacency[v2, v1] = 1

            # v0
            if v1 + 1 not in self.compressedAdjacency[v0]:
                self.compressedAdjacency[v0].append(v1 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v0] = self.numberOfNeigbours[v0] + 1
            if v2 + 1 not in self.compressedAdjacency[v0]:
                self.compressedAdjacency[v0].append(v2 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v0] = self.numberOfNeigbours[v0] + 1
            # v1
            if v0 + 1 not in self.compressedAdjacency[v1]:
                self.compressedAdjacency[v1].append(v0 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v1] = self.numberOfNeigbours[v1] + 1
            if v2 + 1 not in self.compressedAdjacency[v1]:
                self.compressedAdjacency[v1].append(v2 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v1] = self.numberOfNeigbours[v1] + 1
            # v2
            if v0 + 1 not in self.compressedAdjacency[v2]:
                self.compressedAdjacency[v2].append(v0 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v2] = self.numberOfNeigbours[v2] + 1
            if v1 + 1 not in self.compressedAdjacency[v2]:
                self.compressedAdjacency[v2].append(v1 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v2] = self.numberOfNeigbours[v2] + 1
        self.compressedAdjacency = np.asarray(self.compressedAdjacency, dtype=object)
        self.maximumNumNeighbours = int(np.amax(self.numberOfNeigbours))
        self.laplacian = - self.adjacency
        for i in range(0, self.numVert):
            self.laplacian[i, i] = self.numberOfNeigbours[i]

        if self.segWeightsFlag:
            # row weight
            if self.verbose:
                print('     ++ Compute row weights')
            self.rowWeight = np.zeros((self.numVert), dtype=np.float32)

            for i in range(0, self.numVert):
                self.rowWeight[i] = 0.0
                for j in range(0, len(self.compressedAdjacency[i])):
                    nIdx = self.compressedAdjacency[i][j] - 1
                    self.rowWeight[i] = self.rowWeight[i] + (self.vertexWeights[i] + self.vertexWeights[nIdx]) / 2.0
                self.rowWeight[i] = self.rowWeight[i] / float(self.numberOfNeigbours[i])

            # laplacian weighted
            if self.verbose:
                print('     ++ Compute laplacian weights')
            self.adjacencyWeights = np.zeros((self.numVert, self.numVert))
            for i in range(self.numFace):
                v0 = self.faces[i, 0]
                v1 = self.faces[i, 1]
                v2 = self.faces[i, 2]

                self.adjacencyWeights[v0, v1] = (self.vertexWeights[v0] + self.vertexWeights[v1]) / 2.0
                self.adjacencyWeights[v0, v2] = (self.vertexWeights[v0] + self.vertexWeights[v2]) / 2.0
                self.adjacencyWeights[v1, v0] = (self.vertexWeights[v1] + self.vertexWeights[v0]) / 2.0
                self.adjacencyWeights[v1, v2] = (self.vertexWeights[v1] + self.vertexWeights[v2]) / 2.0
                self.adjacencyWeights[v2, v0] = (self.vertexWeights[v2] + self.vertexWeights[v0]) / 2.0
                self.adjacencyWeights[v2, v1] = (self.vertexWeights[v2] + self.vertexWeights[v1]) / 2.0


    def computeNeighbours(self):
        self.neighbours = []
        for i in range(self.numVert):
            self.neighbours.append([])
        for i in range(self.numFace):
            self.neighbours[self.faces[i, 0]].append(self.faces[i, 1])
            self.neighbours[self.faces[i, 0]].append(self.faces[i, 2])
            self.neighbours[self.faces[i, 1]].append(self.faces[i, 0])
            self.neighbours[self.faces[i, 1]].append(self.faces[i, 2])
            self.neighbours[self.faces[i, 2]].append(self.faces[i, 0])
            self.neighbours[self.faces[i, 2]].append(self.faces[i, 1])
        for i in range(self.numVert):
            self.neighbours[i] = list(set(self.neighbours[i]))

    def computeGeodesicDistance(self, idx=-1):
        distances = np.ones(self.numVert) * 1000000
        distances[idx] = 0

        import queue
        Q = queue.Queue()
        Q.put(idx)
        while not Q.empty():
            v = Q.get()
            for j in range(len(self.neighbours[v])):
                # print('v:{},j:{},value:{}'.format(v,j,self.neighbours[v][j]))
                # print(distancesH.shape)
                if distances[self.neighbours[v][j]] == 1000000:
                    Q.put(self.neighbours[v][j])
                    distances[self.neighbours[v][j]] = distances[v] + 1
        # print()
        return distances

    # For visualization
    def getMesh(self, idx=0, useDQ=None):
        """
            1. update/updateFrame
            2. getMesh
        """
        if not useDQ is None:
            useDQS_m = useDQ
        else:
            useDQS_m = self.useDQ

        self.vertTransformations = torch.zeros((self.numVert, 4, 4)).float().to(self.device)
        if not useDQS_m:
            self.vertTransformations = np.einsum('vd,dmn->vmn', self.skinWeight, self.jointTransformations[idx])
        else:
            # verts = np.zeros_like(self.verts0)
            self.DQTransformations = torch.zeros((1, self.skinWeight.shape[1], 8)).float().to(self.device)
            self.DQSigns = torch.ones((1, self.skinWeight.shape[1], 1)).float().to(self.device)
            for i in range(self.skinWeight.shape[1]):
                self.DQTransformations[:, i] = fromTransformation2VectorTorch(self.jointTransformations[idx:idx+1,i])
            for i in range(self.skinWeight.shape[1]):
                # self.DQSigns[:, i] = 1 if torch.dot(self.DQTransformations[:, 0, :4], self.DQTransformations[:, i, :4]) > 0 else -1
                self.DQSigns[:, i] = 1 if torch.einsum('bm,bm->b', self.DQTransformations[:, 0, :4], self.DQTransformations[:, i, :4]) > 0 else -1
            for i in range(self.skinWeight.shape[1]):
                self.DQTransformations[:,i] *= self.DQSigns[:,i]

            self.vertDQTransformations = torch.mm(self.skinWeight, self.DQTransformations[0])
            for i in range(self.skinWeight.shape[0]):
                self.vertTransformations[i] = fromVector2TransformationTorch(self.vertDQTransformations[i][None])[0]
        verts = torch.matmul(self.vertTransformations[:, :3, :3], self.verts0.reshape(self.verts0.shape[0], 3, 1))[:, :,
                0] + \
                self.vertTransformations[:, :3, 3]
        verts = to_cpu(verts)
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces)
        return mesh

    # For visualization
    def getSkelMesh(self, idx=0):
        """
            1. update/updateFrame
            2. getSkelMesh
        """
        return self.skeleton.getMesh(idx=idx)

    def updateFrame(self, frames : list):
        """
            Need motionPath to load all motions.
        """
        assert not self.motionPath is None
        self.skeleton.update(self.motionList[frames])
        charJointtransformations = []
        for i in range(0, len(self.skinBoneIdxs)):
            sc = 1.0
            boneIdx = self.skinBoneIdxs[i]
            charJointtransformations.append(self.skeleton.jointGlobalTransformations[boneIdx])
        self.jointTransformations = np.matmul(np.array(charJointtransformations), self.initialTransformations)

    def update(self, dofParams : torch.Tensor):
        """
            Input:
                dofParams ==> B,D / BatchSize, numDof
            Output:
                 self.jointTransformations ==> B, numBone
        """
        self.skeleton.update(dofParams)
        # charJointtransformations = []
        # for i in range(0, len(self.skinBoneIdxs)):
        #     sc = 1.0
        #     boneIdx = self.skinBoneIdxs[i]
        #     charJointtransformations.append(self.skeleton.jointGlobalTransformations[:, boneIdx])
        # self.jointTransformations = torch.stack(charJointtransformations, dim=1)

        self.jointGlobalTransformations = self.skeleton.jointGlobalTransformations[:, self.skinBoneIdxs]



        self.jointTransformations = torch.einsum('bdmn, bdnk -> bdmk', self.jointGlobalTransformations, self.initialTransformations)

    def forward(self, dofParams, useDQ=None):
        """
            Input:
                dofParams ==> B,D
                useDQ ==> bool, to use DQS or LBS locally
            Output:
                 verts ==> B, numVert, 3
        """
        if not useDQ is None:
            useDQS_m = useDQ
        else:
            useDQS_m = self.useDQ
        self.update(dofParams)
        self.B = self.skeleton.B

        self.vertTransformations = torch.zeros((self.B, self.numVert, 4, 4)).float().to(self.device)
        if not useDQS_m:
            self.vertTransformations = torch.einsum('bvd,bdmn->bvmn', self.skinWeight.repeat(self.B,1,1), self.jointTransformations)
        else:
            self.DQTransformations = torch.zeros((self.B, self.skinWeight.shape[1], 8)).float().to(self.device)
            self.DQSigns = torch.ones((self.B, self.skinWeight.shape[1], 1)).float().to(self.device)
            for i in range(self.skinWeight.shape[1]):
                self.DQTransformations[:, i] = fromTransformation2VectorTorch(self.jointTransformations[:,i])
            for i in range(self.skinWeight.shape[1]):
                # self.DQSigns[:, i] = 1 if torch.dot(self.DQTransformations[:, 0, :4], self.DQTransformations[:, i, :4]) > 0 else -1
                # self.DQSigns[:, i] = 1 if torch.einsum('bm,bm->b', self.DQTransformations[:, 0, :4], self.DQTransformations[:, i, :4]).reshape(self.B, 1) > 0 else -1
                self.DQSigns[:, i] = torch.where(   torch.einsum('bm,bm->b', self.DQTransformations[:, 0, :4],
                                                       self.DQTransformations[:, i, :4]).reshape(self.B, 1) > 0     , 1 ,-1  )

                    # if torch.einsum('bm,bm->b', self.DQTransformations[:, 0, :4],
                    #                                    self.DQTransformations[:, i, :4]).reshape(self.B, 1) > 0 else -1
            for i in range(self.skinWeight.shape[1]):
                self.DQTransformations[:,i] *= self.DQSigns[:,i]

            self.vertDQTransformations = torch.matmul(self.skinWeight.repeat(self.B,1,1), self.DQTransformations) #B,D,8
            for i in range(self.skinWeight.shape[0]):
                self.vertTransformations[:,i] = fromVector2TransformationTorch(self.vertDQTransformations[:,i]) #out: B,N,4,4

        verts = torch.matmul(self.vertTransformations[:, :, :3, :3], self.verts0.reshape(1,self.numVert,3,1))[:, :,:,0] + \
                self.vertTransformations[:, :, :3, 3]

        return verts

class Skeleton():
    def __init__(self, skelPath = None, verbose = False, useDQ=True, device='cpu', newCharacter = False):
        self.skelPath = skelPath
        self.newCharacter = newCharacter
        self.verbose = verbose
        self.useDQ = useDQ
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.numJoint = -1
        self.numMarker = -1
        self.numDof = -1
        self.loadSkeleton()
        # self.update(np.zeros(self.numDof))
        self.update(torch.zeros(1, self.numDof, dtype=torch.float32).to(self.device))

    def loadSkeleton(self):
        ## load joint
        self.jointNames, self.jointParents, self.jointTypes,\
        self.jointOffsets, self.jointAxis, self.jointScales, self.jointIds = loadJoint(self.skelPath, returnTensor=True, device = self.device)
        self.numJoint = len(self.jointNames)
        self.jointParentIdxs = [-1, ]
        for i in range(1, len(self.jointParents)):
            self.jointParentIdxs.append(self.jointNames.index(self.jointParents[i]))
        self.revoluteList = []
        self.prismaticList = []
        for i in range(self.numJoint):
            if self.jointTypes[i]=='revolute':
                self.revoluteList.append(i)
            else:
                self.prismaticList.append(i)

        # self.jointGlobalTransformations = np.tile(np.eye(4),(self.numJoint,1,1))
        # self.jointGlobalPositions = np.zeros([self.numJoint,3])
        self.jointLocalTranslation = torch.eye(4, dtype=torch.float32).repeat(self.numJoint, 1, 1).to(self.device)
        for i in range(self.numJoint):
            self.jointLocalTranslation[i, :3,3] = self.jointOffsets[i] * self.jointScales[i]

        ## load marker
        self.markerNames, self.markerParents, self.markerTypes,\
        self.markerPositions, self.markerSizes, self.markerColors = loadMarker(self.skelPath, returnTensor=True, device = self.device)
        self.numMarker = len(self.markerNames)
        # self.markerGlobalPositions = np.zeros([self.numMarker, 3])
        self.markerParentIdxs = []
        for i in range(self.numMarker):
            self.markerParentIdxs.append(self.jointNames.index(self.markerParents[i]))

        ## load dof
        # self.dofNames, self.dofSubNums, self.dofLimits,\
        # self.dofJointNames, self.dofJointWeights = loadDof(self.skelPath)
        # self.numDof = len(self.dofNames)
        self.dofNames, self.dofSubNums, self.dofLimits,\
        self.dofJointNames, self.dofJointWeightsCompressed = loadDof(self.skelPath)
        self.numDof = len(self.dofNames)
        self.dofJointWeights = torch.zeros([self.numDof,self.numJoint]).to(self.device)
        for i in range(self.numDof):
            for j in range(len(self.dofJointWeightsCompressed[i])):
                # jointIdx_ = self.dofJointWeightsCompressed[i][j]
                # jointIdx_ = self.dofJointWeightsCompressed[i][j]
                jointIdx_ = self.jointNames.index(self.dofJointNames[i][j])

                self.dofJointWeights[i, jointIdx_] = self.dofJointWeightsCompressed[i][j]

        self.dofWeights = torch.ones([1,len(self.dofNames)]).float().to(self.device)
        for i in range(self.numDof):
            if 'scale' in self.dofNames[i]:
                self.dofWeights[0,i] = 0.0

        # self.dofParam = np.zeros(self.numDof)
        # self.jointParam = np.zeros(self.numJoint)
        if self.verbose:
            print("-- Loading skeleton from ==> {}".format(self.skelPath))
            print("-- The skeleton has {} joints".format(self.numJoint))
            print("-- The skeleton has {} markers".format(self.numMarker))
            print("-- The skeleton has {} dofs".format(self.numDof))
            print("-- Use DualQuaternions: {}".format(self.useDQ))

        self.pairs = []
        self.pairsMinus6 = []
        for idx in range(7, len(self.jointNames)):
            self.pairs.append([idx, self.jointNames.index(self.jointParents[idx])])
            self.pairsMinus6.append( [idx - 6 , self.jointNames.index(self.jointParents[idx]) - 6] )

    def update(self, dofParams, updateMarker = False):
        # Update the joint parameter
        """
            Input: dofParam ==> B,numDof
        """
        self.B = B = dofParams.shape[0]
        self.jointGlobalTransformations = torch.eye(4, dtype=torch.float32).repeat(B, self.numJoint, 1, 1).to(self.device)
        if self.useDQ:
            self.jointGlobalTransformationsDQ = torch.tensor([1,0,0,0,0,0,0,0], dtype=torch.float32).repeat(B, self.numJoint, 1).to(self.device)
        self.jointParams = torch.zeros(B, self.numJoint, dtype=torch.float32).to(self.device)
        self.jointGlobalPositions = torch.zeros([B, self.numJoint, 3], dtype=torch.float32).to(self.device)
        self.markerGlobalPositions = torch.zeros([B, self.numMarker, 3], dtype=torch.float32).to(self.device)

        dofParams = dofParams * self.dofWeights

        self.jointParams = torch.mm(dofParams, self.dofJointWeights)


        # for i in range(self.numDof):
        #     dof_param = dofParams[:, i] # B,
        #     """
        #         v1.0
        #     """
        #     if 'scale' in self.dofNames[i]:
        #         dof_param *= 0
        #     """
        #         end
        #     """
        #     dof_joints = self.dofJointNames[i]
        #     dof_joints_weight = self.dofJointWeights[i]
        #     for j in range(len(dof_joints_weight)):
        #         jointIdx = self.jointNames.index(dof_joints[j])
        #         """
        #             By adding instead of setting the value of dof.
        #         """
        #         self.jointParams[:, jointIdx] += dof_joints_weight[j] * dof_param

        jointAxis = self.jointAxis.repeat(B, 1 ,1)
        jointLocalTransformation = torch.eye(4, dtype=torch.float32).repeat(B, self.numJoint, 1, 1).to(self.device)
        jointLocalTransformation[:, self.revoluteList, :3, :3] = axis_angle_to_matrix( jointAxis[:,self.revoluteList]* self.jointParams[:, self.revoluteList][:,:,None])
        jointLocalTransformation[:, self.prismaticList, :3, 3] = jointAxis[:,self.prismaticList]* self.jointParams[:, self.prismaticList][:,:,None]
        jointLocalTransformation = self.jointLocalTranslation.repeat(B, 1, 1, 1).matmul( jointLocalTransformation  )

        dq_locals = fromTransformation2VectorTorch(jointLocalTransformation.view(-1,4,4)).reshape(B,-1,8)

        # if self.useDQ:

        #N,4,4
        # print()
        # # FK
        for i in range(self.numJoint):
        # for i in range(self.numJoint//2):
            if i>0:
                if self.useDQ:
                    parentGlobalTransformationDQ  = self.jointGlobalTransformationsDQ[:, self.jointParentIdxs[i]]
                else:
                    parentGlobalTransformation = self.jointGlobalTransformations[:, self.jointParentIdxs[i]]
            else:
                if self.useDQ:
                    parentGlobalTransformationDQ = torch.tensor([1,0,0,0,0,0,0,0], dtype=torch.float32).repeat(B, 1).to(self.device)
                else:
                    parentGlobalTransformation = torch.eye(4, dtype=torch.float32).repeat(B, 1, 1).to(self.device)
        #     joint_type = self.jointTypes[i]
        #     jointLocalTransformation = torch.eye(4,dtype=torch.float32).repeat(B, 1, 1).to(self.device)
        #     jointLocalTranslation = self.jointLocalTranslation[i].repeat(B, 1, 1)
        #     jointAxis = self.jointAxis[i].repeat(B, 1)
        #     if joint_type == 'revolute':
        #         jointLocalTransformation[:, :3, :3] = axis_angle_to_matrix(jointAxis * self.jointParams[:, i].reshape(B,1))
        #     elif joint_type == 'prismatic':
        #         jointLocalTransformation[:, :3, 3] = jointAxis * self.jointParams[:, i].reshape(B,1)
        #     else:
        #         raise ValueError("No implemation for joint type ==> {}".format(joint_type))
        #
            if self.useDQ:
                pass
        #         # print('[DEBUG] use DQS in joint update')
        #         dq_parent = fromTransformation2VectorTorch(parentGlobalTransformation)

                # local = jointLocalTranslation.matmul(jointLocalTransformation)
                # local = jointLocalTransformation[:,i]
                # dq_local = fromTransformation2VectorTorch(local)
                # dq_parent = parentGlobalTransformationDQ
                # dq_local = dq_locals[:,i]

                # total = imodDQTorch(parentGlobalTransformationDQ, dq_locals[:,i])

                # print()

                # # total = normalizeDQTorch(total)
                self.jointGlobalTransformationsDQ[:, i] = imodDQTorch(parentGlobalTransformationDQ, dq_locals[:, i])
                # self.jointGlobalTransformationsDQ[:, i] = total
                # self.jointGlobalTransformations[:, i] = fromVector2TransformationTorch(total)
            else:
                # print()
                self.jointGlobalTransformations[:, i] = parentGlobalTransformation.clone().matmul(jointLocalTransformation[:,i])

        if self.useDQ:
            self.jointGlobalTransformations = fromVector2TransformationTorch(self.jointGlobalTransformationsDQ.view(-1,8)).reshape(B, self.numJoint ,4,4)

                # print('lbs')
                # self.jointGlobalTransformations[:, i] =  parentGlobalTransformation.matmul(jointLocalTranslation).matmul(jointLocalTransformation[:,i])
        #
            self.jointGlobalPositions = self.jointGlobalTransformations[:, :, :3, 3]

        if updateMarker:
            # FK of marker
            # for i in range(self.numMarker):
            #     parentGlobalTransformation = self.jointGlobalTransformations[:, self.markerParentIdxs[i]]
            #     self.markerGlobalPositions[:,i] = parentGlobalTransformation[:, :3,:3].matmul(self.markerPositions[i].reshape(1,3,1)).reshape(-1,3)+ \
            #                                     parentGlobalTransformation[:, :3, 3].reshape(-1,3)
            parentGlobalTransformations = self.jointGlobalTransformations[:, self.markerParentIdxs]
            # self.markerGlobalPositions = parentGlobalTransformations[:, :, 3,:3].matmul(self.markerPositions[:].reshape(1,3,1)) + \
            #                                 parentGlobalTransformation[:, :3, 3]
            self.markerGlobalPositions  = torch.einsum('tkmn,kn->tkm', parentGlobalTransformations[:, :, :3,:3], self.markerPositions[:,:]) + \
                                          parentGlobalTransformations[:, :, :3, 3]



    # For visualization
    def getMarkerMesh(self,idx=0):
        markerPositions = []
        markerSizes = []
        # for i in range(self.markerNum):
        #     markerPositions.append(self.markerList[i].globalposition)
        markerPositions = to_cpu(self.markerGlobalPositions[idx]).reshape(-1,3)
        # markerPositions = np.array(self.markerGlobalPositions).reshape(-1,3)
        mesh = create_point(markerPositions/1000, r= self.markerSizes/1000/2, colors = self.markerColors * 255.0)
        mesh.vertices *= 1000.0
        return mesh

    # For visualization
    def getMesh(self,idx=0):
        # jointPositions = []
        # for i in range(self.numJoint):
        #     jointPositions.append(self.jointGlobalPositions[i])
        # jointPositions = np.array(jointPositions)
        jointPositions = to_cpu(self.jointGlobalPositions[idx]).reshape(-1, 3)
        # mesh = create_point(jointPositions[6:]/1000)
        # mesh = create_skeleton(jointPositions[6:]/1000, pairs= self.pairs)
        mesh = create_skeleton(jointPositions[6:]/1000, pairs= self.pairsMinus6)

        mesh.vertices *= 1000.0
        return mesh

def loadJoint(skelPath, verbose=False, returnTensor=False, device = None):
    with open(os.path.join(skelPath), 'r') as f:
        data = f.readlines()

    numJoint = -1
    stIdx = -1
    jointNames = []
    jointParents = []
    jointTypes = []
    jointOffsets = []
    jointAxis = []
    jointScales = []
    jointIds = []
    for idx, item in enumerate(data):
        if item.split()[0] == 'joints:':
            numJoint = int(item.split()[1])
            stIdx = idx
            break
        if idx > len(data)-1 and numJoint == -1:
            raise ValueError('No joints loaded from ==> {}.'.format(skelPath))
    for i in range(0, numJoint):
        data_temp = data[stIdx + 1 + i].split()
        joint_name = data_temp[0]
        type_name = data_temp[1]
        parent_name = data_temp[2]
        offset = data_temp[3:6]
        offset = np.array([float(item) for item in offset])
        axis = data_temp[6:9]
        axis = np.array([float(item) for item in axis])
        scale = float(data_temp[9])
        jointNames.append(joint_name)
        jointParents.append(parent_name)
        jointTypes.append(type_name)
        jointOffsets.append(offset)
        jointAxis.append(axis)
        jointScales.append(scale)
        jointIds.append(i)
    # jointNames = np.array(jointNames)
    jointParents = np.array(jointParents)
    jointTypes = np.array(jointTypes)
    jointOffsets = np.array(jointOffsets)
    jointAxis = np.array(jointAxis)
    jointScales = np.array(jointScales)
    jointIds = np.array(jointIds)
    if returnTensor:
        jointOffsets = torch.from_numpy(jointOffsets).float().to(device)
        jointAxis = torch.from_numpy(jointAxis).float().to(device)
        jointScales = torch.from_numpy(jointScales).float().to(device)
    return jointNames, jointParents, jointTypes, jointOffsets, jointAxis, jointScales, jointIds

def loadDof(skelPath, newCharacter=False, verbose=False, returnTensor=False):
    with open(os.path.join(skelPath), 'r') as f:
        data = f.readlines()

    numDof = -1
    stIdx = -1
    dofNames = []
    dofSubNums = []
    dofLimits = []
    dofJointNames = []
    dofJointWeights = []
    for idx, item in enumerate(data):
        if item.split()[0] == 'dofs:':
            numDof = int(item.split()[1])
            stIdx = idx
            break
        if idx > len(data) - 1 and numDof == -1:
            raise ValueError('No dofs loaded from ==> {}.'.format(skelPath))
    count = 0
    j = 0
    while count < numDof:
        data_temp = data[stIdx + 1 + j].split()
        dof_name = data_temp[0]
        dof_subNum = int(data_temp[1])
        dofNames.append(dof_name)
        dofSubNums.append(dof_subNum)
        j += 1
        limit_temp = data[stIdx + 1 + j].split()
        if limit_temp[0] == "nolimits" and len(limit_temp) == 1:
            dofLimits.append([-1])
        elif limit_temp[0] == "limits" and len(limit_temp) == 3:
            dofLimits.append([float(limit_temp[1]), float(limit_temp[2])])
        j += 1
        temp_list = []
        temp_list_weight = []
        for k in range(dof_subNum):
            sub_limit_temp = data[stIdx + 1 + j].split()
            temp_list.append(sub_limit_temp[0])
            temp_list_weight.append(float(sub_limit_temp[1]))
            j += 1
        dofJointNames.append(temp_list)
        dofJointWeights.append(temp_list_weight)
        count += 1
    for i in range(numDof):
        if newCharacter:
            if 'hip' in dofNames[i] or 'shoulder_y' in dofNames[i]:
                dofLimits[i] = [-10, 10]
            if 'shoulder_y' in dofNames[i]:
                dofLimits[i] = [-0.6, 2]
            if 'ankle_twist' in dofNames[i]:
                dofLimits[i] = [-0.5, 0.5]
    # dofNames = np.array(dofNames)
    dofSubNums = np.array(dofSubNums)
    dofLimits = np.array(dofLimits, dtype=object)
    dofJointNames = np.array(dofJointNames, dtype=object)
    dofJointWeights = np.array(dofJointWeights, dtype=object)
    return dofNames, dofSubNums, dofLimits, dofJointNames, dofJointWeights

def loadMarker(skelPath, verbose=False, returnTensor=False, device = None):
    with open(os.path.join(skelPath), 'r') as f:
        data = f.readlines()
        numMarker = -1
        stIdx = -1
        markerNames = []
        markerParents = []
        markerTypes = []
        markerPositions = []
        markerSizes = []
        markerColors = []
        for idx, item in enumerate(data):
            if item.split()[0]=='markers:':
                numMarker = int(item.split()[1])
                stIdx = idx
                break
            if idx > len(data)-1 and numMarker == -1:
                raise ValueError('No markers loaded from ==> {}.'.format(skelPath))
        for idx in range(0, numMarker):
            data_temp = data[stIdx + 1 + idx].split()
            marker_name = data_temp[0]
            parent_name = data_temp[1]
            marker_type = data_temp[2]
            if marker_type == 'point':
                position = data_temp[3:6]
                position = np.array([float(item) for item in position])
                size = float(data_temp[6])
                color = data_temp[7:10]
                color = np.array([float(item) for item in color])
            else:
                raise ValueError('No implemation for marker type ==> {}.'.format(marker_type))
        #     joint_parent = self.getJointByExactName(parent_name)
            # markerList.append(Marker(i, joint_parent, position, marker_name))
            markerNames.append(marker_name)
            markerParents.append(parent_name)
            markerTypes.append(marker_type)
            markerPositions.append(position)
            markerSizes.append(size)
            markerColors.append(color)
        # markerNames = np.array(markerNames)
        markerParents = np.array(markerParents)
        markerTypes = np.array(markerTypes)
        markerPositions = np.array(markerPositions)
        markerSizes = np.array(markerSizes)
        markerColors = np.array(markerColors)
        if returnTensor:
            markerPositions = torch.from_numpy(markerPositions).float().to(device)
            # markerSizes = torch.from_numpy(markerSizes).float()

        return markerNames, markerParents, markerTypes, markerPositions, markerSizes, markerColors

def loadMotion(motionPath, returnIdx= False, returnTensor=False, device = None ):
    """
    Input: motionPath
    Output: motionList, motionIdxList in numpy array format
    """
    with open(os.path.join(motionPath), 'r') as f:
        data = f.readlines()
    assert data[0] == 'Skeletool Motion File V1.0\n'
    frameNum = len(data) - 1
    motionList = []
    for i in range(1, len(data)):
        motion = data[i].split()
        motion = [float(item) for item in motion]
        motion = np.array(motion).reshape(1, -1)  # 58
        motionList.append(motion)
    motionList = np.concatenate(motionList, 0)
    motionIdxList = motionList[:, :1].astype('int')
    motionList = motionList[:, 1:]
    if returnTensor:
        import torch
        motionList = torch.from_numpy(motionList).float().to(device)
        motionIdxList = torch.from_numpy(motionIdxList).to(device)
    if returnIdx:
        return motionList, motionIdxList
    else:
        return motionList

def loadMeshes(meshesPath, returnIdx= False, returnTensor=False, device = None ):
    """
        Input: meshesPath
        Output: meshList, meshIdxList in numpy array format
    """
    with open(meshesPath, 'r') as f:
        data = f.readlines()
    assert data[0] == 'Skeletool Meshes file v2.1 \n'
    meshList = []
    for i in range(1, len(data)):
        mesh = data[i].split()
        mesh = [float(item) for item in mesh]
        mesh = np.array(mesh).reshape(1,-1)
        meshList.append(mesh)
    meshList = np.concatenate(meshList, 0)
    meshIdxList = meshList[:,:1].astype(int)
    meshList = meshList[:,1:].reshape([len(meshIdxList),-1,3])
    if returnTensor:
        import torch
        meshList = torch.from_numpy(meshList).float().to(device)
        meshIdxList = torch.from_numpy(meshIdxList).float().to(device)
    if returnIdx:
        return meshList, meshIdxList
    else:
        return meshList

def saveMeshes(meshesPath, meshList, frameList=None):
    with open(os.path.join(meshesPath), 'w') as f:
        f.write('Skeletool Meshes file v2.1 \n')
        if frameList is None:
            frameList = np.arange(len(meshList))
        for idx, fIdx in enumerate(frameList):
            f.write(str(fIdx) + ' ')
            for vertex in range(0, meshList[idx].shape[0]):
                x = '%.1f' % (meshList[idx][vertex][0])
                y = '%.1f' % (meshList[idx][vertex][1])
                z = '%.1f' % (meshList[idx][vertex][2])
                f.write(x + ' ' + y + ' ' + z + ' ')
            f.write('\n')

def saveMotion(motionPath, motionList, motionIdxList = None):
    if not motionIdxList is None:
        assert len(motionList) == len(motionIdxList)
    else:
        motionIdxList = np.arange(len(motionList)).astype('str')
    motionList = motionList.astype('str')
    with open(os.path.join(motionPath), 'w') as f:
        f.write('Skeletool Motion File V1.0\n')
        for i in range(len(motionIdxList)):
            outstr = ' '.join([motionIdxList[i]] + list(motionList[i])) + '\n'
            f.write(outstr)

def loadChar(charPath):
    with open(os.path.join(charPath), 'r') as f:
        data = f.readlines()
    assert data[0] == 'skeletool character file v1.0\n'
    dirName = os.path.dirname(charPath)
    skelPath = os.path.join(dirName, data[2].split()[0])
    meshPath = os.path.join(dirName, data[4].split()[0])
    skinPath = os.path.join(dirName, data[6].split()[0])
    motionBasePath = os.path.join(dirName, data[8].split()[0])
    return skelPath, meshPath, skinPath, motionBasePath

def saveChar(charPath):
    with open(os.path.join(charPath), 'w') as f:
        f.write("skeletool character file v1.0\n")
        f.write("skeleton\n")
        f.write("  actor.skeleton\n")
        f.write("mesh\n")
        f.write("  actor.obj\n")
        f.write("skin\n")
        f.write("  actor.skin\n")
        f.write("pose\n")
        f.write("  actor.motion\n")

def loadSkin(skinPath, returnTensor=False, device=None):
    with open(os.path.join(skinPath), 'r') as f:
        data = f.readlines()
    assert data[0]== 'Skeletool character skinning file V1.0\n'
    skinBoneNames = data[2].split()
    skinBoneNum = len(skinBoneNames)
    weightRaws = data[4:]
    vertNum = len(weightRaws)
    skinWeight = np.zeros((vertNum, skinBoneNum))
    skinFirstIdxs = []

    for i in range(vertNum):
        temp_weight = weightRaws[i].split()
        rowIdx = int(temp_weight[0])
        for j in range(1, len(temp_weight), 2):
            colIdx = int(temp_weight[0 + j])
            value = float(temp_weight[0 + j + 1])
            skinWeight[rowIdx, colIdx] = value
            if j == 1:
                skinFirstIdxs.append(colIdx)
    skinWeight = skinWeight / np.sum(skinWeight, axis=1).reshape(-1, 1)
    if returnTensor:
        skinWeight = torch.from_numpy(skinWeight).float().to(device)
    return skinBoneNames, skinWeight, skinFirstIdxs


def saveSkin(skinPath, skinBoneNames, skinWeight):
    with open(os.path.join(skinPath), 'w') as f:
        f.write("Skeletool character skinning file V1.0\n")
        f.write("bones:\n")
        f.write('    ' + ' '.join(skinBoneNames) + "\n")
        f.write("vertex weights:\n")
        for i in range(skinWeight.shape[0]):
            outStr = '    ' + str(i)
            for j in range(skinWeight.shape[1]):
                outStr = outStr + '   ' +str(j) + '  ' + str(skinWeight[i,j])
            f.write(outStr + "\n")
        # return skinBoneNames, skinWeight

def saveSkeleton(skelPath,
                 jointNames, jointParents, jointTypes, jointOffsets,jointAxis,
                 dofNames, dofSubNums, dofLimits, dofJointNames, dofJointWeightsCompressed):
    jointNum = len(jointNames)
    markerNum = 1
    dofNum = len(dofNames)
    with open(os.path.join(skinPath), 'w') as f:
        f.write("Skeletool Skeleton Definition V1.0\n")
        f.write("joints: {}\n".format(jointNum))
        for idx in range(jointNum):
            f.write("{} {} {} {} {} {} {} {} {} {} {}\n".format(jointNames[idx], jointTypes[idx], jointParents[idx], \
                                                  jointOffsets[idx][0], jointOffsets[idx][1], jointOffsets[idx][2], \
                                                  jointAxis[idx][0], jointAxis[idx][1], jointAxis[idx][2], 1.0))
            pass
        f.write("markers: {}\n".format(markerNum))
        f.write("  0_head_top           {}              point       5.68411    182.536    17.6206         40          1          1          0\n".format(jointNames[0]))
        f.write("scaling joints: 0\n")
        f.write("dofs: {}\n".format(dofNum))
        for idx in range(dofNum):
            pass
        f.write("textureEntries: 0\n")


if __name__ == '__main__':
    import trimesh
    # print(1)
    # skelPath = r'Y:\HOIMOCAP2\work\data\VladNew\vlad.skeleton'
    # datadir = r'D:\07_Data\07_RTPMC\VladNew'
    datadir = r'Y:\HOIMOCAP2\work\data\VladNew'
    # charPath = osp.join(datadir, 'vlad.character')
    charPath = osp.join(datadir, 'vlad_autorig.character')
    skinPath = osp.join(datadir, 'vlad.skin')
    skelPath = osp.join(datadir, 'vlad.skeleton')
    motionBasePath = osp.join(datadir, 'vlad.motion')
    motionPath = osp.join(datadir, r'training\motions\poseAngles.motion')
    meshPath = osp.join(datadir, 'vlad.obj')
    graphPath = osp.join(datadir, 'vladSimplified486.obj')
    connectionPath = osp.join(datadir, 'vladSimplified486_connection.npz')
    # skleton = Skeleton(skelPath=skelPath,verbose=True, useDQ=True)
    # character = Character(skinPath=skinPath,meshPath=meshPath,motionPath=motionPath,skelPath=skelPath,\
    #                       motionBasePath=motionBasePath,useDQ=True,verbose=True)
    device = 'cuda:0'
    useDQ = True
    verbose = True
    sc = SkinnedCharacter(charPath=charPath, useDQ=useDQ, verbose=verbose,
                          device=device, segWeightsFlag=True, computeAdjacencyFlag=True)
    # eg = EmbeddedGraph(character=sk, graphPath=graphPath, connectionPath=None, useDQ=useDQ, verbose=verbose, device=device)
    # eg = EmbeddedGraph(character=sk, graphPath=graphPath, connectionPath=connectionPath, useDQ=useDQ, verbose=verbose, device=device)
    eg = EmbeddedGraph(character=sc, graphPath=graphPath, computeConnectionFlag=False, connectionPath=connectionPath, useDQ=useDQ, verbose=verbose, device=device)
    motions = loadMotion(motionPath, returnIdx=False, returnTensor=True,device=device)

    deltaTs = torch.randn(2, 486, 3).to(eg.device) * 10
    # skinRs, skinTs = eg.updateNode(motions[[2222, 2223]])
    skinRs, skinTs = eg.updateNode(motions[[8920, 9500]])
    vertsG = eg.forwardG(deltaTs=deltaTs, skinRs = skinRs, skinTs = skinTs)
    vertsF = eg.forwardF(deltaTs=deltaTs, skinRs = skinRs, skinTs = skinTs)

    vertsGInverse = eg.inverseG(vertsG, deltaTs=deltaTs, skinRs = skinRs, skinTs = skinTs)
    # verts = character(motions[[2222]])
    # mesh = character.getMesh(0)
    # motions2 = loadMotion(r'Y:\HOIMOCAP2\work\data\Subject0003\tight\training\recon_neus2\debug\star.motion', returnIdx=False, returnTensor=True, device=device)

    meshG = eg.getMeshG(vertsG, 0)
    meshF = eg.getMeshF(vertsF, 0)
    _ = meshG.export(r'D:\06_Exps\ed\torch\meshG.obj')
    _ = meshF.export(r'D:\06_Exps\ed\torch\meshF.obj')

    deltaTs = torch.randn(1, 486, 3).to(eg.device) * 10
    skinRs, skinTs = eg.updateNode(motions[[8920]])
    vertsF = eg.forwardF(deltaTs=deltaTs, skinRs=skinRs, skinTs=skinTs)
    meshF = eg.getMeshF(vertsF, 0)
    _ = meshF.export(r'D:\06_Exps\ed\torch\inverseFNN\meshF.obj')

    pts = vertsF + torch.randn_like(vertsF) * 50
    save_ply(r'D:\06_Exps\ed\torch\inverseFNN\meshF_pts.ply', pts[0].data.cpu().numpy())




    with torch.no_grad():
        dist_sq, idx, neighbors = ops.knn_points(pts.float(), vertsF.float(), K=1)

    idx = idx.reshape(vertsF.shape[0],-1)
    ptsRecovered = eg.inverseFPts(pts, idx = idx, deltaTs=deltaTs, skinRs=skinRs, skinTs=skinTs)
    save_ply(r'D:\06_Exps\ed\torch\inverseFNN\meshF_ptsRecovered.ply', ptsRecovered[0].data.cpu().numpy())

    meshF0 = eg.getMeshF(eg.character.verts0[None], 0)
    _ = meshF0.export(r'D:\06_Exps\ed\torch\inverseFNN\meshF0.obj')


    vertsF = eg.forwardF(skinRs=skinRs, skinTs=skinTs)
    meshF = eg.getMeshF(vertsF, 0)
    _ = meshF.export(r'D:\06_Exps\ed\torch\inverseF\meshF.obj')

    vertsFDeformed = eg.forwardF()
    meshFDeformed = eg.getMeshF(vertsFDeformed, 0)
    _ = meshFDeformed.export(r'D:\06_Exps\ed\torch\inverseF\meshFDeformed.obj')

    vertsFInverse = eg.inverseF(vertsF, skinRs=skinRs, skinTs=skinTs)
    meshFInverse = eg.getMeshF(vertsFInverse, 0)
    _ = meshFInverse.export(r'D:\06_Exps\ed\torch\inverseF\meshFInverse.obj')


    vertsF0 = eg.forwardF()
    meshF0 = eg.getMeshF(vertsF0, 0)
    _ = meshF0.export(r'D:\06_Exps\ed\torch\inverseF\meshF0.obj')

    vertsF = eg.forwardF(deltaTs=deltaTs)
    meshF = eg.getMeshF(vertsF, 0)
    _ = meshF.export(r'D:\06_Exps\ed\torch\inverseF\meshFDeformed.obj')

    vertsFInverseDeformed = eg.inverseF(vertsF, deltaTs=deltaTs)
    vertsFInverseDeformed = eg.getMeshF(vertsFInverseDeformed, 0)
    _ = vertsFInverseDeformed.export(r'D:\06_Exps\ed\torch\inverseF\vertsFInverseDeformed.obj')

    pts = vertsF + torch.randn_like(vertsF) * 50
    with torch.no_grad():
        dist_sq, idx, neighbors = ops.knn_points(pts.float(), vertsF.float(), K=1)


    print(1)
