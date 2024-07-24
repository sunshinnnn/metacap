"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2022-11-30
"""

import os
import numpy as np
import torch
import trimesh
import pyrender
from pyrender import RenderFlags
import cv2
import time
from PIL import Image

colorDict = {
    'pink': [1.00, 0.75, 0.80],
    'purple': [0.63, 0.13, 0.94],
    'red': [1.0, 0.0, 0.0],
    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [1.00, 0.25, 0.25],
    'blue': [.0, .0, 1.],
    'white': [1., 1., 1.],
    'orange': [1.00, 0.65, 0.00],
    'grey': [0.75, 0.75, 0.75],
    'black': [0., 0., 0.],
}

render_flags = {
    'flip_wireframe': False,
    'all_wireframe': False,
    'all_solid': True,
    'shadows': False,  # TODO:bug exists in shadow mode
    'vertex_normals': False,
    'face_normals': False,
    'cull_faces': False,  # set to False
    'point_size': 1.0,
    'rgba': True
}

temp0 = os.path.realpath(__file__)
temp1 = os.path.dirname(temp0)
temp2 = os.path.dirname(temp1)


def load_sphere():
    mesh = trimesh.load(os.path.join(temp2,'datas', 'vis','sphere2.obj'))
    return mesh.vertices, mesh.faces
def load_cylinder(cType='e'):
    if cType=='e':
        mesh = trimesh.load(os.path.join(temp2, 'datas', 'vis','cylinder5.obj'))
    elif cType=='c':
        mesh = trimesh.load(os.path.join(temp2, 'datas', 'vis','cylinder2.obj'))
    return mesh.vertices, mesh.faces



def create_point(points, r=0.01, colors = None):
    nPoints = points.shape[0]
    vert, face = load_sphere()
    nVerts = vert.shape[0]
    vert = vert[None, :, :].repeat(points.shape[0], 0) # 50,22,3
    import numbers
    if isinstance(r, numbers.Number):
        vert *= r
    else:
        if isinstance(r, list):
            r = np.array(r)
        r = r.reshape( points.shape[0],1,1)
        vert *= r
    vert = vert + points[:, None, :] #50,1,3
    verts = np.vstack(vert)
    face = face[None, :, :].repeat(points.shape[0], 0)
    face = face + nVerts * np.arange(nPoints).reshape(nPoints, 1, 1)
    faces = np.vstack(face)
#     return {'vertices': verts, 'faces': faces, 'name': 'points'}
    mesh = trimesh.Trimesh(vertices=verts,faces=faces,process=False)
    if not colors is None:
        if isinstance(colors, list):
            colors = np.array(colors).reshape(-1,3)
        if colors.shape[1]==3:
            colors = np.concatenate([colors , 255*np.ones((colors.shape[0],1))], -1)
        if colors.shape[0]== nPoints:
            # colors = colors[:, None].repeat(nVerts, axis=1).reshape(nPoints*nVerts,-1) #50,3 -> 50,1
            colors = colors[:, None ,:].repeat(nVerts, axis=1)
            # colors = np.transpose(colors, (1, 0, 2))
            colors = np.vstack(colors)
        mesh.visual.vertex_colors[:] = colors
    return mesh



def calRot(axis, direc):
    direc = direc/np.linalg.norm(direc)
    axis = axis/np.linalg.norm(axis)
    rotdir = np.cross(axis, direc)
    rotdir = rotdir/np.linalg.norm(rotdir)
    rotdir = rotdir * np.arccos(np.dot(direc, axis))
    rotmat, _ = cv2.Rodrigues(rotdir)
    return rotmat

def create_line(start, end, min_length,r=0.01, col=None, cType='e'):
    length = np.linalg.norm(end[:3] - start[:3])
    vertices, faces = load_cylinder(cType = cType)
    vertices[:, :2] *= r * (length/min_length) *2
    vertices[:, 2] *= length/2
    rotmat = calRot(np.array([0, 0, 1]), end - start)
    vertices = vertices @ rotmat.T + (start + end)/2
    mesh = trimesh.Trimesh(vertices=vertices,faces=faces)
    return mesh

def euler(rots, order='xyz', degrees=True):
    from scipy.spatial.transform import Rotation as R
    return R.from_euler('xyz', rots ,degrees=degrees).as_matrix()

def create_skeleton(kpt_3d, num=24, pairs=None, r1=0.02, r2=0.01, color=None, filename=None, cType='e'):
    if pairs is None:
        if num == 24:
            pairs = [[1, 0], [4, 1], [7, 4], [10, 7], [2, 0], [5, 2], [8, 5], [11, 8], [3, 0], [6, 3], [9, 6], [12, 9], [15, 12], [13, 9],
                         [16, 13], [18, 16], [20, 18], [22, 20], [14, 9], [17, 14], [19, 17], [21, 19], [23, 21]]
        elif num == 25:
            pairs = [(1, 8), (9, 10), (10, 11), (8, 9), (8, 12), (12, 13), (13, 14), (1, 2), (2, 3),(3, 4), (1, 5), (5, 6),
                         (6, 7), (1, 0), (0, 15), (0, 16), (15, 17), (16, 18), (14, 19), (11, 22)]
        # elif num == 93:
        #     pairs = []
    else:
        pairs = pairs
    meshes = []
#     for i in range(kpt_3d.shape[0]):
    meshes.append(create_point(kpt_3d,r=r1))
    lengths_list = []
    for l in range(len(pairs)):
        i1, i2 = pairs[l][0],pairs[l][1]
        st,ed = kpt_3d[i1],kpt_3d[i2]
        if np.linalg.norm(ed[:3] - st[:3]) > 1e-2:
            lengths_list.append(np.linalg.norm(ed[:3] - st[:3]))
    min_length = min(lengths_list)
    print(min_length)
    for l in range(len(pairs)):
        i1, i2 = pairs[l][0],pairs[l][1]
        st,ed = kpt_3d[i1],kpt_3d[i2]
        if np.linalg.norm(ed[:3] - st[:3]) > 1e-2:
            meshes.append(create_line(st,ed,min_length,r=r2, cType = cType))
    mesh = trimesh.util.concatenate(meshes)
    if not color is None:
        mesh = Mesh(vertices=mesh.vertices,faces=mesh.faces,vc=color)
    return mesh

def create_camera(K=np.array([[1024,0,0,0], [0,1024,0,0], [0,0,1,0],[0,0,0,1]]), E=np.eye(4),S=1.0, H = 1024, W = 1024):
    if K.shape[0]==3:
        K_ = np.eye(4)
        K_[:3, :3] = K
        K = K_
    P = np.linalg.inv( K.dot(E) )
    pts = []
    pts.append(P.dot(np.array([0, 0, 0, 1]).reshape(4, 1))[:3].T)
    pts.append(P.dot(np.array([W * S, 0, S, 1]).reshape(4, 1))[:3].T)
    pts.append(P.dot(np.array([W * S, H * S, S, 1]).reshape(4, 1))[:3].T)
    pts.append(P.dot(np.array([0, H * S, S, 1]).reshape(4, 1))[:3].T)
    pts.append(P.dot(np.array([0, 0, S, 1]).reshape(4, 1))[:3].T)
    pts = np.concatenate(pts, 0)
    pairs = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    # mesh = create_point(pts)
    mesh = create_skeleton(pts, pairs=pairs, cType='c')
    return mesh

def visMeshO3D(mesh):
    import open3d
    if isinstance(mesh, list):
        mesh = trimesh.util.concatenate(mesh)
    meshO3D = mesh.as_open3d
    # meshO3D.vertex_colors = mesh.visual.vertex_colors[:,:3]
    open3d.visualization.draw_geometries([meshO3D])

def create_cameras(Ks = np.array([[1024,0,0,0], [0,1024,0,0], [0,0,1,0],[0,0,0,1]])[None], Es = np.eye(4)[None], S = 1.0, H = 1024,W= 1024, color=None):
    """
        Ks: Intrinsics
        Es: Extrinsic  w2c
    """
    meshes = []
    for idx in range(len(Ks)):
        meshes.append(create_camera(K=Ks[idx], E=Es[idx], S=S, H=H, W=W))
    mesh = trimesh.util.concatenate(meshes)
    if not color is None:
        mesh = Mesh(vertices=mesh.vertices,faces=mesh.faces,vc=color)
    return mesh


def create_box(R=np.eye(3),T = np.zeros((1,3)),S=1.0):
    mesht = trimesh.load(os.path.join(temp2, 'datas', 'vis', 'box2.obj'))
    verts  = np.asarray(mesht.vertices).dot(R.T)
    verts *= S
    verts += T
    mesht2 = Mesh(vertices = verts, faces = mesht.faces )
#     mesht2.show()
    return mesht2

def create_boxs(R=np.eye(3)[None], T = np.zeros((1,3)), S=np.ones([1,3])):
    meshBox = create_box()
    verts, faces = meshBox.vertices, meshBox.faces
    B = R.shape[0]
    # verts  = np.asarray(verts).dot(R.T)
    nVerts = verts.shape[0]

    faces = faces[None, :, :].repeat(B, 0)
    faces = faces + nVerts * np.arange(B).reshape(B, 1, 1)
    faces = np.vstack(faces)

    verts = np.tile(verts , (B,1,1) )

    if not S.shape[0] == B:
        S = np.tile(S, (B,1, 1))
    verts *= S.reshape(B,1,-1)

    # verts = np.einsum('pn,bnm->bpm', verts, np.transpose(R, (0, 2, 1)))
    verts = np.einsum('bpn,bnm->bpm', verts, np.transpose(R, (0, 2, 1)))
    if not T.shape[0] == B:
        T = np.tile(T, (B,1))
    verts += T[:,None]
    mesht2 = Mesh(vertices = verts.reshape(-1,3), faces = np.asarray(faces))
    mesht2.vertex_normals = -mesht2.vertex_normals
    mesht2.face_normals = -mesht2.face_normals
    return mesht2

def create_boxs_voxel(R=np.eye(3)[None], T = np.zeros((1,3)), S=np.ones([1,3]), dim = 16):
    B = R.shape[0]
    s = 0.5
    xx, yy, zz = np.meshgrid(
        np.linspace(-s, s, dim),
        np.linspace(-s, s, dim),
        np.linspace(-s, s, dim)
    )
    verts = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], -1)

    B = R.shape[0]
    # verts  = np.asarray(verts).dot(R.T)
    nVerts = verts.shape[0]

    verts = np.tile(verts , (B,1,1) )

    if not S.shape[0] == B:
        S = np.tile(S, (B,1, 1))
    verts *= S.reshape(B,1,-1)

    # verts = np.einsum('pn,bnm->bpm', verts, np.transpose(R, (0, 2, 1)))
    verts = np.einsum('bpn,bnm->bpm', verts, np.transpose(R, (0, 2, 1)))
    if not T.shape[0] == B:
        T = np.tile(T, (B,1))
    verts += T[:,None]

    return verts.reshape(-1,3)

def create_axis(R=np.eye(3),T = np.zeros((1,3)),S=1.0):
    import open3d as o3d
    mesht = o3d.geometry.TriangleMesh.create_coordinate_frame()
    verts  = np.asarray(mesht.vertices).dot(R.T)
    verts *= S
    verts += T
    mesht2 = Mesh(vertices = verts, faces = np.asarray(mesht.triangles),\
                  vc = np.concatenate([np.asarray(mesht.vertex_colors),np.ones((np.asarray(mesht.vertex_colors).shape[0],1))],-1) )
#     mesht2.show()
    return mesht2


def create_line_fix(start, end, r=0.01, dir = 'z', color = None):
    verts, faces = load_cylinder(cType='c')
    verts = np.array(verts)
    # s = 0.5
    # verts *= s
    # flag1 = ((verts[:,2] > 0.5) and (verts[:,2] < 0.9) )
    # flag2 = verts[:, 2] > 0.8

    # flag1 = np.logical_and((verts[:, 2] > 0.5 * s), (verts[:, 2] < 0.9 * s))
    # flag2 = verts[:, 2] > 0.9 *s
    length = np.linalg.norm(end[:3] - start[:3])
    verts[:, :2] *= r
    verts[:, 2] *= length
    # if length<0.4:
    #     length = 0.4
    # verts[flag1,2] *= (length-0.2 * s)
    # verts[flag2, 2] += (length-0.2 * s ) * 0.8 * s
    # verts[flag2, 2] += 0.2
    rotmat = calRot(np.array([0, 0, 1]), end - start)
    vertices = verts @ rotmat.T + (start + end)/2
    # vertices = verts @ rotmat.T + start
    if not color is None:
        mesh = Mesh(vertices=vertices,faces=faces,vc=color)
    else:
        mesh = Mesh(vertices=vertices,faces=faces)
    return mesh


def set_vertex_colors(mesh, vc, vertex_ids=None):
    def colors_like(color, array, ids):
        # import pdb
        # pdb.set_trace()
        color = np.array(color).astype('uint8')
        if color.max() <= 1.:
            color = color * 255
        # color = color.astype(np.uint8)
        if color.shape[1] == 3:
            color = (np.concatenate([color, np.ones([color.shape[0], 1]) * 255], axis=1)).astype('uint8')
        n_color = color.shape[0]
        n_ids = ids.shape[0]
        new_color = np.array(array)
        # import pdb
        # pdb.set_trace()
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color
        return new_color

    all_ids = np.arange(mesh.vertices.shape[0])
    if vertex_ids is None:
        vertex_ids = all_ids
    vertex_ids = all_ids[vertex_ids]
    new_vc = colors_like(vc, mesh.visual.vertex_colors, vertex_ids)
    mesh.visual.vertex_colors[:] = new_vc

class Mesh(trimesh.Trimesh):
    '''
    Borrow from https://github.com/otaheri/GRAB
    '''
    def __init__(self, filename=None, mesh=None,
                 vertices=None, faces=None,
                 vc=None, fc=None,
                 vscale=None, process=False,
                 visual=None, wireframe=False, smooth=False,
                 **kwargs):
        self.wireframe = wireframe
        self.smooth = smooth

        if filename is not None:
            mesh = trimesh.load(filename, process=process)
            vertices = mesh.vertices, faces = mesh.faces, visual = mesh.visual
        if mesh is not None:
            # mesh = trimesh.load(filename, process=process)
            # vertices = mesh.vertices, faces = mesh.faces, visual = mesh.visual
            vertices = mesh.vertices
            faces = mesh.faces
        if vscale is not None:
            vertices = vertices * vscale
        if faces is None:
            mesh = points2sphere(vertices)
            vertices = mesh.vertices, faces = mesh.faces, visual = mesh.visual

        super(Mesh, self).__init__(vertices=vertices, faces=faces, process=process, visual=visual)

        if vc is not None:
            self.set_vertex_colors(vc)
        if fc is not None:
            self.set_face_colors(fc)

    def rot_verts(self, vertices, rxyz):
        return np.array(vertices * rxyz.T)

    def colors_like(self, color, array, ids):
        color = np.array(color)
        if color.max() <= 1.:
            color = color * 255
        color = color.astype(np.int8)
        n_color = color.shape[0]
        n_ids = ids.shape[0]
        new_color = np.array(array)
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color
        return new_color

    def set_vertex_colors(self, vc, vertex_ids=None):
        all_ids = np.arange(self.vertices.shape[0])
        if vertex_ids is None:
            vertex_ids = all_ids
        vertex_ids = all_ids[vertex_ids]
        new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
        self.visual.vertex_colors[:] = new_vc

    def set_face_colors(self, fc, face_ids=None):
        if face_ids is None:
            face_ids = np.arange(self.faces.shape[0])
        new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
        self.visual.face_colors[:] = new_fc

    @staticmethod
    def cat(meshes):
        return trimesh.util.concatenate(meshes)


# flags = RenderFlags.FLAT
# flags = RenderFlags.NONE
flags = RenderFlags.SHADOWS_DIRECTIONAL
if render_flags['flip_wireframe']:
    flags |= RenderFlags.FLIP_WIREFRAME
elif render_flags['all_wireframe']:
    flags |= RenderFlags.ALL_WIREFRAME
elif render_flags['all_solid']:
    flags |= RenderFlags.ALL_SOLID

if render_flags['shadows']:
    flags |= RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_SPOT
if render_flags['vertex_normals']:
    flags |= RenderFlags.VERTEX_NORMALS
if render_flags['face_normals']:
    flags |= RenderFlags.FACE_NORMALS
if not render_flags['cull_faces']:
    flags |= RenderFlags.SKIP_CULL_FACES
if render_flags['rgba']:
    flags |= RenderFlags.RGBA

flags |= RenderFlags.VERTEX_NORMALS
# flags |= RenderFlags.FLIP_WIREFRAME



class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

    def update_weakcamera(self,scale,translation):
        if scale is not None:
            self.scale = scale
        if translation is not None:
            self.translation = translation


dir_path = os.path.dirname(os.path.realpath(__file__))
# dir_dir_path = os.path.dirname(dir_path)
SHADER_DIR = os.path.join(os.path.dirname(dir_path),'datas','shaders')
class CustomShaderCache():
    def __init__(self, world = False):
        self.program = None
        self.world = world
    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram(SHADER_DIR + r"/normal.vert", SHADER_DIR + r"/normal.frag",
                                                                 defines=defines)
        if self.world:
            self.program = pyrender.shader_program.ShaderProgram(SHADER_DIR + r"/normal_global.vert", SHADER_DIR + r"/normal_global.frag",
                                                                 defines=defines)
        return self.program
    def clear(self):
        self.program.delete()

class CustomShaderCachePhong():
    def __init__(self, type=None):
        self.program = None
        self.type = type
    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.type is None:
            self.program = pyrender.shader_program.ShaderProgram(SHADER_DIR + r"/phong.vert",
                                                                 SHADER_DIR + r"/phong.frag",
                                                                 defines=defines)
        elif self.type=='skeleton':
            self.program = pyrender.shader_program.ShaderProgram(SHADER_DIR + r"/phong_skel.vert", SHADER_DIR + r"/phong_skel.frag",
                                                                 defines=defines)
        elif self.type=='template':
            self.program = pyrender.shader_program.ShaderProgram(SHADER_DIR + r"/phong_temp.vert",
                                                                 SHADER_DIR + r"/phong_temp.frag",
                                                                 defines=defines)
        return self.program
    def clear(self):
        self.program.delete()

camera_types={
    '0': 'PerspectiveCamera',
    '1': 'IntrinsicsCamera',
    '2': 'WeakPerspectiveCamera',
}
# import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
class Renderer():
    def __init__(self, width=512, height=512, focal_length=1000, bg_color=[0.0, 0.0, 0.0, 0.0], flag=0, use_ground=True, ground_shift=[0, 0, 0], camera_type = 0 \
                 ,ground_rotate=None, ground_sqrange = 20, ground_scale=1.0, ground_up='y', use_axis=True, axis_scale = 1.0):
        self.use_offscreen = True
        self.renderer = pyrender.OffscreenRenderer(viewport_width= width,viewport_height = height)
        self.focal_length = focal_length
        self.bg_color = bg_color
        # self.ambient_light = np.array([0.3, 0.3, 0.3, 1.0])
        self.ambient_light = np.array([0.8, 0.8, 0.8, 1.0])
        self.scene = pyrender.Scene(ambient_light=self.ambient_light, bg_color=colorDict['white'], name='scene')
        self.aspect_ratio = float(width) / height
        # import pdb
        # pdb.set_trace()
        self.znear = 0.01
        self.zfar = 10000.0
        if camera_types[str(camera_type)] == 'PerspectiveCamera':
            self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=self.aspect_ratio)
        elif camera_types[str(camera_type)] == 'IntrinsicsCamera':
            # self.camera = pyrender.IntrinsicsCamera(self.focal_length, self.focal_length, height / 2, width / 2, znear=0.1, zfar=30.0)
            self.camera = pyrender.IntrinsicsCamera(self.focal_length, self.focal_length, height / 2, width / 2, znear=0.01, zfar=10000.0)
        elif camera_types[str(camera_type)] == 'WeakPerspectiveCamera':
            self.camera = WeakPerspectiveCamera(scale=[1.0,1.0],translation=[height / 2, width / 2], zfar=1000.0)

        self.scene.add(self.camera, name='camera')
        self.node_mesh = None
        self.node_light = pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=3.0))
        self.node_light2 = pyrender.Node(light=pyrender.DirectionalLight(color=-np.ones(3), intensity=3.0))
        self.aspect_ratio = float(width) / height
        self.camera_pose = np.eye(4)

        self.ground_shift = np.array(ground_shift).astype('float32')
        self.flag = flags
        if use_ground:
            self.scene.add(self.to_pymesh(create_ground_new(sqrange=ground_sqrange,ground_shift=ground_shift,ground_rotate=ground_rotate,scale=ground_scale, up=ground_up)))
        if use_axis:
            self.scene.add(self.to_pymesh(create_axis(S=axis_scale)))
        self.scene.add_node(self.node_light)
        self.ground_rotate = ground_rotate

        # they must all be the same length (set based on first given sequence)
        self.animation_len = -1
        self.animated_seqs = []  # the actual sequence of pyrender meshes

        # background image sequence
        self.img_seq = None
        self.cur_bg_img = None
        # person mask sequence
        self.mask_seq = None
        self.cur_mask = None
        # current index in the animation sequence
        self.animation_frame_idx = 0
        self.animation_render_time = time.time()

        # key callbacks
        self.is_paused = False

    def acquire_render_lock(self):
        if not self.use_offscreen:
            self.viewer.render_lock.acquire()

    def release_render_lock(self):
        if not self.use_offscreen:
            self.viewer.render_lock.release()

    def to_pymesh(self, mesh):
        wireframe = mesh.wireframe if hasattr(mesh, 'wireframe') else False
        smooth = mesh.smooth if hasattr(mesh, 'smooth') else False
        return pyrender.Mesh.from_trimesh(mesh, wireframe=wireframe, smooth=smooth)

    def add_mesh(self, mesh, smooth=False):
        mesh_py = pyrender.Mesh.from_trimesh(mesh, smooth=smooth)
        self.node_mesh = pyrender.Node(mesh=mesh_py, matrix=np.eye(4))
        self.scene.add_node(self.node_mesh)

    def del_mesh(self):
        self.scene.remove_node(self.node_mesh)


    def add_mesh2(self, mesh, smooth=False):
        mesh_py = pyrender.Mesh.from_trimesh(mesh, smooth=smooth)
        self.node_mesh2 = pyrender.Node(mesh=mesh_py, matrix=np.eye(4))
        self.scene.add_node(self.node_mesh2)

    def del_mesh2(self):
        self.scene.remove_node(self.node_mesh2)

    def del_light(self):
        self.scene.remove_node(self.node_light)

    def cv2gl(self, camera_pose):
        camera_pose = (np.diag([1, -1, -1, 1])).dot(camera_pose.transpose())
        camera_pose = camera_pose.transpose()
        return camera_pose

    def set_camera(self, intrinc, extrinc):
        camera_node = list(self.scene.get_nodes(obj=self.camera))[0]
        # print(camera_node.name)
        self.camera.fx = intrinc[0, 0]
        self.camera.fy = intrinc[1, 1]
        self.camera.cx = intrinc[0, 2]
        self.camera.cy = intrinc[1, 2]
        self.camera_pose = self.cv2gl(extrinc)
        self.scene.set_pose(camera_node, pose=self.camera_pose)
        # self.camera.znear = self.znear
        # self.camera.zfar = self.zfar

    def set_weakcamera(self, scale=None, translation=None):
        # self.scene.remove_node(self.camera)
        # self.camera = WeakPerspectiveCamera(scale=scale, translation=translation, zfar=30.0)
        camera_node = list(self.scene.get_nodes(obj=self.camera))[0]
        # print(type(camera_node))
        # print(len(list(self.scene.get_nodes(obj=self.camera))))
        self.camera.update_weakcamera(scale,translation)
        pose = np.eye(4)
        # self.camera_pose = self.cv2gl(pose)
        self.scene.set_pose(camera_node, pose=self.camera_pose)

    def set_pose(self, pose):
        # camera_node = list(self.scene.get_nodes(obj=self.camera))[0]
        camera_node = list(self.scene.get_nodes(obj=self.camera))[0]
        self.camera_pose = self.cv2gl(pose)
        self.scene.set_pose(camera_node, pose=self.camera_pose)

    def render(self, RGBA = False, albedo = False):
        # camera_node = list(self.scene.get_nodes(obj=self.camera))[0]
        #         self.scene.set_pose(camera_node, self.camera_pose)
        #         light = pyrender.DirectionalLight(color=np.array([255,255,255])/255.0, intensity=3.0)
        #         self.scene.add(light, pose=self.camera_pose)
        # self.del_light()
        # print(self.camera_pose)
        # print(self.camera_pose.shape)
        # self.node_light = pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
        #                                 matrix=self.camera_pose)
        # self.scene.add_node(self.node_light)
        if RGBA:
            flag = self.flag | RenderFlags.RGBA
        else:
            flag = self.flag
        # self.renderer._renderer._program_cache = None
        if albedo:
            color, depth = self.renderer.render(self.scene, flags= RenderFlags.FLAT)
        else:
            color, depth = self.renderer.render(self.scene, flags= flag)
        return color, depth

    def render_normal(self, world=False, RGBA = False):
        # camera_node = list(self.scene.get_nodes(obj=self.camera))[0]
        #         self.scene.set_pose(camera_node, self.camera_pose)
        #         light = pyrender.DirectionalLight(color=np.array([255,255,255])/255.0, intensity=3.0)
        #         self.scene.add(light, pose=self.camera_pose)
        self.del_light()
        self.node_light = pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
                                        matrix=self.camera_pose)
        self.scene.add_node(self.node_light)
        if RGBA:
            flag = self.flag | RenderFlags.RGBA
        else:
            flag = self.flag
        self.renderer._renderer._program_cache = CustomShaderCache(world = world)
        # normal, depth = self.renderer.render(self.scene, flags= 512) #FACE_NORMALS

        # flag |= RenderFlags.VERTEX_NORMALS
        normal, depth = self.renderer.render(self.scene, flags= flag) #FACE_NORMALS
        return normal, depth

    def render_phong(self, type=None, RGBA = False):
        # camera_node = list(self.scene.get_nodes(obj=self.camera))[0]
        #         self.scene.set_pose(camera_node, self.camera_pose)
        #         light = pyrender.DirectionalLight(color=np.array([255,255,255])/255.0, intensity=3.0)
        #         self.scene.add(light, pose=self.camera_pose)
        self.del_light()
        self.node_light = pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
                                        matrix=self.camera_pose)
        self.scene.add_node(self.node_light)
        if RGBA:
            flag = self.flag | RenderFlags.RGBA
        else:
            flag = self.flag
        self.renderer._renderer._program_cache = CustomShaderCachePhong(type = type)
        # normal, depth = self.renderer.render(self.scene, flags= 512) #FACE_NORMALS

        # flag |= RenderFlags.VERTEX_NORMALS
        normal, depth = self.renderer.render(self.scene, flags= flag) #FACE_NORMALS
        return normal, depth



    def save_img(self, save_path):
        camera_node = list(self.scene.get_nodes(obj=self.camera))[0]
        #         light = pyrender.DirectionalLight(color=np.array([255,255,255])/255.0, intensity=3.0 )
        #         self.scene.add(light, pose=self.camera_pose)
        self.del_light()
        self.node_light = pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
                                        matrix=self.camera_pose)
        self.scene.add_node(self.node_light)
        color, depth = self.renderer.render(self.scene, self.flag )
        img = Image.fromarray(color[:,:,:3])
        # print(img)
        img.save(save_path)

    def add_mesh_seq(self, mesh_seq):
        '''
        Add a sequence of trimeshes to render.

        - meshes : List of trimesh.trimesh objects giving each frame of the sequence.
        '''

        # ensure same length as other sequences
        cur_seq_len = len(mesh_seq)
        if self.animation_len != -1:
            if cur_seq_len != self.animation_len:
                print('Unexpected sequence length, all sequences must be the same length!')
                return
        else:
            if cur_seq_len > 0:
                self.animation_len = cur_seq_len
            else:
                print('Warning: mesh sequence is length 0!')
                return

        print('Adding mesh sequence with %d frames...' % (cur_seq_len))

        # create sequence of pyrender meshes and save
        pyrender_mesh_seq = []
        for mid, mesh in enumerate(mesh_seq):
            if mid % 200 == 0:
                print('Caching pyrender mesh %d/%d...' % (mid, len(mesh_seq)))
            if isinstance(mesh, trimesh.Trimesh):
                mesh = pyrender.Mesh.from_trimesh(mesh.copy())
                pyrender_mesh_seq.append(mesh)
            else:
                print('Meshes must be from trimesh!')
                return
        self.add_pyrender_mesh_seq(pyrender_mesh_seq, seq_type='mesh')

    def add_pyrender_mesh_seq(self, pyrender_mesh_seq, seq_type='default'):
         # add to the list of sequences to render
        seq_id = len(self.animated_seqs)
        self.animated_seqs.append(pyrender_mesh_seq)
        self.animated_seqs_type.append(seq_type)

        # create the corresponding node in the scene
        self.acquire_render_lock()
        anim_node = self.scene.add(pyrender_mesh_seq[0], 'anim-mesh-%2d'%(seq_id))
        self.animated_nodes.append(anim_node)
        self.release_render_lock()

    def update_frame(self):
        '''
        Update frame to show the current self.animation_frame_idx
        '''
        for seq_idx in range(len(self.animated_seqs)):
            cur_mesh = self.animated_seqs[seq_idx][self.animation_frame_idx]
            # render the current frame of eqch sequence
            self.acquire_render_lock()

            # replace the old mesh
            anim_node = list(self.scene.get_nodes(name='anim-mesh-%2d'%(seq_idx)))
            anim_node = anim_node[0]
            anim_node.mesh = cur_mesh
            # # update camera pc-camera
            # if self.follow_camera and not self.use_intrins: # don't want to reset if we're going from camera view
            #     if self.animated_seqs_type[seq_idx] == 'mesh':
            #         cam_node = list(self.scene.get_nodes(name='pc-camera'))
            #         cam_node = cam_node[0]
            #         mesh_mean = np.mean(cur_mesh.primitives[0].positions, axis=0)
            #         camera_pose = self.get_init_cam_pose()
            #         camera_pose[:3, 3] = camera_pose[:3, 3] + np.array([mesh_mean[0], mesh_mean[1], 0.0])
            #         self.scene.set_pose(cam_node, camera_pose)

            self.release_render_lock()

        # update background img
        if self.img_seq is not None:
            self.acquire_render_lock()
            self.cur_bg_img = self.img_seq[self.animation_frame_idx]
            self.release_render_lock

        # update mask
        if self.mask_seq is not None:
            self.acquire_render_lock()
            self.cur_mask = self.mask_seq[self.animation_frame_idx]
            self.release_render_lock

    def animate(self, fps=30):
        '''
        Starts animating any given mesh sequences. This should be called last after adding
        all desired components to the scene as it is a blocking operation and will run
        until the user exits (or the full video is rendered if offline).
        '''
        if not self.use_offscreen:
            print('=================================')
            print('VIEWER CONTROLS')
            print('p - pause/play')
            print('\",\" and \".\" - step back/forward one frame')
            print('w - wireframe')
            print('h - render shadows')
            print('q - quit')
            print('=================================')

        print('Animating...')
        frame_dur = 1.0 / float(fps)

        # set up init frame
        self.update_frame()
        imgs = []
        while self.use_offscreen or self.renderer.is_active:
            if self.animation_frame_idx % 120 == 0:
                print('Frame %d/%d...' % (self.animation_frame_idx, self.animation_len))

            if not self.use_offscreen:
                sleep_len = frame_dur - (time.time() - self.animation_render_time)
                if sleep_len > 0:
                    time.sleep(sleep_len)
            else:
                # # render frame
                # if not os.path.exists(self.render_path):
                #     os.mkdir(self.render_path)
                #     print('Rendering frames to %s!' % (self.render_path))
                # cur_file_path = os.path.join(self.render_path, 'frame_%08d.%s' % (self.animation_frame_idx, self.img_extn))
                # self.save_snapshot(cur_file_path)
                # color_img, depth_img = self.renderer.render(self.scene, flags=flags)
                color, depth = self.renderer.render(self.scene, self.flag)
                imgs.append(color)

                if self.animation_frame_idx + 1 >= self.animation_len:
                    break

            self.animation_render_time = time.time()
            if self.is_paused:
                self.update_frame() # just in case there's a single frame update
                continue

            self.animation_frame_idx = (self.animation_frame_idx + 1) % self.animation_len
            self.update_frame()

            if self.single_frame:
                break

        self.animation_frame_idx = 0

        return True



def create_ground_new(sqrange=20, ground_shift=[0,0,0], ground_rotate=None, scale = 1.0, up = 'y'):
    center = [0, 0, 0]
    if up=='y':
        xdir = [1, 0, 0]
        ydir = [0, 0, -1]
    elif up=='z':
        xdir = [0, 1, 0]
        ydir = [-1, 0, 0]
    #     xdir=[1, 0, 0]
    #     ydir=[0, 0, -1]
    step = 2
    xrange = sqrange
    yrange = sqrange
    # white = (np.array([1., 1., 1., 1.]) * 255).astype('uint8')
    white = (np.array([193, 210, 240,255])).astype('uint8')
    black = (np.array([0., 0., 0., 1.]) * 255).astype('uint8')
    # black = (np.array([1., 1., 1.,1.])*255).astype('uint8')
    two_sides = True

    if isinstance(center, list):
        center = np.array(center)
        xdir = np.array(xdir)
        ydir = np.array(ydir)
    #     print('[Vis Info] {}, x: {}, y: {}'.format(center, xdir, ydir))
    xdir = xdir * step
    ydir = ydir * step
    vertls, trils, colls = [], [], []
    cnt = 0
    min_x = -xrange if two_sides else 0
    min_y = -yrange if two_sides else 0
    # x_list = np.linspace(min_x, xrange,sqnum*2 if two_sides else sqnum)
    # y_list = np.linspace(min_y, yrange, sqnum * 2 if two_sides else sqnum)
    x_list = np.arange(min_x,xrange)
    y_list = np.arange(min_y,yrange)
    # for i in range(min_x, xrange):
    #     for j in range(min_y, yrange):
    for i in x_list:
        for j in y_list:
            point0 = center + i * xdir + j * ydir
            point1 = center + (i + 1) * xdir + j * ydir
            point2 = center + (i + 1) * xdir + (j + 1) * ydir
            point3 = center + (i) * xdir + (j + 1) * ydir
            point0 = point0 * scale
            point1 = point1 * scale
            point2 = point2 * scale
            point3 = point3 * scale
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                col = white
            else:
                col = black
            vert = np.stack([point0, point1, point2, point3])
            col = np.stack([col for _ in range(vert.shape[0])])
            tri = np.array([[2, 3, 0], [0, 1, 2]]) + vert.shape[0] * cnt
            cnt += 1
            vertls.append(vert)
            trils.append(tri)
            colls.append(col)
    # ground_shift = [0,0,0]
    vertls = np.vstack(vertls)
    trils = np.vstack(trils)
    colls = np.vstack(colls)
    if ground_rotate is not None:
        # R = make_rotate(ground_rotate[0],ground_rotate[1],ground_rotate[2])
        R = ground_rotate
        vertls = vertls.dot(R.T)

    vertls += ground_shift

    # print(R)
    mesh = trimesh.Trimesh(vertices=vertls, faces=trils, process=False)
    mesh.visual.vertex_colors[:] = colls
    return mesh

### Camera tools
def make_rotate(rx, ry, rz, angle = True):
    if angle:
        rx,ry,rz = np.radians(rx),np.radians(ry),np.radians(rz)
    sinX,sinY,sinZ = np.sin(rx),np.sin(ry),np.sin(rz)
    cosX,cosY,cosZ = np.cos(rx),np.cos(ry),np.cos(rz)

    Rx,Ry,Rz = np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))
    Rx[0, 0] = 1.0;Rx[1, 1] = cosX;Rx[1, 2] = -sinX;Rx[2, 1] = sinX;Rx[2, 2] = cosX
    Ry[0, 0] = cosY;Ry[0, 2] = sinY;Ry[1, 1] = 1.0;Ry[2, 0] = -sinY;Ry[2, 2] = cosY
    Rz[0, 0] = cosZ;Rz[0, 1] = -sinZ;Rz[1, 0] = sinZ;Rz[1, 1] = cosZ;Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

def get_points_from_angles(distance, elevation, azimuth, angle=True, newrotate=False):
    if isinstance(distance, float) or isinstance(distance, int):
        if angle:
            elevation = np.radians(elevation)
            azimuth = np.radians(azimuth)
        if newrotate:
            return np.array([
                -distance * np.cos(elevation) * np.cos(azimuth),
                 distance * np.sin(elevation),
                distance * np.cos(elevation) * np.sin(azimuth)
                ])
        else:
            return np.array([
                distance * np.cos(elevation) * np.sin(azimuth),
                distance * np.sin(elevation),
                -distance * np.cos(elevation) * np.cos(azimuth)])
    else:
        if angle:
            elevation = np.pi / 180. * elevation
            azimuth = np.pi / 180. * azimuth
        #
        if newrotate:
            return torch.stack([
                -distance * torch.cos(elevation) * torch.cos(azimuth),
                distance * torch.sin(elevation),
                distance * torch.cos(elevation) * torch.sin(azimuth)
            ]).transpose(1, 0)
        else:
            return torch.stack([
                distance * torch.cos(elevation) * torch.sin(azimuth),
                distance * torch.sin(elevation),
                -distance * torch.cos(elevation) * torch.cos(azimuth)
            ]).transpose(1, 0)


def R_from_ea(elevation, azimuth):
    temp = np.eye(3)
    temp[0, 0] = -1
    temp[1, 1] = -1
    # R = make_rotate(rx = 0, ry = azimuth, rz=0,anlge = True).T.dot(temp)
    R = make_rotate(rx=0, ry=azimuth, rz=0, angle=True).T.dot(temp)
    R = make_rotate(rx=elevation, ry=0, rz=0, angle=True).dot(R)
    return R


def get_extrinc_from_sphere(distance, elevation, azimuth, t_shift=[0, 0, 0],newrotate = None):
    t = get_points_from_angles(distance, elevation, azimuth) + t_shift
    R = R_from_ea(elevation, azimuth)
    t = -R.dot(t)
    E_w2c = np.eye(4)
    E_w2c[:3, :3] = R
    E_w2c[:3, 3:] = t.reshape(3, 1)
    if newrotate is not None:
        R_init = make_rotate(rx=newrotate[0], ry=newrotate[1], rz=newrotate[2], angle=True)
        temp = np.eye(4)
        temp[:3,:3] = R_init
        R_init = temp
        E_w2c = E_w2c.dot(R_init)
    extrinc = np.linalg.inv(E_w2c)
    # extrinc = E_w2c

    return extrinc

if __name__ == '__main__':
    pass
