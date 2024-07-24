import gc
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import tinycudann as tcnn
import numpy as np


# Compute barycentric coordinates (u, v, w) for
# point p with respect to triangle (a, b, c)
def Barycentric(p, a, b, c):
    """
        from https://github.com/lingjie0206/Neural_Actor_Main_Code/blob/master/fairnr/data/geometry.py
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return u, v, w


def barycentric_interp(queries, triangles, embeddings):
    """
        from https://github.com/lingjie0206/Neural_Actor_Main_Code/blob/master/fairnr/data/geometry.py
    """
    # queries: B x 3
    # triangles: B x 3 x 3
    # embeddings: B x 3 x D
    P = queries
    A, B, C = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    u, v, w = Barycentric(P, A, B, C)
    values = u[:, None] * embeddings[:, 0] + \
             v[:, None] * embeddings[:, 1] + \
             w[:, None] * embeddings[:, 2]
    return values

def _compute_dot(p1, p2):
    """
        from kaolin
    """
    return p1[..., 0] * p2[..., 0] + \
        p1[..., 1] * p2[..., 1] + \
        p1[..., 2] * p2[..., 2]

def _project_edge(vertex, edge, point):
    """
        from kaolin
    """
    point_vec = point - vertex
    length = _compute_dot(edge, edge)
    return _compute_dot(point_vec, edge) / length

def _project_plane(vertex, normal, point):
    """
        from kaolin
    """
    point_vec = point - vertex
    unit_normal = normal / torch.norm(normal, dim=-1, keepdim=True)
    dist = _compute_dot(point_vec, unit_normal)
    return point - unit_normal * dist.view(-1, 1)

def _is_not_above(vertex, edge, norm, point):
    """
        from kaolin
    """
    edge_norm = torch.cross(norm, edge, dim=-1)
    return _compute_dot(edge_norm.view(1, -1, 3),
                        point.view(-1, 1, 3) - vertex.view(1, -1, 3)) <= 0

def _point_at(vertex, edge, proj):
    """
        from kaolin
    """
    return vertex + edge * proj.view(-1, 1)

def _unbatched_point_to_mesh_interp(points, min_dist_idx, dist_type, face_vertices, embeddings):
    """
        points: N,3
        face_vertices: F,3,3
        embeddings: F,3,8
    """
    num_points = points.shape[0]
    num_faces = face_vertices.shape[0]

    device = points.device
    dtype = points.dtype

    # selected_face_vertices = face_vertices[min_dist_idx]
    selected_face_vertices = face_vertices[min_dist_idx]
    selected_embeddings = embeddings[min_dist_idx]
    v1 = selected_face_vertices[:, 0]
    v2 = selected_face_vertices[:, 1]
    v3 = selected_face_vertices[:, 2]
    e21 = v2 - v1
    e32 = v3 - v2
    e13 = v1 - v3
    normals = -torch.cross(e21, e13)
    uab = _project_edge(v1, e21, points)
    ubc = _project_edge(v2, e32, points)
    uca = _project_edge(v3, e13, points)
    counter_p = torch.zeros((num_points, 3), device=device, dtype=dtype)
    cond = (dist_type == 1)
    counter_p[cond] = v1[cond]
    cond = (dist_type == 2)
    counter_p[cond] = v2[cond]
    cond = (dist_type == 3)
    counter_p[cond] = v3[cond]
    cond = (dist_type == 4)
    counter_p[cond] = _point_at(v1, e21, uab)[cond]
    cond = (dist_type == 5)
    counter_p[cond] = _point_at(v2, e32, ubc)[cond]
    cond = (dist_type == 6)
    counter_p[cond] = _point_at(v3, e13, uca)[cond]
    cond = (dist_type == 0)
    counter_p[cond] = _project_plane(v1, normals, points)[cond]
    min_dist = torch.sum((counter_p - points) ** 2, dim=-1)


    interp_embeddings = barycentric_interp(counter_p, selected_face_vertices, selected_embeddings)

    return interp_embeddings, min_dist

def save_ply(fname, points, faces=None, colors=None):
    if faces is None and colors is None:
        points = points.reshape(-1,3)
        to_save = points
        return np.savetxt(fname,
                  to_save,
                  fmt='%.6f %.6f %.6f',
                  comments='',
                  header=(
                      'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nend_header'.format(points.shape[0]))
                        )
    elif faces is None and not colors is None:
        points = points.reshape(-1,3)
        colors = colors.reshape(-1,3)
        to_save = np.concatenate([points, colors], axis=-1)
        return np.savetxt(fname,
                          to_save,
                          fmt='%.6f %.6f %.6f %d %d %d',
                          comments='',
                          header=(
                      'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header'.format(
                      points.shape[0])))
    elif not faces is None and colors is None:
        points = points.reshape(-1,3)
        faces = faces.reshape(-1,3)
        with open(fname,'w') as f:
            f.write('ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nelement face {:d}\nproperty list uchar int vertex_indices\nend_header\n'.format(
                      points.shape[0],faces.shape[0]))
            for i in range(points.shape[0]):
                f.write('%.6f %.6f %.6f\n'%(points[i,0],points[i,1],points[i,2]))
            for i in range(faces.shape[0]):
                f.write('3 %d %d %d\n'%(faces[i,0],faces[i,1],faces[i,2]))
    elif not faces is None and not colors is None:
        points = points.reshape(-1,3)
        colors = colors.reshape(-1,3)
        faces = faces.reshape(-1,3)
        with open(fname,'w') as f:
            f.write('ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {:d}\nproperty list uchar int vertex_indices\nend_header\n'.format(
                      points.shape[0],faces.shape[0]))
            for i in range(points.shape[0]):
                f.write('%.6f %.6f %.6f %d %d %d\n'%(points[i,0],points[i,1],points[i,2],colors[i,0],colors[i,1],colors[i,2]))
            for i in range(faces.shape[0]):
                f.write('3 %d %d %d\n'%(faces[i,0],faces[i,1],faces[i,2]))

def gradient(sdf, x):
    """
    Derivative of the SDF
    Inputs:
        x [batch_size, num_samples, in_dim]: input points
        sdf [batch_size, num_samples]: SDF value at the input points
    Outputs:
        sdf_grad [batch_size, num_samples, in_dim]: gradient of the SDF at the input points
    """
    sdf_grad = torch.autograd.grad(
        outputs=sdf,
        inputs=x,
        grad_outputs=torch.ones_like(sdf),
        create_graph=True)[0]

    return sdf_grad

def chunk_batch(func, chunk_size, move_to_cpu, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    out = defaultdict(list)
    out_type = None

    for i in range(0, B, chunk_size):
        # print(i, chunk_size)
        out_chunk = func(*([arg[i:i + chunk_size] if isinstance(arg, torch.Tensor) else arg for arg in args] + [kwargs[key]   for key in kwargs.keys()]))
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(f'Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}.')
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            v = v.cpu() if move_to_cpu else v
            out[k].append(v)
    
    if out_type is None:
        return

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply


def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == 'none':
        return lambda x: x
    elif name.startswith('scale'):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0., scale_factor) / scale_factor
    elif name.startswith('clamp'):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0., clamp_max)
    elif name.startswith('mul'):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == 'lin2srgb':
        return lambda x: torch.where(x > 0.0031308, torch.pow(torch.clamp(x, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*x).clamp(0., 1.)
    elif name == 'trunc_exp':
        return trunc_exp
    elif name.startswith('+') or name.startswith('-'):
        return lambda x: x + float(name)
    elif name == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif name == 'tanh':
        return lambda x: torch.tanh(x)
    else:
        return getattr(F, name)


def dot(x, y):
    return torch.sum(x*y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


def scale_anything(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    dat = (dat  - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()
