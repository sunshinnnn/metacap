"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-01-04
"""
import numpy as np
# import torch
import matplotlib.pyplot as plt
import torch

def uniform_laplacian(num_vertices, faces, labelMatrix = None):
    from kaolin.ops.mesh import  adjacency_matrix
    r"""Calculates the uniform laplacian of a mesh.
    :math:`L[i, j] = \frac{1}{num\_neighbours(i)}` if i, j are neighbours.
    :math:`L[i, j] = -1` if i == j. 
    :math:`L[i, j] = 0` otherwise.

    Args:
        num_vertices (int): Number of vertices for the mesh.
        faces (torch.LongTensor):
            Faces of shape :math:`(\text{num_faces}, \text{face_size})` of the mesh.

    Returns:
        (torch.Tensor):
            Uniform laplacian of the mesh of size :math:`(\text{num_vertices}, \text{num_vertices})`
    Example:
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> uniform_laplacian(3, faces)
        tensor([[-1.0000,  0.5000,  0.5000],
                [ 0.5000, -1.0000,  0.5000],
                [ 0.5000,  0.5000, -1.0000]])
    """
    batch_size = faces.shape[0]

    dense_adjacency = adjacency_matrix(num_vertices, faces).to_dense()
    if not labelMatrix is None:
        if isinstance(labelMatrix, np.ndarray):
            dense_adjacency *= torch.Tensor(labelMatrix)
        elif isinstance(labelMatrix, torch.Tensor):
            dense_adjacency *= torch.Tensor(labelMatrix)
        else:
            print('[ERROR] wrong type of labelMatrix')

    # Compute the number of neighbours of each vertex
    num_neighbour = torch.sum(dense_adjacency, dim=1).view(-1, 1)

    L = torch.div(dense_adjacency, num_neighbour)

    torch.diagonal(L)[:] = -1

    # Fill NaN value with 0
    L[torch.isnan(L)] = 0

    return L

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


def point_to_mesh_distance(pointclouds, face_vertices):
    """
        from kaolin
    """
    r"""Computes the distances from pointclouds to meshes (represented by vertices and faces.)
    For each point in the pointcloud, it finds the nearest triangle
    in the mesh, and calculated its distance to that triangle.

    Type 0 indicates the distance is from a point on the surface of the triangle.

    Type 1 to 3 indicates the distance is from a point to a vertices.

    Type 4 to 6 indicates the distance is from a point to an edge.

    Args:
        pointclouds (torch.Tensor):
            pointclouds, of shape :math:`(\text{batch_size}, \text{num_points}, 3)`.
        face_vertices (torch.Tensor):
            vertices of each face of meshes,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, 3})`.

    Returns:
        (torch.Tensor, torch.LongTensor, torch.IntTensor):

            - Distances between pointclouds and meshes,
              of shape :math:`(\text{batch_size}, \text{num_points})`.
            - face indices selected, of shape :math:`(\text{batch_size}, \text{num_points})`.
            - Types of distance of shape :math:`(\text{batch_size}, \text{num_points})`.

    Example:
        >>> from kaolin.ops.mesh import index_vertices_by_faces
        >>> point = torch.tensor([[[0.5, 0.5, 0.5],
        ...                        [3., 4., 5.]]], device='cuda')
        >>> vertices = torch.tensor([[[0., 0., 0.],
        ...                           [0., 1., 0.],
        ...                           [0., 0., 1.]]], device='cuda')
        >>> faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device='cuda')
        >>> face_vertices = index_vertices_by_faces(vertices, faces)
        >>> distance, index, dist_type = point_to_mesh_distance(point, face_vertices)
        >>> distance
        tensor([[ 0.2500, 41.0000]], device='cuda:0')
        >>> index
        tensor([[0, 0]], device='cuda:0')
        >>> dist_type
        tensor([[5, 5]], device='cuda:0', dtype=torch.int32)
    """

    batch_size = pointclouds.shape[0]
    num_points = pointclouds.shape[1]
    device = pointclouds.device
    dtype = pointclouds.dtype

    distance = []
    face_idx = []
    dist_type = []

    for i in range(batch_size):
        if pointclouds.is_cuda:
            cur_dist, cur_face_idx, cur_dist_type = _UnbatchedTriangleDistanceCuda.apply(
                pointclouds[i], face_vertices[i])
        else:
            cur_dist, cur_face_idx, cur_dist_type = _unbatched_naive_point_to_mesh_distance(
                pointclouds[i], face_vertices[i])

        distance.append(cur_dist)
        face_idx.append(cur_face_idx)
        dist_type.append(cur_dist_type)
    return torch.stack(distance, dim=0), torch.stack(face_idx, dim=0), \
        torch.stack(dist_type, dim=0)



def _unbatched_point_to_mesh_interp(points, face_vertices, embeddings):
    """
        points: N,3
        face_vertices: F,3,3
        embeddings: F,3,8
    """
    num_points = points.shape[0]
    num_faces = face_vertices.shape[0]

    device = points.device
    dtype = points.dtype

    v1 = face_vertices[:, 0]
    v2 = face_vertices[:, 1]
    v3 = face_vertices[:, 2]

    e21 = v2 - v1
    e32 = v3 - v2
    e13 = v1 - v3

    normals = -torch.cross(e21, e13)

    uab = _project_edge(v1.view(1, -1, 3), e21.view(1, -1, 3), points.view(-1, 1, 3))
    ubc = _project_edge(v2.view(1, -1, 3), e32.view(1, -1, 3), points.view(-1, 1, 3))
    uca = _project_edge(v3.view(1, -1, 3), e13.view(1, -1, 3), points.view(-1, 1, 3))

    is_type1 = (uca > 1.) & (uab < 0.)
    is_type2 = (uab > 1.) & (ubc < 0.)
    is_type3 = (ubc > 1.) & (uca < 0.)
    is_type4 = (uab >= 0.) & (uab <= 1.) & _is_not_above(v1, e21, normals, points)
    is_type5 = (ubc >= 0.) & (ubc <= 1.) & _is_not_above(v2, e32, normals, points)
    is_type6 = (uca >= 0.) & (uca <= 1.) & _is_not_above(v3, e13, normals, points)
    is_type0 = ~(is_type1 | is_type2 | is_type3 | is_type4 | is_type5 | is_type6)

    face_idx = torch.zeros(num_points, device=device, dtype=torch.long)
    all_closest_points = torch.zeros((num_points, num_faces, 3), device=device,
                                     dtype=dtype)

    all_type0_idx = torch.where(is_type0)
    all_type1_idx = torch.where(is_type1)
    all_type2_idx = torch.where(is_type2)
    all_type3_idx = torch.where(is_type3)
    all_type4_idx = torch.where(is_type4)
    all_type5_idx = torch.where(is_type5)
    all_type6_idx = torch.where(is_type6)

    all_types = is_type1.int() + is_type2.int() * 2 + is_type3.int() * 3 + \
        is_type4.int() * 4 + is_type5.int() * 5 + is_type6.int() * 6

    all_closest_points[all_type0_idx] = _project_plane(
        v1[all_type0_idx[1]], normals[all_type0_idx[1]], points[all_type0_idx[0]])
    all_closest_points[all_type1_idx] = v1.view(-1, 3)[all_type1_idx[1]]
    all_closest_points[all_type2_idx] = v2.view(-1, 3)[all_type2_idx[1]]
    all_closest_points[all_type3_idx] = v3.view(-1, 3)[all_type3_idx[1]]
    all_closest_points[all_type4_idx] = _point_at(v1[all_type4_idx[1]], e21[all_type4_idx[1]],
                                                  uab[all_type4_idx])
    all_closest_points[all_type5_idx] = _point_at(v2[all_type5_idx[1]], e32[all_type5_idx[1]],
                                                  ubc[all_type5_idx])
    all_closest_points[all_type6_idx] = _point_at(v3[all_type6_idx[1]], e13[all_type6_idx[1]],
                                                  uca[all_type6_idx])
    all_vec = (all_closest_points - points.view(-1, 1, 3))
    all_dist = _compute_dot(all_vec, all_vec)

    # _, min_dist_idx = torch.min(all_dist, dim=-1)
    min_dist, min_dist_idx = torch.min(all_dist, dim=-1)
    dist_type = all_types[torch.arange(num_points, device=device), min_dist_idx]
    torch.cuda.synchronize()


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

    return interp_embeddings

def _unbatched_naive_point_to_mesh_distance(points, face_vertices):
    """
        from kaolin
    """
    """
    description of distance type:
        - 0: distance to face
        - 1: distance to vertice 0
        - 2: distance to vertice 1
        - 3: distance to vertice 2
        - 4: distance to edge 0-1
        - 5: distance to edge 1-2
        - 6: distance to edge 2-0

    Args:
        points (torch.Tensor): of shape (num_points, 3).
        faces_vertices (torch.LongTensor): of shape (num_faces, 3, 3).

    Returns:
        (torch.Tensor, torch.LongTensor, torch.IntTensor):

            - distance, of shape (num_points).
            - face_idx, of shape (num_points).
            - distance_type, of shape (num_points).
    """
    num_points = points.shape[0]
    num_faces = face_vertices.shape[0]

    device = points.device
    dtype = points.dtype

    v1 = face_vertices[:, 0]
    v2 = face_vertices[:, 1]
    v3 = face_vertices[:, 2]

    e21 = v2 - v1
    e32 = v3 - v2
    e13 = v1 - v3

    normals = -torch.cross(e21, e13)

    uab = _project_edge(v1.view(1, -1, 3), e21.view(1, -1, 3), points.view(-1, 1, 3))
    ubc = _project_edge(v2.view(1, -1, 3), e32.view(1, -1, 3), points.view(-1, 1, 3))
    uca = _project_edge(v3.view(1, -1, 3), e13.view(1, -1, 3), points.view(-1, 1, 3))

    is_type1 = (uca > 1.) & (uab < 0.)
    is_type2 = (uab > 1.) & (ubc < 0.)
    is_type3 = (ubc > 1.) & (uca < 0.)
    is_type4 = (uab >= 0.) & (uab <= 1.) & _is_not_above(v1, e21, normals, points)
    is_type5 = (ubc >= 0.) & (ubc <= 1.) & _is_not_above(v2, e32, normals, points)
    is_type6 = (uca >= 0.) & (uca <= 1.) & _is_not_above(v3, e13, normals, points)
    is_type0 = ~(is_type1 | is_type2 | is_type3 | is_type4 | is_type5 | is_type6)

    face_idx = torch.zeros(num_points, device=device, dtype=torch.long)
    all_closest_points = torch.zeros((num_points, num_faces, 3), device=device,
                                     dtype=dtype)

    all_type0_idx = torch.where(is_type0)
    all_type1_idx = torch.where(is_type1)
    all_type2_idx = torch.where(is_type2)
    all_type3_idx = torch.where(is_type3)
    all_type4_idx = torch.where(is_type4)
    all_type5_idx = torch.where(is_type5)
    all_type6_idx = torch.where(is_type6)

    all_types = is_type1.int() + is_type2.int() * 2 + is_type3.int() * 3 + \
        is_type4.int() * 4 + is_type5.int() * 5 + is_type6.int() * 6

    all_closest_points[all_type0_idx] = _project_plane(
        v1[all_type0_idx[1]], normals[all_type0_idx[1]], points[all_type0_idx[0]])
    all_closest_points[all_type1_idx] = v1.view(-1, 3)[all_type1_idx[1]]
    all_closest_points[all_type2_idx] = v2.view(-1, 3)[all_type2_idx[1]]
    all_closest_points[all_type3_idx] = v3.view(-1, 3)[all_type3_idx[1]]
    all_closest_points[all_type4_idx] = _point_at(v1[all_type4_idx[1]], e21[all_type4_idx[1]],
                                                  uab[all_type4_idx])
    all_closest_points[all_type5_idx] = _point_at(v2[all_type5_idx[1]], e32[all_type5_idx[1]],
                                                  ubc[all_type5_idx])
    all_closest_points[all_type6_idx] = _point_at(v3[all_type6_idx[1]], e13[all_type6_idx[1]],
                                                  uca[all_type6_idx])
    all_vec = (all_closest_points - points.view(-1, 1, 3))
    all_dist = _compute_dot(all_vec, all_vec)

    # _, min_dist_idx = torch.min(all_dist, dim=-1)
    min_dist, min_dist_idx = torch.min(all_dist, dim=-1)
    dist_type = all_types[torch.arange(num_points, device=device), min_dist_idx]
    torch.cuda.synchronize()

    # selected_face_vertices = face_vertices[min_dist_idx]

    # min_dist = torch.sum((counter_p - points) ** 2, dim=-1)
    # Recompute the shortest distances
    # This reduce the backward pass to the closest faces instead of all faces
    # O(num_points) vs O(num_points * num_faces)
    selected_face_vertices = face_vertices[min_dist_idx]
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

    return min_dist, min_dist_idx, dist_type

def index_vertices_by_faces(vertices_features, faces):
    """
        from kaolin
    """
    r"""Index vertex features to convert per vertex tensor to per vertex per face tensor. 

    Args:
        vertices_features (torch.FloatTensor):
            vertices features, of shape
            :math:`(\text{batch_size}, \text{num_points}, \text{knum})`,
            ``knum`` is feature dimension, the features could be xyz position,
            rgb color, or even neural network features.
        faces (torch.LongTensor):
            face index, of shape :math:`(\text{num_faces}, \text{num_vertices})`.
    Returns:
        (torch.FloatTensor):
            the face features, of shape
            :math:`(\text{batch_size}, \text{num_faces}, \text{num_vertices}, \text{knum})`.
    """
    assert vertices_features.ndim == 3, \
        "vertices_features must have 3 dimensions of shape (batch_size, num_points, knum)"
    assert faces.ndim == 2, "faces must have 2 dimensions of shape (num_faces, num_vertices)"
    input = vertices_features.unsqueeze(2).expand(-1, -1, faces.shape[-1], -1)
    indices = faces[None, ..., None].expand(vertices_features.shape[0], -1, -1, vertices_features.shape[-1])
    return torch.gather(input=input, index=indices, dim=1)

def projectPoints(v3d, K, Ew2c):
    # R = E[:3,:3].T
    # t = -R.dot(E[:3,3:])
    R = Ew2c[:3,:3]
    t = Ew2c[:3,3:]
    v2d = K.dot(np.dot(R,v3d.T) + t.reshape(3,1))
    v2d = v2d/v2d[2:]
    return v2d


def load_off(fname):
    vertices = []
    faces = []

    with open(fname, 'r') as f:
        lines = f.readlines()

    if not lines:
        raise ValueError("The file is empty.")

    if lines[0].strip().lower() != "off":
        raise ValueError("The file is not in OFF format.")

    num_vertices, num_faces, _ = map(int, lines[1].split())

    for line in lines[2:2+num_vertices]:
        vertex = [float(coord) for coord in line.strip().split()]
        vertices.append(vertex)

    for line in lines[2+num_vertices:]:
        face = [int(idx) for idx in line.strip().split()[1:]]  # Skip the number of vertices in the face
        faces.append(face)

    # return {
    #     'vertices': np.array(vertices),
    #     'faces': np.array(faces)
    # }
    return  np.array(vertices), np.array(faces)

def save_off(fname, points, faces):
    """
        fname:
        points:
        faces:
    """
    with open(fname,'w') as f:
        f.write('OFF\n')
        f.write('{} {} 0\n'.format(  len(points), len(faces) ))
        for i in range(len(points)):
            f.write('{} {} {}\n'.format(points[i,0], points[i,1], points[i,2]))
        for i in range(len(faces)):
            f.write('3 {} {} {}\n'.format(faces[i, 0], faces[i, 1], faces[i, 2]))

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


def load_ply(mesh_file):
    vertex_data = []
    face_data = []
    vertNum = -1
    faceNum = -1
    startIdx = -1
    with open(mesh_file,'r') as f:
        data = f.readlines()
        for idx in range(20):
            if data[idx][:14] == 'element vertex':
                vertNum = int(data[idx].split(' ')[-1])
            if data[idx][:12] == 'element face':
                faceNum = int(data[idx].split(' ')[-1])
            if data[idx] == 'end_header\n':
                startIdx = idx
        for idx in range(startIdx+1, startIdx + vertNum + 1):
            values = data[idx].split()
            v = list(map(float, values[0:3]))
            vertex_data.append(v)
        for idx in range(startIdx + vertNum + 1, len(data)):
            values = data[idx].split()
            v = list(map(int, values[1:4]))
            face_data.append(v)
    vertices = np.array(vertex_data)
    faces = np.array(face_data)
    return vertices, faces



def save_obj(file_path, vertices, faces, normals=None, texture_coords=None):
    with open(file_path, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        if normals is not None:
            for normal in normals:
                f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

        if texture_coords is not None:
            for tex_coord in texture_coords:
                f.write(f"vt {tex_coord[0]} {tex_coord[1]}\n")

        for face in faces:
            face_str = "f"
            for vertex_indices in face:

                if normals is not None and texture_coords is not None:
                    # face_str += f" {vertex_indices[0]}/{vertex_indices[1]}/{vertex_indices[2]}"
                    face_str += f" {vertex_indices + 1}/{vertex_indices + 1}/{vertex_indices + 1}"
                elif normals is not None:
                    face_str += f" {vertex_indices[0]}//{vertex_indices[2]}"
                elif texture_coords is not None:
                    face_str += f" {vertex_indices[0]}/{vertex_indices[1]}"
                else:
                    face_str += f" {vertex_indices + 1}"
                # print()
                # for idx in range(3):
                #     face_str += f" {vertex_indices[idx]}"
                # face_str += f" {vertex_indices}"
            f.write(face_str + "\n")




def load_obj_mesh(mesh_file, with_color=False, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []
    color_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
            if with_color:
                c = list(map(float, values[4:7]))
                color_data.append(c)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1
    colors = np.array(color_data)

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_normal and with_color:
        norms = np.array(norm_data)
        # print(norms.shape)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1

        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, colors, norms , face_normals
    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        # norms = np.array(norm_data)
        # norms = normalize_v3(norms)
        # face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    if with_color:
        return vertices, faces, colors

    return vertices, faces

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # print()
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def compute_normal_torch(vertices, faces):
    # import torch
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    # norm = torch.zeros(vertices.shape, dtype=vertices.dtype)
    normals = torch.zeros_like(vertices)  # B,N,3
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[:, faces]  # B,F,N,3
    # n = torch.cross(tris[:, :, :, 1] - tris[:, :, :, 0], tris[:, :, :, 2] - tris[:, :, :, 0])
    n = torch.cross(tris[:, :, 1, :] - tris[:, :, 0, :], tris[:, :, 2, :] - tris[:, :, 0, :])
    # n = torch.cross(tris[..., 1] - tris[..., 0], tris[..., 2] - tris[..., 0])
    # n = torch.cross(tris[..., 1] - tris[..., 0], tris[..., 2] - tris[..., 0])
    # n = torch.cross(tris[..., 1][0] - tris[..., 0][0], tris[..., 2][0] - tris[..., 0][0])
    # print()
    n = torch.nn.functional.normalize(n, dim=-1)
    # n = n / torch.linalg.norm(n, ord = 2,  dim=-1, keepdim = True)

    normals[:, faces[:, 0]] += n
    normals[:, faces[:, 1]] += n
    normals[:, faces[:, 2]] += n
    normals = torch.nn.functional.normalize(normals, dim=-1)
    # print(normals.shape)
    # normals = normals / torch.linalg.norm(normals, ord = 2,  dim=-1, keepdim = True)
    # print(torch.linalg.norm(normals, dim=-1, keepdim = True).shape)

    return normals


# compute tangent and bitangent
def compute_tangent(vertices, faces, normals, uvs=None, faceuvs=None):
    # NOTE: this could be numerically unstable around [0,0,1]
    # but other current solutions are pretty freaky somehow
    c1 = np.cross(normals, np.array([0, 1, 0.0]))
    tan = c1
    normalize_v3(tan)
    btan = np.cross(normals, tan)

    # NOTE: traditional version is below

    # pts_tris = vertices[faces]
    # uv_tris = uvs[faceuvs]

    # W = np.stack([pts_tris[::, 1] - pts_tris[::, 0], pts_tris[::, 2] - pts_tris[::, 0]],2)
    # UV = np.stack([uv_tris[::, 1] - uv_tris[::, 0], uv_tris[::, 2] - uv_tris[::, 0]], 1)

    # for i in range(W.shape[0]):
    #     W[i,::] = W[i,::].dot(np.linalg.inv(UV[i,::]))

    # tan = np.zeros(vertices.shape, dtype=vertices.dtype)
    # tan[faces[:,0]] += W[:,:,0]
    # tan[faces[:,1]] += W[:,:,0]
    # tan[faces[:,2]] += W[:,:,0]

    # btan = np.zeros(vertices.shape, dtype=vertices.dtype)
    # btan[faces[:,0]] += W[:,:,1]
    # btan[faces[:,1]] += W[:,:,1]
    # btan[faces[:,2]] += W[:,:,1]

    # normalize_v3(tan)

    # ndott = np.sum(normals*tan, 1, keepdims=True)
    # tan = tan - ndott * normals

    # normalize_v3(btan)
    # normalize_v3(tan)

    # tan[np.sum(np.cross(normals, tan) * btan, 1) < 0,:] *= -1.0

    return tan, btan

if __name__ == '__main__':
    meshPath = r'Y:\HOIMOCAP2\work\data\Subject0002\tight\smoothCharacter\actor.obj'
    verts, faces = load_obj_mesh(meshPath)

    # normals = compute_normal(verts, faces)
    # normalsTorch = compute_normal_torch(torch.Tensor(verts)[None], torch.Tensor(faces).to(torch.long))
    point = torch.tensor([[[0.5, 0.5, 0.5],
                           [0.25, 0.25, 0.25],
                           [0, 0, 1.1],
                           [3., 4., 5.]]], device='cpu')

    # vertices = torch.tensor([[[0., 0., 0.],
    #                           [0., 1., 0.],
    #                           [0., 0., 1.]]], device='cpu')
    # faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device='cpu')

    vertices = torch.tensor([[[0., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 1.],
                              [2., 1., 0.],]], device='cpu')
    faces = torch.tensor([[0, 1, 2], [3, 1, 2]], dtype=torch.long, device='cpu')

    face_vertices = index_vertices_by_faces(vertices, faces)
    distance, index, dist_type = point_to_mesh_distance(point, face_vertices)
    print()
    face_vertices2 = index_vertices_by_faces(torch.Tensor(verts)[None], torch.Tensor(faces).to(torch.long))

    verts = torch.Tensor(verts,  device ='cpu')
    faces = torch.Tensor(faces,  device= 'cpu').to(torch.long)
    face_vertices = verts[faces]
    # embed = torch.randn(verts.shape[0], 8 )
    embed = torch.randn(verts.shape[0], 4, 4)
    face_embed = embed[faces]
    embed_interp = _unbatched_point_to_mesh_interp(point[0], face_vertices, face_embed)

