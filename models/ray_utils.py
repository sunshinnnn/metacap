import torch
import numpy as np

def cast_rays(ori, dir, z_vals):
    return ori[..., None, :] + z_vals[..., None] * dir[..., None, :]

def unproject_points(depth, mask, H, W, K, E, use_pixel_centers=True):
    """
        Ec2w: c2w
    """
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(np.arange(W, dtype=np.float32) + pixel_center, np.arange(H, dtype=np.float32) + pixel_center,  indexing='xy')
    directions = np.stack([(i - K[0,2]) / K[0,0], (j - K[1,2]) / K[1,1], np.ones_like(i)], -1) # (H, W, 3)

    directions = directions.reshape(-1,3)[mask.reshape(-1)>0]
    depth = depth.reshape(-1,1)[mask.reshape(-1)>0]
    points = directions * depth
    points = points.dot(E[:3,:3].T) + E[:3,3:].reshape(1,3)
    return points.reshape(-1,3)

def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1) # (H, W, 3)
    directions /= torch.linalg.norm(directions,ord = 2, dim = -1, keepdims = True)

    return directions


def get_ray_directions_batch(W, H, K, use_pixel_centers=True):
    """
        K: N,3,3
        directions: N,H,W,3
    """
    pixel_center = 0.5 if use_pixel_centers else 0
    x, y = np.meshgrid(np.arange(W) + pixel_center, np.arange(H) + pixel_center, indexing="xy")
    xy = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float32) # H,W,3
    directions = np.einsum('hwm,kmn->khwn', xy, np.linalg.inv(K).transpose([0,2,1]))  # H,W,3 x C,3,3
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = torch.Tensor(directions)
    return directions

def get_ray_directions_batch_naive(W, H, use_pixel_centers=True):
    """
        K: N,3,3
        directions: N,H,W,3
    """
    pixel_center = 0.5 if use_pixel_centers else 0
    x, y = np.meshgrid(np.arange(W) + pixel_center, np.arange(H) + pixel_center, indexing="xy")
    xy = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float32) # H,W,3
    directions = xy
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = torch.Tensor(directions)
    return directions

def get_rays(directions, c2w, keepdim=True):
    """
        Inputs:
            directions: N,3
            c2w: N,3,4
            keepdim: True
        Outputs:
            rays_o: N,3
            rays_d: N,3
    """
    assert directions.shape[-1] == 3
    assert directions.ndim == 2
    assert c2w.ndim == 3

    rays_d = (c2w[:, :3, :3] @ directions[:, :, None])[:, :, 0]
    rays_o = c2w[:, :3, 3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d

def get_rays_batch(directions, c2w, keepdim=True):
    """
        Inputs:
            directions: B,N,3   -> B,N,3,1
            c2w: B,3,4          -> B,1,
            keepdim: True
        Outputs:
            rays_o: B,N,3
            rays_d: B,N,3
    """
    assert directions.shape[-1] == 3
    assert directions.ndim == 3
    assert c2w.ndim == 4

    rays_d = (c2w[:, :, :3, :3] @ directions[:, :, :, None])[:, :, :, 0]
    rays_o = c2w[:, :, :3, 3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_rays_image(directions, c2w, keepdim=True):
    """
        Inputs:
            rays_o: N,H,W,3 or 1,H,W,3
            rays_d: N,H,W,3 or 1,H,W,3
            c2w: N,4,4 or 1,4,4
        Outputs:
            rays_o: N,H,W,3
            rays_d: N,H,W,3
    """


    """
        Inputs:
            directions: B,N,3   -> B,N,3,1
            c2w: B,3,4          -> B,1,
            keepdim: True
        Outputs:
            rays_o: B,N,3
            rays_d: B,N,3
    """


    # rays_d = (c2w[:, :, :3, :3] @ directions[:, :, :, None])[:, :, :, 0]
    # rays_o = c2w[:, :, :3, 3].expand(rays_d.shape)

    rays_d = (c2w[:, None, None, :3, :3] @ directions[..., None])[..., 0]
    rays_o =  c2w[:, None, None, :3, 3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d

def transform_rays_w2s(rays_o, rays_d, w2s):
    """
        Inputs:
            rays_o: N,3
            rays_d: N,3
            w2s: 1,4,4
        Outputs:
            rays_o: N,3
            rays_d: N,3
    """
    rays_d = (w2s[:, :3, :3] @ rays_d[:, :, None])[:, :, 0]
    rays_o = (w2s[:, :3, :3] @ rays_o[:, :, None])[:, :, 0] + w2s[:,:3,3]
    return rays_o, rays_d


def transform_rays_w2s_image(rays_o, rays_d, w2s):
    """
        Inputs:
            rays_o: N,H,W,3 or 1,H,W,3
            rays_d: N,H,W,3 or 1,H,W,3
            w2s: N,4,4 or 1,4,4
        Outputs:
            rays_o: N,H,W,3
            rays_d: N,H,W,3
    """
    N, H, W = rays_o.shape[:3]
    rays_d = (w2s[:, None, None, :3, :3] @ rays_d[..., None])[..., 0]
    rays_o = (w2s[:, None, None, :3, :3] @ rays_o[..., None])[..., 0]  + w2s[:, None, None, :3, 3]

    # rays_d = (w2s[:, :3, :3] @ rays_d[:, :, None])[:, :, 0]
    # rays_o = (w2s[:, :3, :3] @ rays_o[:, :, None])[:, :, 0] + w2s[:,:3,3]
    return rays_o, rays_d


def transform_rays_w2s_batch(rays_o, rays_d, w2s):
    """
        Inputs:
            rays_o: B,N,3   -> B,N,3,1
            rays_d: B,N,3
            w2s: 1,4,4     -> 1,1,4,4
        Outputs:
            rays_o: B,N,3
            rays_d: B,N,3
    """
    rays_d = (w2s[None, :, :3, :3] @ rays_d[:, :, :, None])[:, :, :, 0]
    rays_o = (w2s[None, :, :3, :3] @ rays_o[:, :, :, None])[:, :, :, 0] + w2s[None, :,:3,3]
    return rays_o, rays_d