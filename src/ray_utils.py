import torch
import numpy as np

def get_rays(height, width, focal_length, c2w):
    """
    Generate rays for each pixel in an image.
    
    Args:
        height (int): Image height in pixels.
        width (int): Image width in pixels.
        focal_length (float): Focal length of camera.
        c2w (tensor): Camera-to-world transformation matrix of shape (3, 4) or (4, 4).
        
    Returns:
        origins: Ray origins of shape (height, width, 3).
        directions: Ray directions of shape (height, width, 3).
    """
    # Create grid of pixel coordinates: (height, width, 2)
    i, j = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij'
    )
    
    # Convert pixel coordinates to camera coordinates
    x = (j - width * 0.5) / focal_length
    y = -(i - height * 0.5) / focal_length
    z = -torch.ones_like(x)
    
    # Stack to create directions in camera space
    directions = torch.stack([x, y, z], dim=-1)  # (height, width, 3)
    
    # Extract rotation matrix from c2w
    if c2w.shape[-1] == 4:  # Handle 3x4 or 4x4 matrix
        rotation = c2w[..., :3, :3]
    else:
        rotation = c2w
    
    # Rotate rays to world space
    directions = directions.unsqueeze(-2)  # (height, width, 1, 3)
    directions = directions * rotation  # (height, width, 3, 3)
    directions = directions.sum(dim=-1)  # (height, width, 3)
    
    # Normalize directions
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    # Get ray origins (camera position)
    origins = c2w[..., :3, 3].expand(directions.shape)  # (height, width, 3)
    
    return origins, directions

def sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=True):
    """
    Sample points along each ray with stratified sampling.
    
    Args:
        rays_o: Ray origins of shape (..., 3).
        rays_d: Ray directions of shape (..., 3).
        near: Near bound of the sampling range.
        far: Far bound of the sampling range.
        n_samples: Number of samples per ray.
        perturb: If True, applies stratified sampling.
        
    Returns:
        z_vals: Depths of sampled points along rays.
        pts: 3D coordinates of sampled points.
    """
    # Create sampling bins
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    z_vals = near + t_vals * (far - near)  # (n_samples)
    
    # Add shape dimensions to match rays_o
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])
    
    # Apply stratified sampling
    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=rays_o.device)
        z_vals = lower + (upper - lower) * t_rand
    
    # Calculate 3D sample positions
    # rays_o: (..., 3), rays_d: (..., 3), z_vals: (..., n_samples)
    # pts: (..., n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    return z_vals, pts

def sample_importance(rays_o, rays_d, z_vals, weights, n_importance):
    """
    Perform importance sampling to focus more samples in regions likely to contribute to the final render.
    
    Args:
        rays_o: Ray origins of shape (..., 3).
        rays_d: Ray directions of shape (..., 3).
        z_vals: Depths of sampled points along rays from initial sampling.
        weights: Weights from initial sampling.
        n_importance: Number of additional samples.
        
    Returns:
        z_vals_combined: Combined depths from initial and new sampling.
        pts_combined: Combined 3D points from initial and new sampling.
    """
    # Create sampling bins weighted by rendering weights
    eps = 1e-5
    weights = weights + eps  # Avoid zero weights
    weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize
    
    # Create CDF for inverse transform sampling
    cdf = torch.cumsum(weights, dim=-1)  # (..., n_samples)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # Add 0 at start
    
    # Sample uniformly from [0, 1]
    u = torch.linspace(0., 1., n_importance+1, device=rays_o.device)[:-1]
    u = u.expand(list(cdf.shape[:-1]) + [n_importance])  # (..., n_importance)
    
    # Add random perturbation within each bin
    u = u + torch.rand(u.shape, device=u.device) / n_importance
    
    # Inverse transform sampling to get new samples
    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1]-1)
    
    inds_g = torch.stack([below, above], dim=-1)  # (..., n_importance, 2)
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(*cdf.shape[:-1], n_importance, cdf.shape[-1]), 
                        dim=-1, 
                        index=inds_g)  # (..., n_importance, 2)
    
    z_vals_g = torch.gather(z_vals.unsqueeze(-2).expand(*z_vals.shape[:-1], n_importance, z_vals.shape[-1]), 
                           dim=-1, 
                           index=inds_g)  # (..., n_importance, 2)
    
    # Compute new z_vals
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    z_vals_fine = z_vals_g[..., 0] + t * (z_vals_g[..., 1] - z_vals_g[..., 0])
    
    # Combine coarse and fine samples and sort
    z_vals_combined = torch.cat([z_vals, z_vals_fine], dim=-1)
    _, indices = torch.sort(z_vals_combined, dim=-1)
    z_vals_combined = torch.gather(z_vals_combined, dim=-1, index=indices)
    
    # Calculate 3D sample positions
    pts_combined = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
    
    return z_vals_combined, pts_combined
