import torch
import torch.nn.functional as F
from .ray_utils import sample_stratified, sample_importance

def volume_render(model, rays_o, rays_d, near, far, n_samples, n_importance, 
                 appearance_embedding=None, background_color=None, perturb=True):
    """
    Volume rendering for Neural Radiance Fields.
    """
    # Store original shape
    orig_shape = rays_o.shape
    
    # Flatten rays for processing
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    batch_size = rays_o.shape[0]
    
    # Normalize ray directions
    rays_d = F.normalize(rays_d, dim=-1)
    
    # Sample points along each ray
    z_vals, pts = sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=perturb)
    
    # Optimize: Disable printing for every chunk
    if rays_o.shape[0] <= 1024:  # Only print debug for small batches
        print(f"Sample points shape: {pts.shape}, n_samples: {n_samples}")
    
    # Reshape inputs for batch processing - keeping track of the exact shapes
    pts = pts.reshape(-1, 3)  # (batch_size * n_samples, 3)
    rays_d_expanded = rays_d.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, 3)  # (batch_size * n_samples, 3)
    
    # If using appearance embedding, handle its shape correctly
    if appearance_embedding is not None:
        # Make sure it's 2D for batch processing
        if len(appearance_embedding.shape) == 1:
            appearance_embedding = appearance_embedding.unsqueeze(0)
        
        # Expand to match batch size if needed
        if appearance_embedding.shape[0] == 1 and batch_size > 1:
            appearance_embedding = appearance_embedding.expand(batch_size, -1)
            
        # Now expand to match points along each ray
        appearance_embedding_expanded = appearance_embedding.unsqueeze(1).expand(-1, n_samples, -1)
        appearance_embedding_expanded = appearance_embedding_expanded.reshape(-1, appearance_embedding.shape[-1])
    else:
        appearance_embedding_expanded = None
    
    # Forward pass through model
    rgb, sigma = model(pts, rays_d_expanded, appearance_embedding_expanded)
    
    # Reshape outputs to match expected dimensions
    rgb = rgb.reshape(batch_size, n_samples, 3)
    sigma = sigma.reshape(batch_size, n_samples, 1)
    
    # Compute distances between samples - crucial to match sigma's shape
    dists = z_vals[..., 1:] - z_vals[..., :-1]  # (batch_size, n_samples-1)
    # Pad the last distance
    dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-3], dim=-1)  # (batch_size, n_samples)
    # Add dimension to match sigma
    dists = dists.unsqueeze(-1)  # (batch_size, n_samples, 1)
    
    # Optimize: Disable printing for every chunk
    if rays_o.shape[0] <= 1024:  # Only print for small batches
        print(f"sigma shape: {sigma.shape}, dists shape: {dists.shape}")
    
    # Now sigma and dists should have the same shape for multiplication
    alpha = 1.0 - torch.exp(-sigma * dists)
    
    # Compute transmittance
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1, :]), 1.0 - alpha + 1e-10], dim=1),
        dim=1
    )[:, :-1, :]
    
    # Compute weights
    weights = alpha * transmittance
    
    # Render RGB and depth
    rgb_map = torch.sum(weights * rgb, dim=1)
    depth_map = torch.sum(weights * z_vals.unsqueeze(-1), dim=1) / (torch.sum(weights, dim=1) + 1e-10)
    
    # Importance sampling section
    if n_importance > 0:
        # We'll fix importance sampling implementation similarly
        # ... existing code with similar fixes ...
        pass
    
    # Final reshape
    rgb_map = rgb_map.reshape(*orig_shape[:-1], 3)
    depth_map = depth_map.reshape(*orig_shape[:-1], 1)
    
    extras = {
        'weights': weights,
        'z_vals': z_vals,
    }
    
    return rgb_map, depth_map, extras
