import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from .models import NeRF
from .render import volume_render

def train_nerf(config, dataset, save_dir="checkpoints"):
    """
    Train a NeRF model.
    
    Args:
        config: Configuration object
        dataset: NeRF dataset
        save_dir: Directory to save checkpoints
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Use a much smaller batch size for the first few iterations
    initial_batch_size = min(64, config.batch_size)
    print(f"Starting with reduced batch size: {initial_batch_size}")
    
    # Create model
    model = NeRF(config).to(config.device)
    
    # Create optimizer
    params = list(model.parameters())
    
    # Add appearance embeddings to optimization if used
    if config.use_appearance:
        params.append(dataset.appearance_embeddings)
        
    optimizer = optim.Adam(params, lr=config.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                         step_size=config.scheduler_step_size,
                                         gamma=config.scheduler_gamma)
    
    # Training loop
    start_time = time.time()
    pbar = tqdm(range(1, config.num_iterations + 1))
    
    # Metrics
    losses = []
    psnrs = []
    
    for i in pbar:
        # Get random batch of rays
        if i <= 5:  # Use smaller batch for first few iterations
            batch = dataset.get_rays(batch_size=initial_batch_size)
        else:
            batch = dataset.get_rays()
            
        rays_o = batch['rays_o'].to(config.device)
        rays_d = batch['rays_d'].to(config.device)
        target_rgb = batch['rgb'].to(config.device)
        
        # Get appearance embedding if used
        if config.use_appearance:
            appearance_idx = batch['appearance_idx']
            appearance_embedding = dataset.appearance_embeddings[appearance_idx].to(config.device)
            
            # Debug info for first few iterations
            if i <= 3:
                print(f"Appearance embedding shape: {appearance_embedding.shape}")
        else:
            appearance_embedding = None
            
        # Render rays
        rgb, depth, extras = volume_render(
            model, rays_o, rays_d, 
            near=dataset.near, far=dataset.far,
            n_samples=config.num_samples, 
            n_importance=config.num_importance,
            appearance_embedding=appearance_embedding,
            perturb=True
        )
        
        # Compute loss
        loss = nn.functional.mse_loss(rgb, target_rgb)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        if i % config.scheduler_step_size == 0:
            scheduler.step()
        
        # Calculate PSNR for monitoring
        with torch.no_grad():
            psnr = -10. * torch.log10(loss)
            
        # Log metrics
        losses.append(loss.item())
        psnrs.append(psnr.item())
        
        # Update progress bar
        if i % 10 == 0:
            pbar.set_description(
                f"Loss: {loss.item():.5f}, PSNR: {psnr.item():.2f}"
            )
            
        # Save checkpoint periodically
        if i % 1000 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'psnr': psnr.item(),
                'iteration': i
            }
            
            if config.use_appearance:
                checkpoint['appearance_embeddings'] = dataset.appearance_embeddings.data
                
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_{i:06d}.pt'))
            
            # Save a sample render
            with torch.no_grad():
                # Get validation view
                val_data = dataset.get_rays(idx=len(dataset) - 1)
                val_rays_o = val_data['rays_o'].to(config.device)[:1000]
                val_rays_d = val_data['rays_d'].to(config.device)[:1000]
                
                if config.use_appearance:
                    val_appearance_idx = val_data['appearance_idx']
                    val_appearance_embedding = dataset.appearance_embeddings[val_appearance_idx].to(config.device)
                else:
                    val_appearance_embedding = None
                    
                val_rgb, val_depth, _ = volume_render(
                    model, val_rays_o, val_rays_d,
                    near=dataset.near, far=dataset.far,
                    n_samples=config.num_samples,
                    n_importance=config.num_importance,
                    appearance_embedding=val_appearance_embedding,
                    perturb=False
                )
                
                # Save RGB and depth visualizations
                plt.figure(figsize=(10, 5))
                
                plt.subplot(1, 2, 1)
                # Reshape RGB correctly to H×W image format for visualization
                rgb_viz = val_rgb.reshape(-1, 3).cpu().numpy()
                # For the sample visualization, use a square shape
                viz_size = int(np.sqrt(rgb_viz.shape[0]))
                rgb_viz = rgb_viz[:viz_size*viz_size].reshape(viz_size, viz_size, 3)
                plt.imshow(rgb_viz)
                plt.title(f"RGB - Iteration {i}")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                # Reshape depth correctly to H×W image format for visualization
                depth_viz = val_depth.reshape(-1).cpu().numpy()
                # For the sample visualization, use a square shape
                depth_viz = depth_viz[:viz_size*viz_size].reshape(viz_size, viz_size)
                plt.imshow(depth_viz, cmap='viridis')
                plt.title(f"Depth - Iteration {i}")
                plt.colorbar()
                plt.axis('off')
                
                plt.savefig(os.path.join(save_dir, f'render_{i:06d}.png'))
                plt.close()
    
    # Final checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'psnr': psnr.item(),
        'iteration': config.num_iterations
    }
    
    if config.use_appearance:
        checkpoint['appearance_embeddings'] = dataset.appearance_embeddings.data
        
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_final.pt'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(psnrs)
    plt.title('Training PSNR')
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    
    print(f"Training completed in {time.time() - start_time:.2f}s")
    return model
