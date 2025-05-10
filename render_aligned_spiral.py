import torch
import numpy as np
from PIL import Image
import os
import argparse
import math
from tqdm import tqdm

from src.models import NeRF
from src.dataset import NeRFDataset
from src.ray_utils import get_rays
from src.render import volume_render
from config import Config

def render_aligned_spiral(model, dataset, config, output_dir, num_frames=120, fps=60, loops=2, rotation_axis='x'):
    """Render a spiral path around the object with proper alignment."""
    # Set device
    device = config.device
    
    # Ensure output is organized under the output directory
    if not output_dir.startswith('output/'):
        output_dir = os.path.join('output', output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Center point and camera distance
    center = np.array([0, 0, 0])
    radius = 4.0
    
    # Define alignment correction based on rotation parameter
    if rotation_axis == 'x':
        # X-axis rotation (90 degrees) - good for chair
        rot_angle = math.pi/2
        alignment_matrix = np.array([
            [1, 0, 0],
            [0, math.cos(rot_angle), -math.sin(rot_angle)],
            [0, math.sin(rot_angle), math.cos(rot_angle)]
        ])
        # Adjust center point for chair if needed
        if config.scene == 'chair':
            center = np.array([0, 0.5, 0])
    elif rotation_axis == 'y':
        # Y-axis rotation (90 degrees)
        rot_angle = math.pi/2
        alignment_matrix = np.array([
            [math.cos(rot_angle), 0, math.sin(rot_angle)],
            [0, 1, 0],
            [-math.sin(rot_angle), 0, math.cos(rot_angle)]
        ])
    elif rotation_axis == 'z':
        # Z-axis rotation (90 degrees) - good for hotdog
        rot_angle = math.pi/2
        alignment_matrix = np.array([
            [math.cos(rot_angle), -math.sin(rot_angle), 0],
            [math.sin(rot_angle), math.cos(rot_angle), 0],
            [0, 0, 1]
        ])
    else:  # 'none'
        # No rotation
        alignment_matrix = np.eye(3)
    
    # Define up vector (default Y-up)
    up = np.array([0, 1, 0])
    
    # Create spiral path
    theta = np.linspace(0, 2 * math.pi * loops, num_frames)
    
    # Vertical oscillation - up and down during the spiral
    # Range from slightly below to slightly above the center
    phi = np.linspace(-0.3, 0.3, num_frames)
    
    # Render each frame along the spiral
    model.eval()
    print(f"Rendering {num_frames} frames on aligned spiral path...")
    
    for i in tqdm(range(num_frames)):
        # Calculate camera position in a spiral path with vertical oscillation
        angle = theta[i]
        height = phi[i]
        
        # Base camera position before alignment
        base_cam_pos = np.array([
            radius * math.sin(angle),
            height * radius,  # Vertical variation
            radius * math.cos(angle)
        ])
        
        # Apply alignment correction
        cam_pos = alignment_matrix @ base_cam_pos
        
        # Set up vector (corrected for alignment)
        aligned_up = alignment_matrix @ up
        
        # Create camera-to-world matrix
        forward = center - cam_pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-10:
            forward = np.array([0, 0, -1])
        else:
            forward = forward / forward_norm
            
        right = np.cross(forward, aligned_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-10:
            # Handle degenerate case
            right = np.array([1, 0, 0])
        else:
            right = right / right_norm
            
        camera_up = np.cross(right, forward)
        camera_up_norm = np.linalg.norm(camera_up)
        if camera_up_norm < 1e-10:
            camera_up = aligned_up
        else:
            camera_up = camera_up / camera_up_norm
        
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = camera_up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos
        
        c2w = torch.tensor(c2w, dtype=torch.float32).to(device)
        
        # Generate rays
        H, W = dataset.H, dataset.W
        rays_o, rays_d = get_rays(H, W, dataset.focal, c2w)
        
        # Appearance embedding if used
        appearance_embedding = None
        if config.use_appearance:
            appearance_embedding = dataset.appearance_embeddings[0].to(device)
        
        # Render in chunks
        chunk_size = 4096
        rgb_chunks = []
        depth_chunks = []
        
        with torch.no_grad():
            for j in range(0, rays_o.reshape(-1, 3).shape[0], chunk_size):
                rays_o_chunk = rays_o.reshape(-1, 3)[j:j+chunk_size].to(device)
                rays_d_chunk = rays_d.reshape(-1, 3)[j:j+chunk_size].to(device)
                
                rgb_chunk, depth_chunk, _ = volume_render(
                    model, rays_o_chunk, rays_d_chunk,
                    near=dataset.near, far=dataset.far,
                    n_samples=config.num_samples,
                    n_importance=config.num_importance,
                    appearance_embedding=appearance_embedding,
                    perturb=False
                )
                
                rgb_chunks.append(rgb_chunk.cpu())
                depth_chunks.append(depth_chunk.cpu())
        
        # Combine chunks
        rgb = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3)
        depth = torch.cat(depth_chunks, dim=0).reshape(H, W)
        
        # Convert to numpy arrays
        rgb_img = (rgb * 255).numpy().astype(np.uint8)
        
        # Save RGB image
        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        Image.fromarray(rgb_img).save(output_path)
        
        # Save depth visualization (optional)
        if i % 10 == 0:  # Save depth every 10 frames to save space
            depth_img = depth.numpy()
            depth_norm = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min()) * 255
            depth_norm = depth_norm.astype(np.uint8)
            
            depth_path = os.path.join(output_dir, f"depth_{i:04d}.png")
            Image.fromarray(depth_norm).save(depth_path)
    
    # Create video file
    try:
        import cv2
        video_path = os.path.join(output_dir, f"{config.scene}_spiral.mp4")
        print(f"Creating video at {fps} fps: {video_path}")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(output_dir, "frame_0000.png"))
        h, w, _ = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        
        # Add each frame to video
        for i in range(num_frames):
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            frame = cv2.imread(frame_path)
            if frame is not None:
                video.write(frame)
            else:
                print(f"Warning: Could not read frame {frame_path}")
            
        # Release resources
        video.release()
        print(f"Video created successfully: duration {num_frames/fps:.2f} seconds")
    except ImportError:
        print("OpenCV not found. Skipping video creation. Install with: pip install opencv-python")
        print(f"You can create video manually from frames in {output_dir}")
    except Exception as e:
        print(f"Error creating video: {e}")
        print(f"You can create video manually from frames in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Render aligned spiral path around object')
    parser.add_argument('--scene', type=str, default='chair', help='Scene name')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
    parser.add_argument('--output_dir', type=str, default='spiral_render', help='Output directory')
    parser.add_argument('--frames', type=int, default=120, help='Number of frames')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for output video')
    parser.add_argument('--loops', type=int, default=2, help='Number of loops around the object')
    parser.add_argument('--rotation', type=str, default='x', choices=['x', 'y', 'z', 'none'], 
                       help='Rotation axis for alignment (x, y, z, or none)')
    
    args = parser.parse_args()
    
    # Load config and set scene
    config = Config()
    config.scene = args.scene
    
    # If checkpoint not specified, try to use scene-specific default path
    if not args.checkpoint:
        default_checkpoint = f"checkpoints_{args.scene}/checkpoint_final.pt"
        if os.path.exists(default_checkpoint):
            args.checkpoint = default_checkpoint
            print(f"Using default checkpoint: {args.checkpoint}")
        else:
            print(f"No checkpoint specified and default not found at {default_checkpoint}")
            parser.print_help()
            return
    
    # Create dataset
    dataset = NeRFDataset(config)
    
    # Create and load model
    model = NeRF(config).to(config.device)
    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load appearance embeddings if available
    if config.use_appearance and 'appearance_embeddings' in checkpoint:
        dataset.appearance_embeddings.data = checkpoint['appearance_embeddings']
    
    # Render aligned spiral
    render_aligned_spiral(model, dataset, config, args.output_dir, args.frames, args.fps, args.loops, args.rotation)

if __name__ == '__main__':
    main()
