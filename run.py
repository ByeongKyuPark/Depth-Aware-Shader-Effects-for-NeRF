import os
import torch
import argparse
from config import Config
from src.dataset import NeRFDataset
from src.models import NeRF
from src.train import train_nerf
from src.render import volume_render
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='NeRF-W runner')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode: train, render, or video')
    parser.add_argument('--scene', type=str, default='hotdog',
                        help='Scene from NeRF synthetic dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to load for rendering')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for renders')
    parser.add_argument('--use_shader', action='store_true',
                        help='Enable post-processing effects')
    parser.add_argument('--create_video', action='store_true',
                        help='Create a video from rendered images')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the output video (default: 30)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory containing images for video creation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the video file')
    parser.add_argument('--pattern', type=str, default='rgb_*.png',
                        help='Image filename pattern (default: rgb_*.png)')
    parser.add_argument('--shader', type=str, default=None,
                        help='Shader to apply to the rendered images')
    parser.add_argument('--width', type=int, default=800,
                        help='Output image width')
    parser.add_argument('--height', type=int, default=800,
                        help='Output image height') 
    parser.add_argument('--frames', type=int, default=120,
                        help='Number of frames to render')
    parser.add_argument('--quality', type=str, default='high', choices=['preview', 'medium', 'high'],
                        help='Rendering quality preset')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='First frame to render')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='Last frame to render')
    parser.add_argument('--save_depth', action='store_true',
                        help='Save depth maps along with RGB images')
    parser.add_argument('--raw_output', action='store_true',
                        help='Save raw unprocessed renders without applying shaders')
    parser.add_argument('--camera_path', type=str, default='circle', choices=['circle', 'spiral', 'hemisphere', 'horizontal_only'],
                        help='Camera path type for rendering')
    parser.add_argument('--spiral_loops', type=float, default=2.0,
                        help='Number of loops in the spiral path')
    parser.add_argument('--height_range', type=float, nargs=2, default=[-0.5, 0.5],
                        help='Height range for spiral path [min, max]')
    return parser.parse_args()

def render_path(model, dataset, config, output_dir, num_frames=120, use_shader=False, quality='high', 
                width=800, height=800, start_frame=0, end_frame=None, create_video=False, save_depth=False, raw_output=False,
                camera_path='circle', spiral_loops=2.0, height_range=[-0.5, 0.5]):
    """Render a path of novel views."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize shader system if needed
    shader_system = None
    if use_shader:
        try:
            from src.shader_system import ShaderSystem
            shader_system = ShaderSystem()
            print("Shader system initialized")
            
            # Create shaders directory if it doesn't exist
            if not os.path.exists("shaders"):
                os.makedirs("shaders")
                print("Created shaders directory")
                
            # Check if any shaders exist
            if os.path.exists("shaders") and not os.listdir("shaders"):
                print("Warning: No shader files found in shaders directory")
        except Exception as e:
            print(f"Error initializing shader system: {e}")
    
    # Apply quality presets
    if quality == 'preview':
        n_samples = config.num_samples // 2  # Half the samples
        chunk_size = 8192                    # Larger chunks for speed
        perturb = False                      # Skip randomization
    elif quality == 'medium':
        n_samples = config.num_samples
        chunk_size = 4096
        perturb = True
    else:  # high
        n_samples = config.num_samples
        chunk_size = 2048                    # Smaller chunks for better memory handling
        perturb = True
        
    # Set frame range
    if end_frame is None:
        end_frame = num_frames
    
    # Set look-at point and up direction based on the scene
    center = np.array([0, 0, 0])  # Default origin
    up = np.array([0, 1, 0])      # Default up vector (Y-up)
    
    if config.scene == 'lego':
        # Adjust to correct the orientation - the model is facing down by default
        center = np.array([0, 0.5, 0])  
        up = np.array([0, 0, 1])  # Use Z as up direction instead of Y
    
    elif config.scene == 'chair':
        center = np.array([0, 0.5, 0])
    
    # Create camera path based on specified type
    if camera_path == 'circle':
        theta = np.linspace(0, 2*np.pi, num_frames)
        
        if config.scene == 'lego':
            heights = np.zeros_like(theta) + 0.5  # Higher viewpoint
        else:
            heights = np.zeros_like(theta)
            
        phi = np.zeros_like(theta)
    
    elif camera_path == 'spiral':
        theta = np.linspace(0, 2*np.pi * spiral_loops, num_frames)
        
        if config.scene == 'lego':
            height_range = [0.3, 0.7]
        
        heights = np.linspace(height_range[0], height_range[1], num_frames)
        phi = np.zeros_like(theta)
    
    elif camera_path == 'horizontal_only':
        theta = np.linspace(0, 2*np.pi * spiral_loops, num_frames)
        heights = np.full_like(theta, 0.5)
        phi = np.zeros_like(theta)
    
    elif camera_path == 'hemisphere':
        indices = np.arange(0, num_frames, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / num_frames) - np.pi/2
        theta = np.pi * (1 + 5**0.5) * indices
        heights = np.zeros_like(theta)
    
    radius = 4.0
    
    print(f"Rendering {num_frames} frames along {camera_path} path")
    if camera_path == 'spiral':
        print(f"  Spiral loops: {spiral_loops}")
        print(f"  Height range: {height_range}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    model.eval()
    
    print(f"Rendering frames {start_frame} to {end_frame-1} ({len(theta)} total)")
    
    render_H, render_W = height, width
    
    for i, angle in enumerate(tqdm(theta, desc="Rendering frames")):
        frame_idx = start_frame + i
        
        if camera_path in ['circle', 'spiral', 'horizontal_only']:
            cam_pos = np.array([
                radius * np.sin(angle),
                heights[i],
                radius * np.cos(angle)
            ])
        
        elif camera_path == 'hemisphere':
            cam_pos = np.array([
                radius * np.cos(phi[i]) * np.sin(angle),
                radius * np.sin(phi[i]),
                radius * np.cos(phi[i]) * np.cos(angle)
            ])
        
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        camera_up = np.cross(right, forward)
        camera_up = camera_up / np.linalg.norm(camera_up)
        
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = camera_up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos
        
        c2w = torch.tensor(c2w, dtype=torch.float32).to(config.device)
        
        scale_factor = width / dataset.W
        scaled_focal = dataset.focal * scale_factor
        from src.ray_utils import get_rays
        rays_o, rays_d = get_rays(render_H, render_W, scaled_focal, c2w)
        
        if config.use_appearance:
            appearance_embedding = dataset.appearance_embeddings[0].to(config.device)
        else:
            appearance_embedding = None
            
        rgb_chunks = []
        depth_chunks = []
        
        chunk_indices = list(range(0, rays_o.reshape(-1, 3).shape[0], chunk_size))
        for j in tqdm(chunk_indices, desc=f"Frame {frame_idx}/{end_frame-1}", leave=False):
            rays_o_chunk = rays_o.reshape(-1, 3)[j:j+chunk_size].to(config.device)
            rays_d_chunk = rays_d.reshape(-1, 3)[j:j+chunk_size].to(config.device)
            
            with torch.no_grad():
                rgb_chunk, depth_chunk, _ = volume_render(
                    model, rays_o_chunk, rays_d_chunk,
                    near=dataset.near, far=dataset.far,
                    n_samples=n_samples,
                    n_importance=config.num_importance if quality != 'preview' else 0,
                    appearance_embedding=appearance_embedding,
                    perturb=perturb
                )
                
            rgb_chunks.append(rgb_chunk.cpu())
            depth_chunks.append(depth_chunk.cpu())
            
        rgb = torch.cat(rgb_chunks, dim=0).reshape(render_H, render_W, 3)
        depth = torch.cat(depth_chunks, dim=0).reshape(render_H, render_W)
        
        rgb_img = (rgb * 255).numpy().astype(np.uint8)
        depth_img = depth.numpy()
        
        if raw_output or save_depth:
            raw_dir = os.path.join(output_dir, 'raw')
            os.makedirs(raw_dir, exist_ok=True)
            
        if raw_output:
            raw_rgb_img_pil = Image.fromarray(rgb_img)
            raw_rgb_img_pil.save(os.path.join(raw_dir, f'rgb_{frame_idx:03d}.png'))
        
        if save_depth:
            depth_path = os.path.join(raw_dir, f'depth_{frame_idx:03d}.npy')
            np.save(depth_path, depth_img)
        
        if use_shader and shader_system is not None and not raw_output:
            depth_norm = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-6)
            
            try:
                if i == 0:
                    print("\nProcessing first frame with shader editor...")
                    rgb_img = shader_system.show_shader_editor(rgb_img, depth_norm)
                    print("Shader applied to first frame, continuing with batch processing...")
                else:
                    if hasattr(shader_system, 'current_effect'):
                        from src.post_processor import PostProcessor
                        processor = PostProcessor()
                        processor.current_effect = shader_system.current_effect
                        processor.params = shader_system.params.copy()
                        rgb_img = processor.apply_effect(rgb_img, depth_norm)
            except Exception as e:
                print(f"Error applying shader: {e}")
                import traceback
                traceback.print_exc()
        
        rgb_img_pil = Image.fromarray(rgb_img)
        rgb_img_pil.save(os.path.join(output_dir, f'rgb_{frame_idx:03d}.png'))
        
        plt.figure(figsize=(render_W/100, render_H/100), dpi=100)
        plt.imshow(depth.numpy(), cmap='viridis')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'depth_{frame_idx:03d}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        
    print(f"Rendered {len(theta)} frames to {output_dir}")
    
    if create_video:
        from create_video import create_video_from_images
        video_path = os.path.join(output_dir, f"{config.scene}_render.mp4")
        create_video_from_images(output_dir, video_path, pattern='rgb_*.png', fps=30)

def create_video(input_dir, output_path, pattern='rgb_*.png', fps=30):
    """Create a video from an existing image sequence."""
    from create_video import create_video_from_images
    return create_video_from_images(input_dir, output_path, pattern, fps)

def train(model, dataset, config, output_dir=None):
    """Train a NeRF model on a dataset."""
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints_{config.scene}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save checkpoint
    if i % config.checkpoint_interval == 0 or i == config.num_iterations - 1:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{i}.pt')
        torch.save({
            'iteration': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'appearance_embeddings': dataset.appearance_embeddings if config.use_appearance else None
        }, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_final.pt')
    torch.save({
        'iteration': config.num_iterations,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'appearance_embeddings': dataset.appearance_embeddings if config.use_appearance else None
    }, final_checkpoint_path)
    print(f'Saved final checkpoint to {final_checkpoint_path}')

def main():
    args = parse_args()
    config = Config()
    
    config.scene = args.scene
    
    dataset = NeRFDataset(config)
    
    if args.mode == 'train':
        print("Testing model dimensions...")
        model = NeRF(config).to(config.device)
        test_positions = torch.randn(10, 3).to(config.device)
        test_directions = torch.randn(10, 3).to(config.device)
        test_directions = F.normalize(test_directions, dim=-1)
        with torch.no_grad():
            rgb, sigma = model(test_positions, test_directions)
        print(f"✓ Test passed! Output shapes: rgb={rgb.shape}, sigma={sigma.shape}")
        
        if config.use_appearance:
            test_appearance = torch.randn(1, config.appearance_dim).to(config.device)
            try:
                rgb, sigma = model(test_positions, test_directions, test_appearance)
                print(f"✓ Appearance test passed! Output shapes: rgb={rgb.shape}, sigma={sigma.shape}")
            except Exception as e:
                print(f"✗ Error in test: {e}")
                import traceback
                traceback.print_exc()
                return
        
        model = train_nerf(config, dataset)
    
    elif args.mode == 'render':
        # If checkpoint not specified, try to use scene-specific default path
        if not args.checkpoint:
            default_checkpoint = f"checkpoints_{args.scene}/checkpoint_final.pt"
            if os.path.exists(default_checkpoint):
                args.checkpoint = default_checkpoint
                print(f"Using default checkpoint: {args.checkpoint}")
            else:
                print(f"No checkpoint specified and default not found at {default_checkpoint}")
                print("Please specify a checkpoint file using --checkpoint")
                return
        
        model = NeRF(config).to(config.device)
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if config.use_appearance and 'appearance_embeddings' in checkpoint:
            dataset.appearance_embeddings.data = checkpoint['appearance_embeddings']
        
        render_path(model, dataset, config, args.output_dir, 
                    num_frames=args.frames,
                    use_shader=args.use_shader,
                    quality=args.quality,
                    width=args.width, 
                    height=args.height,
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    create_video=args.create_video if hasattr(args, 'create_video') else False,
                    save_depth=args.save_depth,
                    raw_output=args.raw_output,
                    camera_path=args.camera_path,
                    spiral_loops=args.spiral_loops,
                    height_range=args.height_range)
    
    elif args.mode == 'video':
        if not args.input_dir or not os.path.isdir(args.input_dir):
            print(f"Error: Input directory {args.input_dir} does not exist")
            return
        if not args.output:
            args.output = os.path.join(args.input_dir, "render.mp4")
        create_video(args.input_dir, args.output, args.pattern, args.fps)

if __name__ == "__main__":
    main()
