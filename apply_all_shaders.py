import os
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
import cv2
import shutil

from src.post_processor import PostProcessor

def find_frames_with_depth(input_dir):
    """Find frame numbers that have corresponding depth maps."""
    depth_files = [f for f in os.listdir(input_dir) if f.startswith('depth_') and f.endswith('.png')]
    frame_numbers = [f.split('_')[1].split('.')[0] for f in depth_files]
    return sorted(frame_numbers)

def apply_all_shader_effects(input_dir, output_base_dir, args):
    """Apply all shader effects to rendered images and create videos for each."""
    # Create post processor
    processor = PostProcessor()
    
    # Get list of all shader effects
    effects = list(processor.effects.keys())
    print(f"Found {len(effects)} shader effects to apply")
    
    # Find all RGB images in the input directory
    all_rgb_files = sorted([f for f in os.listdir(input_dir) if f.startswith('frame_') and f.endswith('.png')])
    
    if not all_rgb_files:
        print(f"No frame images found in {input_dir}")
        return
    
    print(f"Found {len(all_rgb_files)} total frames")
    
    # Find frames that have depth maps
    frame_numbers_with_depth = find_frames_with_depth(input_dir)
    depth_rgb_files = [f"frame_{num}.png" for num in frame_numbers_with_depth]
    
    print(f"Found {len(frame_numbers_with_depth)} frames with depth maps")
    
    # Check for existing original videos first
    original_video_path = os.path.join(output_base_dir, "original.mp4")
    original_depth_video_path = os.path.join(output_base_dir, "original_depth_available.mp4")
    
    # Process original frames if video doesn't exist
    if not os.path.exists(original_video_path):
        # Create a directory for the original frames
        original_dir = os.path.join(output_base_dir, "original")
        os.makedirs(original_dir, exist_ok=True)
        
        # Copy the original frames
        print(f"Copying original frames for comparison...")
        for rgb_file in all_rgb_files:
            src_path = os.path.join(input_dir, rgb_file)
            dst_path = os.path.join(original_dir, rgb_file)
            shutil.copy2(src_path, dst_path)
        
        # Create video of all original frames
        create_video(original_dir, original_video_path)
    else:
        print(f"Skipping original video creation as it already exists: {original_video_path}")
    
    # Process original frames with depth if that video doesn't exist
    if not os.path.exists(original_depth_video_path):
        # Create directory for original frames with depth
        original_depth_dir = os.path.join(output_base_dir, "original_depth_available")
        os.makedirs(original_depth_dir, exist_ok=True)
        
        # Copy only the frames with depth maps
        print(f"Copying original frames that have depth maps...")
        for rgb_file in depth_rgb_files:
            src_path = os.path.join(input_dir, rgb_file)
            dst_path = os.path.join(original_depth_dir, rgb_file)
            shutil.copy2(src_path, dst_path)
        
        # Create video of original frames with depth
        create_video(original_depth_dir, original_depth_video_path)
    else:
        print(f"Skipping depth-available original video creation as it already exists: {original_depth_video_path}")
    
    # Try to find depth maps for better effects
    depth_files = {}
    for rgb_file in all_rgb_files:
        frame_num = rgb_file.split('_')[1].split('.')[0]
        depth_file = f"depth_{frame_num}.png"
        if os.path.exists(os.path.join(input_dir, depth_file)):
            depth_files[rgb_file] = depth_file
    
    print(f"Matched {len(depth_files)} frames with their depth maps")
    
    # For each effect, check if the video already exists before processing
    for effect_name in effects:
        # Skip any non-fog effects if fog_only flag is set
        if args.fog_only and effect_name != "Fog":
            continue
            
        # Get video path for this effect
        video_path = os.path.join(output_base_dir, f"{effect_name.lower().replace(' ', '_')}.mp4")
        
        # Skip if video already exists
        if os.path.exists(video_path):
            print(f"Skipping effect '{effect_name}' as video already exists: {video_path}")
            continue
            
        print(f"Processing effect: {effect_name}")
        
        # Create effect directory
        effect_dir = os.path.join(output_base_dir, effect_name.lower().replace(' ', '_'))
        os.makedirs(effect_dir, exist_ok=True)
        
        # Set current effect
        processor.current_effect = effect_name
        
        # For fog effect, use only frames with depth
        if effect_name == "Fog":
            rgb_files = depth_rgb_files
            print(f"Using only {len(rgb_files)} frames with depth maps for fog effect")
        else:
            rgb_files = all_rgb_files
        
        # Process each frame with this effect
        for rgb_file in tqdm(rgb_files, desc=f"Applying {effect_name}"):
            # Load RGB image
            rgb_path = os.path.join(input_dir, rgb_file)
            rgb_img = np.array(Image.open(rgb_path))
            
            # Load depth map if available
            depth_img = None
            if rgb_file in depth_files:
                depth_path = os.path.join(input_dir, depth_files[rgb_file])
                depth_img = np.array(Image.open(depth_path))
                # Normalize depth
                depth_img = depth_img / 255.0
            
            # Apply effect to this frame
            processed_img = processor.apply_effect(rgb_img, depth_img)
            
            # Save processed frame
            output_path = os.path.join(effect_dir, rgb_file)
            Image.fromarray(processed_img).save(output_path)
        
        # Create video for this effect
        create_video(effect_dir, video_path)

def create_video(frame_dir, video_path, fps=60):
    """Create a video from a directory of frames."""
    print(f"Creating video: {video_path}")
    
    # Get list of frames
    frames = sorted([f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.png')])
    
    if not frames:
        print(f"No frames found in {frame_dir}")
        return
    
    print(f"Using {len(frames)} frames for video")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frame_dir, frames[0]))
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Add each frame to the video
    for frame_file in tqdm(frames, desc="Creating video"):
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            video.write(frame)
    
    # Release video writer
    video.release()
    print(f"Video created: {video_path}")

def main():
    parser = argparse.ArgumentParser(description='Apply shader effects to rendered images')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with rendered frames')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for effects (defaults to input_dir + "_effects")')
    parser.add_argument('--skip_effects', type=str, nargs='+', default=[], help='List of effects to skip')
    parser.add_argument('--fog_only', action='store_true', help='Only process fog effect')
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = args.input_dir + "_effects"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Apply all effects and create videos
    apply_all_shader_effects(args.input_dir, args.output_dir, args)

if __name__ == "__main__":
    main()
