import cv2
import os
import glob
from tqdm import tqdm
import argparse
import numpy as np

def create_video_from_images(image_dir, output_path, pattern='rgb_*.png', fps=30, quality=95, resolution=None):
    """
    Create a video file from a sequence of images.
    
    Args:
        image_dir: Directory containing the image sequence
        output_path: Path for the output video file
        pattern: Filename pattern to match (default: 'rgb_*.png')
        fps: Frames per second for the video (default: 30)
        quality: Video quality, 0-100 (default: 95)
        resolution: Optional (width, height) to resize images
    """
    # Find all matching images
    images = sorted(glob.glob(os.path.join(image_dir, pattern)))
    
    if not images:
        print(f"No images found matching pattern '{pattern}' in directory: {image_dir}")
        return False
    
    # Read the first image to get dimensions
    first_img = cv2.imread(images[0])
    if first_img is None:
        print(f"Failed to read image: {images[0]}")
        return False
    
    # Handle resolution
    if resolution:
        width, height = resolution
    else:
        height, width, _ = first_img.shape
    
    # Determine the output video format based on file extension
    _, ext = os.path.splitext(output_path)
    if ext.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Use the file extension
        video_ext = ext.lower()
    else:
        # Default to MP4
        video_ext = '.mp4'
        output_path = os.path.splitext(output_path)[0] + video_ext
    
    # Set the codec based on the output format
    if video_ext == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif video_ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif video_ext in ['.mov', '.mkv']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Most compatible
    
    # Create video writer
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each image to the video
    print(f"Creating video from {len(images)} images...")
    for img_path in tqdm(images):
        img = cv2.imread(img_path)
        if img is not None:
            # Resize if needed
            if resolution:
                img = cv2.resize(img, resolution)
            video.write(img)
    
    # Release resources
    video.release()
    
    print(f"Video created successfully: {output_path}")
    print(f"Duration: {len(images) / fps:.2f} seconds ({len(images)} frames at {fps} FPS)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Create a video from a sequence of images.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the image sequence')
    parser.add_argument('--output', type=str, required=True, help='Output video file path')
    parser.add_argument('--pattern', type=str, default='rgb_*.png', help='Image filename pattern (default: rgb_*.png)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--quality', type=int, default=95, help='Video quality, 0-100 (default: 95)')
    parser.add_argument('--resolution', type=int, nargs=2, help='Optional output resolution as width height')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    create_video_from_images(
        args.input_dir, 
        args.output, 
        args.pattern, 
        args.fps, 
        args.quality,
        args.resolution
    )

if __name__ == "__main__":
    main()
