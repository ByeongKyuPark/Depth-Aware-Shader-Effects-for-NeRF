import argparse
import os
from reconstruction_pipeline import ReconstructionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Indoor 3D Reconstruction')
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Directory to store output files')
    parser.add_argument('--feature_type', default='sift', choices=['sift', 'orb'], help='Feature detection algorithm')
    parser.add_argument('--skip_dense', action='store_true', help='Skip dense reconstruction')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize reconstruction pipeline
    pipeline = ReconstructionPipeline(
        feature_type=args.feature_type,
        skip_dense=args.skip_dense
    )
    
    # Run the reconstruction pipeline
    pipeline.run(args.input_dir, args.output_dir)
    
    print(f"Reconstruction complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
