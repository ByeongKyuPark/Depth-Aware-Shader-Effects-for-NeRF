import os
import sys

def ensure_directories():
    """Ensure required directories exist."""
    required_dirs = [
        'checkpoints',
        'output',
        'shaders',
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Creating missing directory: {directory}")
            os.makedirs(directory)
        else:
            print(f"Directory exists: {directory}")
    
    print("\nProject directories verified.")
    
    # Check for trained models
    if os.path.exists('checkpoints'):
        checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pt')]
        if checkpoints:
            print("\nAvailable trained models:")
            scenes = set()
            for ckpt in checkpoints:
                scene = ckpt.split('_')[0]
                scenes.add(scene)
                
            for scene in scenes:
                final_ckpt = f"{scene}_final.pt"
                if final_ckpt in checkpoints:
                    print(f"  - {scene} (fully trained)")
                else:
                    iterations = [int(c.split('_')[1].split('.')[0]) for c in checkpoints if c.startswith(f"{scene}_")]
                    if iterations:
                        print(f"  - {scene} (partially trained, max iteration: {max(iterations)})")
        else:
            print("\nNo trained models found. You need to train models before rendering.")
            print("Train a model with: python run.py --mode train --scene hotdog")

if __name__ == "__main__":
    ensure_directories()
    
    print("\nTo train a new model:")
    print("  python run.py --mode train --scene hotdog")
    print("\nTo render an existing model:")
    print("  python run.py --mode render --scene hotdog --checkpoint checkpoints/hotdog_final.pt")
```
