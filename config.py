import torch

class Config:
    # Dataset parameters
    dataset_type = 'nerf_synthetic'  # New parameter to specify dataset type
    dataset_path = 'data/nerf_synthetic'  # Update this with your actual path
    scene = 'lego'  # Options: chair, drums, ficus, hotdog, lego, materials, mic, ship (download from NeRF repo)
    
    # Model parameters
    hidden_dim = 256
    num_layers = 8
    skip_connect_layers = [4]
    num_samples = 64  # Samples per ray (coarse)
    num_importance = 64  # Additional samples per ray (fine)
    
    # Appearance embedding parameters
    use_appearance = True
    appearance_dim = 32
    
    # Training parameters - adjusted for synthetic data
    batch_size = 1024  # Rays per batch
    learning_rate = 5e-4
    num_iterations = 30000
    scheduler_step_size = 10000  # Learning rate scheduler step size
    scheduler_gamma = 0.5        # Learning rate decay factor
    
    # Updated bounds for synthetic scenes
    near = 2.0
    far = 6.0
    
    # Encoding parameters
    pos_enc_levels = 10  # Number of frequency levels for position
    dir_enc_levels = 4   # Number of frequency levels for direction
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
