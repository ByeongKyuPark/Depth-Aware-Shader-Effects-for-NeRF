import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding:
    def __init__(self, num_frequencies, include_input=True):
        """
        Positional encoding for NeRF as described in the original paper.
        """
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
    def __call__(self, x):
        """
        Apply positional encoding to input tensor x.
        
        Args:
            x: Input tensor of shape [..., dim]
            
        Returns:
            Encoded tensor of shape [..., dim * (2 * num_frequencies + include_input)]
        """
        input_dims = x.shape[-1]
        output_dims = 0
        
        if self.include_input:
            output_dims += input_dims
            
        output_dims += input_dims * 2 * self.num_frequencies
        
        # Print shape information for debugging
        batch_shape = list(x.shape[:-1])
        
        results = []
        if self.include_input:
            results.append(x)
            
        for i in range(self.num_frequencies):
            freq = 2**i
            for func in [torch.sin, torch.cos]:
                results.append(func(freq * x))
                
        result = torch.cat(results, dim=-1)
        # Verify output shape
        assert result.shape[-1] == output_dims, f"Expected {output_dims} output dims but got {result.shape[-1]}"
        return result
        
    def output_dim(self, input_dim):
        """Calculate output dimension given input dimension."""
        if self.include_input:
            return input_dim * (1 + 2 * self.num_frequencies)
        else:
            return input_dim * 2 * self.num_frequencies


class NeRF(nn.Module):
    def __init__(self, config):
        """
        Neural Radiance Field model with appearance embedding for NeRF-W.
        
        Args:
            config: Configuration object containing model parameters
        """
        super(NeRF, self).__init__()
        self.config = config
        
        # Positional encoding dimensions
        self.pos_encoder = PositionalEncoding(config.pos_enc_levels)
        self.dir_encoder = PositionalEncoding(config.dir_enc_levels)
        
        # Calculate exact dimensions
        pos_enc_dim = 3 * (1 + 2 * config.pos_enc_levels)  # xyz
        dir_enc_dim = 3 * (1 + 2 * config.dir_enc_levels)  # Direction
        
        print(f"Position encoding dimension: {pos_enc_dim}")
        print(f"Direction encoding dimension: {dir_enc_dim}")
        
        # Network architecture
        self.pts_linears = nn.ModuleList()
        
        # First layer takes encoded position as input
        self.pts_linears.append(nn.Linear(pos_enc_dim, config.hidden_dim))
        
        # Additional hidden layers
        for i in range(1, config.num_layers):
            if i in config.skip_connect_layers:
                self.pts_linears.append(nn.Linear(config.hidden_dim + pos_enc_dim, config.hidden_dim))
            else:
                self.pts_linears.append(nn.Linear(config.hidden_dim, config.hidden_dim))
        
        # Density head
        self.density_head = nn.Linear(config.hidden_dim, 1)
        
        # Direction and color head
        self.dir_linear = nn.Linear(config.hidden_dim + dir_enc_dim, config.hidden_dim // 2)
        
        # Adding appearance embedding
        if config.use_appearance:
            self.appearance_projection = nn.Linear(config.appearance_dim, config.hidden_dim // 2)
            self.rgb_linear = nn.Linear(config.hidden_dim // 2, 3)
        else:
            self.rgb_linear = nn.Linear(config.hidden_dim // 2, 3)
            
    def forward(self, x, d, appearance_embedding=None):
        """
        Forward pass through the network.
        
        Args:
            x: Position input tensor (batch_size, 3)
            d: Direction input tensor (batch_size, 3)
            appearance_embedding: Per-image appearance embedding (batch_size, appearance_dim)
            
        Returns:
            rgb: RGB color (batch_size, 3)
            sigma: Density (batch_size, 1)
        """
        # Print input shapes for debugging
        input_shape = x.shape
        
        # Encode inputs
        encoded_x = self.pos_encoder(x)
        encoded_d = self.dir_encoder(d)
        
        print(f"Input shape: {input_shape}, Encoded shape: {encoded_x.shape}")
        
        # Pass through the MLP backbone
        h = encoded_x
        for i, linear in enumerate(self.pts_linears):
            if i in self.config.skip_connect_layers:
                h = torch.cat([h, encoded_x], dim=-1)
                
            h = linear(h)
            h = F.relu(h)
        
        # Density prediction
        sigma = self.density_head(h)
        sigma = F.relu(sigma)  # Ensure density is non-negative
        
        # Direction and color prediction
        h_dir = torch.cat([h, encoded_d], dim=-1)
        h_dir = self.dir_linear(h_dir)
        h_dir = F.relu(h_dir)
        
        # Add appearance embedding if provided
        if self.config.use_appearance and appearance_embedding is not None:
            # Make sure appearance_embedding has the right shape
            if len(appearance_embedding.shape) == 1:
                appearance_embedding = appearance_embedding.unsqueeze(0)
                
            # Expand if needed
            if appearance_embedding.shape[0] == 1 and h_dir.shape[0] > 1:
                appearance_embedding = appearance_embedding.expand(h_dir.shape[0], -1)
                
            appearance_feature = self.appearance_projection(appearance_embedding)
            h_dir = h_dir + appearance_feature
            
        # RGB prediction
        rgb = self.rgb_linear(h_dir)
        rgb = torch.sigmoid(rgb)  # Ensure RGB is in [0, 1]
        
        return rgb, sigma


class AnimatedNeRF(nn.Module):
    def __init__(self, config):
        super(AnimatedNeRF, self).__init__()
        
        # Positional encoding dimensions
        self.pos_encoder = PositionalEncoding(config.pos_enc_levels)
        self.dir_encoder = PositionalEncoding(config.dir_enc_levels)
        self.time_encoder = PositionalEncoding(config.time_enc_levels)
        
        # Calculate exact dimensions
        pos_enc_dim = 3 * (1 + 2 * config.pos_enc_levels)  # xyz
        dir_enc_dim = 3 * (1 + 2 * config.dir_enc_levels)  # Direction
        time_enc_dim = 1 * (1 + 2 * config.time_enc_levels)  # Time
        
        print(f"Position encoding dimension: {pos_enc_dim}")
        print(f"Direction encoding dimension: {dir_enc_dim}")
        print(f"Time encoding dimension: {time_enc_dim}")
        
        # Network architecture
        self.pts_linears = nn.ModuleList()
        
        # First layer takes encoded position and time as input
        self.pts_linears.append(nn.Linear(pos_enc_dim + time_enc_dim, config.hidden_dim))
        
        # Additional hidden layers
        for i in range(1, config.num_layers):
            if i in config.skip_connect_layers:
                self.pts_linears.append(nn.Linear(config.hidden_dim + pos_enc_dim + time_enc_dim, config.hidden_dim))
            else:
                self.pts_linears.append(nn.Linear(config.hidden_dim, config.hidden_dim))
        
        # Density head
        self.density_head = nn.Linear(config.hidden_dim, 1)
        
        # Direction and color head
        self.dir_linear = nn.Linear(config.hidden_dim + dir_enc_dim, config.hidden_dim // 2)
        
        # Adding appearance embedding
        if config.use_appearance:
            self.appearance_projection = nn.Linear(config.appearance_dim, config.hidden_dim // 2)
            self.rgb_linear = nn.Linear(config.hidden_dim // 2, 3)
        else:
            self.rgb_linear = nn.Linear(config.hidden_dim // 2, 3)
            
    def forward(self, x, d, t, appearance_embedding=None):
        """
        Forward pass with time parameter.
        
        Args:
            x: Position (batch_size, 3)
            d: Direction (batch_size, 3)
            t: Time value from 0 to 1 (batch_size, 1)
            appearance_embedding: Optional appearance conditioning
            
        Returns:
            rgb: RGB color (batch_size, 3)
            sigma: Density (batch_size, 1)
        """
        # Encode inputs
        encoded_x = self.pos_encoder(x)
        encoded_d = self.dir_encoder(d)
        encoded_t = self.time_encoder(t)
        
        # Concatenate position and time encodings
        h = torch.cat([encoded_x, encoded_t], dim=-1)
        
        # Pass through the MLP backbone
        for i, linear in enumerate(self.pts_linears):
            if i in self.config.skip_connect_layers:
                h = torch.cat([h, encoded_x, encoded_t], dim=-1)
                
            h = linear(h)
            h = F.relu(h)
        
        # Density prediction
        sigma = self.density_head(h)
        sigma = F.relu(sigma)  # Ensure density is non-negative
        
        # Direction and color prediction
        h_dir = torch.cat([h, encoded_d], dim=-1)
        h_dir = self.dir_linear(h_dir)
        h_dir = F.relu(h_dir)
        
        # Add appearance embedding if provided
        if self.config.use_appearance and appearance_embedding is not None:
            # Make sure appearance_embedding has the right shape
            if len(appearance_embedding.shape) == 1:
                appearance_embedding = appearance_embedding.unsqueeze(0)
                
            # Expand if needed
            if appearance_embedding.shape[0] == 1 and h_dir.shape[0] > 1:
                appearance_embedding = appearance_embedding.expand(h_dir.shape[0], -1)
                
            appearance_feature = self.appearance_projection(appearance_embedding)
            h_dir = h_dir + appearance_feature
            
        # RGB prediction
        rgb = self.rgb_linear(h_dir)
        rgb = torch.sigmoid(rgb)  # Ensure RGB is in [0, 1]
        
        return rgb, sigma
