import subprocess
import os
import tempfile
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk
import platform

class ShaderSystem:
    """Interface to the C++ shader processor."""
    
    def __init__(self, shader_processor_path=None):
        """Initialize the shader system with path to C++ processor."""
        # Determine appropriate default path based on platform
        if shader_processor_path is None:
            if platform.system() == "Windows":
                # Look in several possible locations for Windows builds
                possible_paths = [
                    "shader_processor/build/Release/shader_processor.exe",
                    "shader_processor/build/Debug/shader_processor.exe",
                    "shader_processor/build/shader_processor.exe",
                    "shader_processor/shader_processor.exe"
                ]
                
                # Use the first path that exists
                for path in possible_paths:
                    if os.path.exists(path):
                        shader_processor_path = path
                        break
                
                # If none found, use default
                if shader_processor_path is None:
                    shader_processor_path = "shader_processor/build/Release/shader_processor.exe"
            else:
                # Unix-like systems
                shader_processor_path = "shader_processor/build/shader_processor"
        
        self.shader_processor_path = shader_processor_path
        self.shader_dir = "shaders"
        
        # Create shader directory if it doesn't exist
        if not os.path.exists(self.shader_dir):
            os.makedirs(self.shader_dir)
            self._create_default_shaders()
            
        # Check if shader processor exists
        if not os.path.exists(self.shader_processor_path):
            print(f"Warning: Shader processor not found at {self.shader_processor_path}")
            print("Using fallback Python-based effects. Please build the C++ shader processor.")
            self.use_cpp = False
        else:
            print(f"Found shader processor at {self.shader_processor_path}")
            self.use_cpp = True
    
    def _create_default_shaders(self):
        """Create default shader files."""
        # Create a default passthrough shader
        default_shader = """#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D u_texture;

void main()
{
    vec4 color = texture(u_texture, TexCoord);
    FragColor = color;
}
"""
        with open(os.path.join(self.shader_dir, "default.frag"), 'w') as f:
            f.write(default_shader)
            
        # Create a toon shader
        toon_shader = """#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D u_texture;
uniform sampler2D u_depth;
uniform float u_levels = 4.0;
uniform float u_edge_threshold = 0.1;

void main()
{
    vec4 color = texture(u_texture, TexCoord);
    
    // Quantize colors
    color.rgb = floor(color.rgb * u_levels) / u_levels;
    
    // Edge detection using depth
    float depth = texture(u_depth, TexCoord).r;
    float depth_right = texture(u_depth, TexCoord + vec2(0.001, 0.0)).r;
    float depth_bottom = texture(u_depth, TexCoord + vec2(0.0, 0.001)).r;
    
    // Calculate edge intensity based on depth discontinuities
    float edge = step(u_edge_threshold, abs(depth - depth_right) + abs(depth - depth_bottom));
    
    // Apply edge as black outline
    color.rgb *= (1.0 - edge);
    
    FragColor = color;
}
"""
        with open(os.path.join(self.shader_dir, "toon.frag"), 'w') as f:
            f.write(toon_shader)
    
    def get_available_shaders(self):
        """Get list of available shaders."""
        return [f for f in os.listdir(self.shader_dir) if f.endswith('.frag') or f.endswith('.glsl')]
    
    def apply_shader(self, image, depth=None, shader_name="default.frag"):
        """Apply shader to image using the C++ processor."""
        shader_path = os.path.join(self.shader_dir, shader_name)
        
        if not os.path.exists(shader_path):
            print(f"Shader {shader_name} not found. Using default shader.")
            shader_path = os.path.join(self.shader_dir, "default.frag")
        
        # Convert images to temporary files
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_in, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_depth, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_out:
            
            in_path = temp_in.name
            depth_path = temp_depth.name if depth is not None else ""
            out_path = temp_out.name
            
            # Save images
            img_pil = Image.fromarray(image)
            img_pil.save(in_path)
            
            if depth is not None:
                # Normalize depth to 0-1 range
                depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-10)
                depth_pil = Image.fromarray((depth_norm * 255).astype(np.uint8))
                depth_pil.save(depth_path)
        
        try:
            if self.use_cpp:
                # Call C++ shader processor
                cmd = [
                    self.shader_processor_path,
                    "--input", in_path,
                    "--output", out_path,
                    "--shader", shader_path
                ]
                
                if depth is not None:
                    cmd.extend(["--depth", depth_path])
                
                result = subprocess.run(cmd, check=True, capture_output=True)
                
                # Check if successful
                if result.returncode != 0:
                    print(f"Warning: Shader processor failed: {result.stderr.decode()}")
                    return image
                
                # Load processed image
                processed_img = np.array(Image.open(out_path))
                return processed_img
            else:
                # Fallback to Python-based effects
                return self._apply_fallback_effect(image, depth, shader_name)
        except Exception as e:
            print(f"Error applying shader: {e}")
            return image
        finally:
            # Clean up temporary files
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(depth_path) and depth is not None:
                os.remove(depth_path)
            if os.path.exists(out_path):
                os.remove(out_path)
    
    def _apply_fallback_effect(self, image, depth, shader_name):
        """Apply a fallback effect when C++ processor is not available."""
        import cv2
        
        # Simple implementations of shader effects using OpenCV
        if "toon" in shader_name.lower():
            # Implement toon shader using OpenCV
            levels = 5
            edge_threshold = 20
            
            # Color quantization
            img_float = image.astype(np.float32)
            img_quantized = np.floor(img_float / 255.0 * levels) / levels * 255.0
            
            # Edge detection
            if depth is not None:
                # Use depth for edges
                depth_norm = depth / np.max(depth)
                edges = cv2.Laplacian(depth_norm, cv2.CV_64F)
                edges = np.abs(edges)
                edges = np.where(edges > 0.05, 255, 0).astype(np.uint8)
            else:
                # Use color for edges
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Laplacian(gray, cv2.CV_64F)
                edges = np.abs(edges)
                edges = np.where(edges > edge_threshold, 255, 0).astype(np.uint8)
                
            # Apply edges
            edges_rgb = np.stack([edges, edges, edges], axis=2)
            result = np.where(edges_rgb > 0, 0, img_quantized.astype(np.uint8))
            return result
        
        # Add more fallback effects here
        
        # Default: return original
        return image
    
    def show_shader_editor(self, image, depth=None):
        """Show interactive shader editor UI."""
        # Create a simple UI for shader selection
        print("Opening shader selection interface...")
        
        from src.post_processor import PostProcessor
        
        # For now, use the existing PostProcessor class
        processor = PostProcessor()
        processed_img = processor.show_editor_ui(image, depth)
        
        # Store the selected effect and parameters for future frames
        self.current_effect = processor.current_effect
        self.params = processor.params.copy()
        
        print(f"Selected effect: {self.current_effect}")
        return processed_img
