import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time

class PostProcessor:
    """Simple image post-processor using NumPy/OpenCV instead of shaders."""
    
    def __init__(self):
        """Simple image post-processor using NumPy/OpenCV instead of shaders."""
        
        # Standard effects (original versions, all handle depth=None gracefully)
        self.effects = {
            "Original": self._effect_original,
            "Toon Shader": self._effect_toon,
            "Color Boost": self._effect_color_boost,
            "Sepia": self._effect_sepia,
            "Bloom": self._effect_bloom,
            "Vignette": self._effect_vignette,
            "Night Vision": self._effect_night_vision,
            "Film Grain": self._effect_film_grain,
            "Pencil Sketch": self._effect_sketch,
            "Cross Processing": self._effect_cross_processing,
            "Posterize": self._effect_posterize,
            "Neon Glow": self._effect_neon_glow,
            "Hologram": self._effect_hologram,
            "Fog": self._effect_fog  # Renamed from "Enhanced Fog"
        }
        
        # Default parameters
        self.params = {
            "toon_levels": 5,
            "toon_edge_strength": 1.0,
            "edge_threshold": 20,
            "color_saturation": 1.5,
            "bloom_strength": 0.3,
            "bloom_size": 15,
            "vignette_strength": 0.5,
            "fog_density": 5.0,
            "fog_color_r": 200,
            "fog_color_g": 220,
            "fog_color_b": 255,
            "fog_start": 0.1,
            "fog_ray_intensity": 0.5,
            "fog_opacity": 0.8,
            "film_grain_amount": 0.2,
            "sketch_strength": 1.0,
            "posterize_levels": 4,
            "neon_glow_intensity": 0.7,
            "neon_glow_radius": 10,
            "hologram_lines": 50,
            "hologram_intensity": 0.8,
        }
        
        # Current effect
        self.current_effect = "Original"
        
    def _effect_original(self, image, depth=None):
        """Return the original image."""
        return image
        
    def _effect_toon(self, image, depth=None):
        """Apply a cartoon/toon shader effect."""
        # Get parameters
        levels = self.params.get("toon_levels", 5)  # Default to 5 if not set
        edge_strength = self.params.get("toon_edge_strength", 1.0)  # Default to 1.0 if not set
        
        # Quantize colors
        img_float = image.astype(np.float32)
        img_quantized = np.floor(img_float / 255.0 * levels) / levels * 255.0
        
        # Edge detection using depth if available
        if depth is not None:
            # Ensure depth is properly normalized
            depth_norm = depth.copy()
            if depth_norm.max() > 1.0:
                depth_norm = depth_norm / depth_norm.max()
            
            # Apply bilateral filter to depth to preserve edges but smooth noise
            depth_filtered = cv2.bilateralFilter(depth_norm.astype(np.float32), 9, 75, 75)
            
            # Calculate gradient of depth using Sobel operator
            depth_grad_x = cv2.Sobel(depth_filtered, cv2.CV_32F, 1, 0, ksize=3)
            depth_grad_y = cv2.Sobel(depth_filtered, cv2.CV_32F, 0, 1, ksize=3)
            depth_grad = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
            
            # Normalize gradient
            if depth_grad.max() > 0:
                depth_grad = depth_grad / depth_grad.max()
            
            # Create binary edge mask using adaptive threshold
            edges = np.where(depth_grad > 0.05, 1.0, 0.0)
            
            # Dilate edges slightly for better visibility
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1).astype(np.float32)
            
            # Apply edges to quantized image
            edges_rgb = np.stack([edges] * 3, axis=2)
            result = img_quantized * (1 - edge_strength * edges_rgb)
        else:
            # Fallback to color-based edge detection if depth isn't available
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Laplacian(gray, cv2.CV_32F)
            edges = np.abs(edges)
            # Normalize and threshold
            if edges.max() > 0:
                edges = edges / edges.max()
            edges = np.where(edges > 0.1, 1.0, 0.0)
            
            # Apply edges
            edges_rgb = np.stack([edges] * 3, axis=2)
            result = img_quantized * (1 - edge_strength * edges_rgb)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _effect_color_boost(self, image, depth=None):
        """Boost color saturation."""
        # Convert to HSV for easier saturation adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Boost saturation
        saturation = self.params["color_saturation"]
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)
            
        # Convert back to RGB
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def _effect_sepia(self, image, depth=None):
        """Apply sepia tone effect."""
        sepia_kernel = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        # Apply sepia transformation
        sepia = np.copy(image).astype(np.float32)
        sepia = cv2.transform(sepia, sepia_kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        
        return sepia
    
    def _effect_bloom(self, image, depth=None):
        """Apply bloom/glow effect."""
        strength = self.params["bloom_strength"]
        size = int(self.params["bloom_size"])
        if size % 2 == 0:
            size += 1  # Ensure size is odd
                    
        # Create a blurred version of the image
        blur = cv2.GaussianBlur(image, (size, size), 0)
        
        # Add the blur to the original image
        bloom = cv2.addWeighted(image, 1.0, blur, strength, 0)
        
        return bloom
    
    def _effect_vignette(self, image, depth=None):
        """Apply vignette effect (darkened corners)."""
        height, width = image.shape[:2]
            
        # Create a radial gradient
        y, x = np.ogrid[0:height, 0:width]
        center_y, center_x = height // 2, width // 2
        
        # Calculate distance from center
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize distance to [0, 1]
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = dist / max_dist
        
        # Create vignette mask
        strength = self.params["vignette_strength"]
        vignette = 1 - dist * strength
        vignette = np.clip(vignette, 0, 1)
        
        # Apply vignette
        result = image.astype(np.float32)
        for i in range(3):
            result[:,:,i] = result[:,:,i] * vignette
            
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _effect_night_vision(self, image, depth=None):
        """Create a night vision goggles effect."""
        # Convert to grayscale and enhance
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Add green tint
        result = np.zeros_like(image)
        result[:,:,1] = gray  # Green channel
        
        # Add noise to simulate night vision noise
        noise = np.random.normal(0, 15, gray.shape).astype(np.float32)
        result[:,:,1] = np.clip(result[:,:,1] + noise, 0, 255).astype(np.uint8)
        
        # Add vignette effect typical of night vision
        height, width = image.shape[:2]
        y, x = np.ogrid[0:height, 0:width]
        center_y, center_x = height//2, width//2
        mask = ((x - center_x)**2 + (y - center_y)**2) / (width/2)**2
        mask = np.clip(2 - mask * 1.5, 0, 1)
        mask = np.stack([mask] * 3, axis=2)
        
        result = (result.astype(np.float32) * mask).astype(np.uint8)
        
        return result
    
    def _effect_film_grain(self, image, depth=None):
        """Add film grain effect."""
        amount = self.params["film_grain_amount"]
        
        # Create noise
        grain = np.random.normal(0, 50, image.shape).astype(np.float32)
        
        # Apply the grain to the image
        result = image.astype(np.float32) + grain * amount
        
        return np.clip(result, 0, 255).astype(np.uint8)
        
    def _effect_sketch(self, image, depth=None):
        """Create a pencil sketch effect."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Invert the grayscale image
        inv_gray = 255 - gray
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        
        # Invert the blurred image
        inv_blur = 255 - blur
        
        # Create sketch by dividing grayscale by inverted blur
        sketch = cv2.divide(gray, inv_blur, scale=256.0)
        
        # Create a colored sketch by blending with original
        strength = self.params.get("sketch_strength", 1.0)
        
        # Create a mask from depth to reduce background noise
        mask = np.ones_like(gray, dtype=np.float32)
        if depth is not None:
            depth_norm = depth.copy()
            if depth_norm.max() > 1.0:
                depth_norm = depth_norm / depth_norm.max()
                
            # Create mask that fades out background
            depth_threshold = np.percentile(depth_norm, 70)
            mask = 1.0 - np.clip((depth_norm - depth_threshold) * 5, 0, 1)
        
        # Apply sketch effect with mask
        # Create result as float32 to avoid overflow issues
        result = np.zeros_like(image, dtype=np.float32)
        mask3 = np.stack([mask] * 3, axis=2)
        
        for i in range(3):
            # Blend original with sketch effect, apply mask to reduce background noise
            result[:,:,i] = ((1-strength) * image[:,:,i] + strength * sketch) * mask
            # Add back the original in background areas
            result[:,:,i] += image[:,:,i] * (1 - mask)
                
        # Convert back to uint8 only at the end
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _effect_cross_processing(self, image, depth=None):
        """Create a cross-processing effect popular in film photography."""
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Adjust individual channels (typical cross-processing look)
        # Enhance blues in shadows, yellows in highlights
        img_float[:,:,0] = np.clip(img_float[:,:,0] * 1.1, 0, 1)  # R
        img_float[:,:,1] = np.clip(img_float[:,:,1] * 1.3, 0, 1)  # G
        img_float[:,:,2] = np.clip(img_float[:,:,2] * 0.8, 0, 1)  # B
        
        # Add contrast
        img_float = (img_float - 0.5) * 1.4 + 0.5
        
        # Convert back to uint8
        result = np.clip(img_float * 255, 0, 255).astype(np.uint8)
        
        # Add slight vignette for film look
        height, width = image.shape[:2]
        y, x = np.ogrid[0:height, 0:width]
        center_y, center_x = height//2, width//2
        mask = ((x - center_x)**2 + (y - center_y)**2) / (width/2)**2
        mask = np.clip(1.2 - mask * 0.4, 0, 1)
        mask = np.stack([mask] * 3, axis=2)
        
        result = (result.astype(np.float32) * mask).astype(np.uint8)
        
        return result
    
    def _effect_posterize(self, image, depth=None):
        """Create a posterized effect with reduced color palette."""
        levels = self.params["posterize_levels"]
        
        # Posterize each channel
        img_float = image.astype(np.float32)
        img_posterized = np.floor(img_float / 255.0 * levels) / levels * 255.0
        
        # Add slight edge emphasis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.abs(edges)
        edges = np.where(edges > 20, 255, 0).astype(np.uint8)
        edges_rgb = np.stack([edges] * 3, axis=2)
        
        # Combine posterized image with edges
        result = np.where(edges_rgb > 0, edges_rgb * 0.3 + img_posterized * 0.7, img_posterized)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _effect_neon_glow(self, image, depth=None):
        """Create a neon-style glow effect."""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges using either depth or color information
        if depth is not None:
            # Make sure depth is properly formatted for Sobel
            depth_norm = depth.copy()
            if depth_norm.max() > 1.0:
                depth_norm = depth_norm / depth_norm.max()
                    
            # Convert to single-channel array if needed
            if len(depth_norm.shape) > 2:
                depth_norm = depth_norm[:, :, 0]
                
            # Ensure depth_norm is float32 and properly shaped
            depth_norm = depth_norm.astype(np.float32)
            
            # Use Canny edge detector instead of Sobel for more robustness
            depth_edges = cv2.Canny((depth_norm * 255).astype(np.uint8), 50, 150)
            edges = depth_edges
        else:
            # Use color-based edge detection
            edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to make them more visible
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Create colored edges with saturation based on original image
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Create vibrant colored edges
        edge_hue = (h + 120) % 180  # Shift hue for contrast
        edge_sat = np.full_like(s, 255)  # Full saturation
        
        # Combine into HSV and convert back to RGB
        edge_hsv = cv2.merge([edge_hue, edge_sat, np.minimum(edges, 255)])
        edge_rgb = cv2.cvtColor(edge_hsv, cv2.COLOR_HSV2RGB)
        
        # Apply glow using gaussian blur
        intensity = self.params.get("neon_glow_intensity", 0.7)
        radius = self.params.get("neon_glow_radius", 10)
        
        glow = cv2.GaussianBlur(edge_rgb, (radius*2+1, radius*2+1), 0)
        
        # Blend with original image
        result = np.clip(image.astype(np.float32) * 0.7 + glow.astype(np.float32) * intensity, 0, 255)
        
        return result.astype(np.uint8)
    
    def _effect_hologram(self, image, depth=None):
        """Create a hologram-like effect with scanlines and color shift."""
        # Convert to float32
        img_float = image.astype(np.float32) / 255.0
        
        # Create a cyan/blue tint for hologram
        cyan_img = np.zeros_like(img_float)
        cyan_img[:, :, 0] = img_float[:, :, 0] * 0.8  # Boost blue
        cyan_img[:, :, 1] = img_float[:, :, 1] * 1.0  # Keep green
        cyan_img[:, :, 2] = img_float[:, :, 2] * 0.2  # Reduce red
            
        # Create scanlines
        height, width = image.shape[:2]
        num_lines = self.params.get("hologram_lines", 50)
        line_height = height / num_lines
        
        scanlines = np.ones_like(img_float)
        for i in range(num_lines):
            y_start = int(i * line_height)
            y_end = int(min((i + 0.7) * line_height, height))
            scanlines[y_start:y_end, :, :] *= 0.85  # Darken scanlines
        
        # Apply scanlines to cyan image
        hologram_base = cyan_img * scanlines
        
        # Add flickering/noise effect
        noise = np.random.normal(0, 0.03, img_float.shape).astype(np.float32)
        
        # Add edge glow based on depth if available
        edge_glow = np.zeros_like(img_float)
        
        if depth is not None:
            # Process depth map carefully to avoid errors
            try:
                depth_norm = depth.copy()
                if depth_norm.max() > 1.0:
                    depth_norm = depth_norm / depth_norm.max()
                    
                # Make sure depth has the right shape and type
                if len(depth_norm.shape) > 2:
                    depth_norm = depth_norm[:, :, 0]
                    
                # Ensure it's 32-bit float for Sobel
                depth_norm = depth_norm.astype(np.float32)
                        
                # Use Sobel edge detection instead of Laplacian (more robust)
                sobel_x = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
                edges = np.sqrt(sobel_x**2 + sobel_y**2)  # Gradient magnitude
                
                if edges.max() > 0:
                    edges = edges / edges.max()
                                
                # Create glowing edges
                glow_strength = self.params.get("hologram_intensity", 0.8)
                edge_glow = np.stack([
                    edges * 0.1,  # Blue glow
                    edges * 0.6,  # Green glow - stronger
                    edges * 0.3   # Red glow
                ], axis=2)
                
            except Exception as e:
                print(f"Warning: Error processing depth map for hologram effect: {e}")
                # Continue without edge glow if there's an error
                pass
        
        # Add edge glow and noise to hologram
        hologram = hologram_base + edge_glow + noise
                
        # Add vertical distortion lines (like CRT interference)
        for _ in range(3):
            x_pos = np.random.randint(0, width)
            x_width = np.random.randint(2, 6)
            x_end = min(x_pos + x_width, width)
            hologram[:, x_pos:x_end, :] *= 1.5  # Brighten random vertical lines
        
        return np.clip(hologram * 255, 0, 255).astype(np.uint8)
    
    def _effect_fog(self, image, depth=None):
        """Apply an extremely thick fog effect that makes objects barely visible."""
        # Get parameters - dramatically increased density for extremely thick fog
        density = self.params.get("fog_density", 25.0)  # Increased from 15.0 to 25.0
        fog_color = np.array([
            255,  # Pure white fog
            255, 
            255
        ], dtype=np.float32)
        fog_start = self.params.get("fog_start", 0.0)  # Start fog immediately at camera
        
        height, width = image.shape[:2]
        
        # We must have depth information
        if depth is None:
            # If no depth is provided, create a uniform extreme fog
            print("Warning: No depth information for fog effect")
            # Create a barely visible image through extremely heavy fog
            result = image.astype(np.float32) * 0.05 + fog_color * 0.95  # Only 5% visibility
            return np.clip(result, 0, 255).astype(np.uint8)
        
        # Normalize depth
        depth_norm = depth.copy()
        if len(depth_norm.shape) > 2:
            depth_norm = depth_norm[:, :, 0]
        if depth_norm.max() > 1.0:
            depth_norm = depth_norm / depth_norm.max()
        
        # Calculate adjusted depth with fog start threshold
        adjusted_depth = np.maximum(depth_norm - fog_start, 0.0) / (1.0 - fog_start)
        adjusted_depth = np.clip(adjusted_depth, 0.0, 1.0)
        
        adjusted_depth = adjusted_depth ** 3.0 # 1.0 for chair, 3.0 for hotdog
        
        # Further reduce visibility to make objects barely visible
        adjusted_depth = adjusted_depth * 0.3  # Only 30% of normal visibility
        
        # Create 3D fog (stack for RGB channels)
        fog_factor_3d = np.stack([adjusted_depth] * 3, axis=2)
        
        # Create the final image: blend scene with fog
        result = image.astype(np.float32) * fog_factor_3d + fog_color * (1.0 - fog_factor_3d)
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_effect(self, image, depth=None):
        """Apply the current effect to an image."""
        if self.current_effect in self.effects:
            return self.effects[self.current_effect](image, depth)
        return image
    
    def show_editor_ui(self, image, depth=None):
        """Show interactive UI for adjusting effect parameters."""
        # Create window
        root = tk.Tk()
        root.title("NeRF Post-Processing")
        
        # Frame for controls
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Effect selector
        ttk.Label(control_frame, text="Select Effect:").pack(anchor=tk.W)
        effect_var = tk.StringVar()
        effect_dropdown = ttk.Combobox(
            control_frame,
            textvariable=effect_var,
            values=list(self.effects.keys())
        )
        effect_dropdown.pack(fill=tk.X, pady=5)
        effect_dropdown.current(0)
        
        # Parameters frame
        param_frame = ttk.LabelFrame(control_frame, text="Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=10)
        
        # Sliders for parameters
        sliders = {}
        
        # Image display
        img_display = ttk.Label(root)
        img_display.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Current processed image
        processed_img = image.copy()
        
        # Update image function
        def update_image():
            nonlocal processed_img
            self.current_effect = effect_var.get()
            processed_img = self.apply_effect(image.copy(), depth)
            
            # Resize for display
            display_img = Image.fromarray(processed_img)
            display_img = display_img.resize((min(800, display_img.width), 
                                              min(600, display_img.height)))
            tk_img = ImageTk.PhotoImage(display_img)
            img_display.config(image=tk_img)
            img_display.image = tk_img
        
        # Parameter update function
        def update_param(param_name, value):
            try:
                self.params[param_name] = float(value)
                update_image()
            except ValueError:
                pass
        
        # Create parameter sliders based on effect
        def update_parameters():
            # Clear existing sliders
            for widget in param_frame.winfo_children():
                widget.destroy()
            sliders.clear()
            
            effect = effect_var.get()
            
            # Add appropriate sliders for the current effect
            if effect == "Toon Shader":
                add_slider("toon_levels", "Color Levels", 2, 10, 1)
                add_slider("toon_edge_strength", "Edge Strength", 0, 2, 0.1)
                
            elif effect == "Color Boost":
                add_slider("color_saturation", "Saturation", 0.5, 3.0, 0.1)
                
            elif effect == "Bloom":
                add_slider("bloom_strength", "Strength", 0, 1, 0.05)
                add_slider("bloom_size", "Size", 3, 51, 2)
                
            elif effect == "Vignette":
                add_slider("vignette_strength", "Strength", 0, 1, 0.05)
            
            elif effect == "Film Grain":
                add_slider("film_grain_amount", "Amount", 0, 1, 0.05)
                
            elif effect == "Pencil Sketch":
                add_slider("sketch_strength", "Strength", 0, 1, 0.05)
                
            elif effect == "Posterize":
                add_slider("posterize_levels", "Levels", 2, 16, 1)
            
            elif effect == "Neon Glow":
                add_slider("neon_glow_intensity", "Intensity", 0, 1, 0.05)
                add_slider("neon_glow_radius", "Radius", 1, 20, 1)
                
            elif effect == "Hologram":
                add_slider("hologram_lines", "Lines", 10, 100, 5)
                add_slider("hologram_intensity", "Intensity", 0, 1, 0.05)
                
            elif effect == "Fog":
                add_slider("fog_density", "Density", 0.5, 5.0, 0.1)
                add_slider("fog_start", "Start", 0.0, 1.0, 0.05)
                add_slider("fog_ray_intensity", "Ray Intensity", 0.0, 1.0, 0.05)
            
            update_image()
        
        def add_slider(param_name, display_name, min_val, max_val, step):
            frame = ttk.Frame(param_frame)
            frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(frame, text=display_name, width=15).pack(side=tk.LEFT)
            value_var = tk.DoubleVar(value=self.params[param_name])
            slider = ttk.Scale(
                frame,
                from_=min_val,
                to=max_val,
                variable=value_var,
                command=lambda _: update_param(param_name, value_var.get())
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            value_label = ttk.Label(frame, width=5)
            value_label.pack(side=tk.RIGHT)
            
            def update_label(*_):
                value_label.config(text=f"{value_var.get():.1f}")
            
            slider.config(command=lambda _: (update_label(), update_param(param_name, value_var.get())))
            update_label()
            
            sliders[param_name] = slider
        
        # Effect change callback
        def on_effect_change(*_):
            update_parameters()
            
        effect_dropdown.bind("<<ComboboxSelected>>", on_effect_change)
        
        # Initial display
        update_parameters()
        update_image()
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
            
        ttk.Button(
            button_frame, 
            text="Save", 
            command=lambda: cv2.imwrite("processed_output.png", cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Close", 
            command=root.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
        # Start UI
        root.mainloop()
        
        return processed_img
