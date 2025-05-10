import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

try:
    import moderngl
    HAS_MODERNGL = True
except ImportError:
    HAS_MODERNGL = False
    print("ModernGL not found. Install with: pip install moderngl")

class ShaderEditor:
    def __init__(self, shader_dir='shaders'):
        """Initialize the shader editor."""
        self.shader_dir = shader_dir
        self.current_shader = None
        self.shader_uniforms = {}
        
        # Check if ModernGL is available
        if not HAS_MODERNGL:
            print("Warning: ModernGL not found. Using fallback mode.")
            return
            
        # Create ModernGL context
        self.ctx = moderngl.create_standalone_context()
        
        # Setup rendering quad
        self.quad_buffer = self.ctx.buffer(
            data=np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype='f4')
        )
        self.quad_vao = self.ctx.simple_vertex_array(
            self.ctx.program(
                vertex_shader='''
                    #version 330
                    in vec2 in_position;
                    out vec2 v_texcoord;
                    void main() {
                        gl_Position = vec4(in_position, 0.0, 1.0);
                        v_texcoord = (in_position + 1.0) * 0.5;
                    }
                ''',
                fragment_shader='''
                    #version 330
                    uniform sampler2D u_texture;
                    in vec2 v_texcoord;
                    out vec4 fragColor;
                    void main() {
                        fragColor = texture(u_texture, v_texcoord);
                    }
                '''
            ),
            self.quad_buffer,
            'in_position'
        )
        
        # Load available shaders
        self.shaders = self._load_shaders()
    
    def _load_shaders(self):
        """Load all GLSL shaders from the shader directory."""
        shaders = {}
        if not os.path.exists(self.shader_dir):
            print(f"Shader directory {self.shader_dir} not found.")
            return shaders
            
        for filename in os.listdir(self.shader_dir):
            if filename.endswith('.glsl'):
                path = os.path.join(self.shader_dir, filename)
                name = os.path.splitext(filename)[0]
                
                with open(path, 'r') as f:
                    shader_code = f.read()
                
                # Extract uniforms from shader code
                uniforms = {}
                for line in shader_code.split('\n'):
                    if line.strip().startswith('uniform') and ';' in line:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            # Extract name and default value if present
                            uniform_name = parts[2].split(';')[0].split('=')[0].strip()
                            
                            # Check for default value
                            default_value = 1.0
                            if '=' in line:
                                try:
                                    default_value = float(line.split('=')[1].split(';')[0].strip())
                                except ValueError:
                                    pass
                                    
                            # Only add if it's not a sampler
                            if not parts[1].startswith('sampler'):
                                uniforms[uniform_name] = default_value
                
                shaders[name] = {
                    'code': shader_code,
                    'uniforms': uniforms
                }
                
        return shaders
    
    def set_shader(self, shader_name):
        """Set the current shader by name."""
        if not HAS_MODERNGL:
            return False
            
        if shader_name not in self.shaders:
            print(f"Shader {shader_name} not found.")
            return False
        
        shader = self.shaders[shader_name]
        
        try:
            self.current_shader = self.ctx.program(
                vertex_shader='''
                    #version 330
                    in vec2 in_position;
                    out vec2 v_texcoord;
                    void main() {
                        gl_Position = vec4(in_position, 0.0, 1.0);
                        v_texcoord = (in_position + 1.0) * 0.5;
                    }
                ''',
                fragment_shader=shader['code']
            )
            
            # Setup default uniform values
            self.shader_uniforms = shader['uniforms'].copy()
            return True
        except Exception as e:
            print(f"Error compiling shader {shader_name}: {e}")
            return False
    
    def apply_shader(self, image, depth=None):
        """Apply the current shader to an image."""
        if not HAS_MODERNGL or self.current_shader is None:
            return image
            
        # Get image dimensions
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        # Debug info
        print(f"Image shape: {image.shape}, channels: {channels}")
        
        try:
            # Convert to float and normalize
            image_float = image.astype(np.float32) / 255.0
            
            # ModernGL expects data to be tightly packed in a specific format
            texture = self.ctx.texture((width, height), 3)
            texture.write(np.ascontiguousarray(image_float))
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
            # Handle depth texture similarly
            depth_texture = None
            if depth is not None:
                depth_height, depth_width = depth.shape[:2]
                # Ensure depth map has the same dimensions as the image
                if depth_height != height or depth_width != width:
                    depth = cv2.resize(depth, (width, height))
                
                depth_float = depth.astype(np.float32)
                depth_texture = self.ctx.texture((width, height), 1)
                depth_texture.write(np.ascontiguousarray(depth_float))
                depth_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                
                # Set the depth texture as uniform 1
                if 'u_depth' in self.current_shader:
                    self.current_shader['u_depth'] = 1
                depth_texture.use(1)
            
            # Set texture as uniform 0
            self.current_shader['u_texture'] = 0
            texture.use(0)
            
            # Set other uniform values
            for name, value in self.shader_uniforms.items():
                try:
                    self.current_shader[name] = value
                except Exception:
                    pass
                    
            # Setup framebuffer for output with matching size
            fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.texture((width, height), 3)]
            )
            fbo.use()
            
            # Render
            self.ctx.clear()
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)
            
            # Read result with correct dimensions
            buffer = fbo.color_attachments[0].read()
            buffer_size = len(buffer) // 4  # Size in floats (4 bytes per float)
            print(f"Buffer size: {len(buffer)} bytes, {buffer_size} floats")
            
            # Calculate expected dimensions
            expected_size = width * height * 3
            print(f"Expected size: {expected_size} floats")
            
            # Handle potential format differences
            if buffer_size == width * height * 3:  # RGB format
                result = np.frombuffer(buffer, dtype=np.float32).reshape(height, width, 3)
            elif buffer_size == width * height * 4:  # RGBA format
                # Convert RGBA to RGB
                result = np.frombuffer(buffer, dtype=np.float32).reshape(height, width, 4)[:,:,:3]
            elif buffer_size == width * height:  # Single channel
                result = np.frombuffer(buffer, dtype=np.float32).reshape(height, width, 1)
                result = np.repeat(result, 3, axis=2)  # Convert to RGB
            else:
                # Fall back to a different approach - try to determine dimensions from buffer size
                channels_from_size = buffer_size // (width * height)
                if channels_from_size > 0:
                    print(f"Inferred {channels_from_size} channels from buffer size")
                    if channels_from_size == 4:  # RGBA format
                        result = np.frombuffer(buffer, dtype=np.float32).reshape(height, width, 4)[:,:,:3]
                    else:
                        # Try to use as many channels as we can
                        usable_channels = min(3, channels_from_size)
                        result = np.frombuffer(buffer, dtype=np.float32).reshape(height, width, channels_from_size)
                        result = result[:,:,:usable_channels]
                else:
                    # Last resort - resize the buffer to match expected dimensions
                    print("Warning: Buffer size mismatch. Falling back to image without shader.")
                    return image
            
            # Convert back to uint8
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            
            # Clean up
            fbo.release()
            texture.release()
            if depth_texture:
                depth_texture.release()
                
            return result
            
        except Exception as e:
            print(f"Error applying shader: {e}")
            import traceback
            traceback.print_exc()
            # Return the original image if shader application fails
            return image
    
    def show_editor_ui(self, image, depth=None):
        """Show interactive UI for adjusting shader parameters."""
        # If ModernGL isn't available, show a message and return the original image
        if not HAS_MODERNGL:
            print("ModernGL is required for shader effects. Install with: pip install moderngl")
            return image
        
        # If no shaders are available, show a message and return the original image    
        if not self.shaders:
            print("No shaders found in the shaders directory.")
            return image
            
        # Create a backup copy of the original image
        original_image = image.copy()
        
        # Create window
        root = tk.Tk()
        root.title("NeRF Shader Editor")
        
        # Frame for controls
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Shader selector
        ttk.Label(control_frame, text="Select Shader:").pack(anchor=tk.W)
        shader_var = tk.StringVar()
        shader_dropdown = ttk.Combobox(
            control_frame, 
            textvariable=shader_var, 
            values=list(self.shaders.keys())
        )
        shader_dropdown.pack(fill=tk.X, pady=5)
        if self.shaders:
            shader_dropdown.current(0)
        
        # Uniform sliders
        sliders = {}
        slider_frame = ttk.LabelFrame(control_frame, text="Parameters", padding="10")
        slider_frame.pack(fill=tk.X, pady=10)
        
        # Image display
        img_label = ttk.Label(root)
        img_label.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Current processed image
        processed_img = image.copy()
        
        # Display the original image initially
        display_img = Image.fromarray(processed_img)
        display_img = display_img.resize((min(800, display_img.width), 
                                          min(600, display_img.height)))
        tk_img = ImageTk.PhotoImage(display_img)
        img_label.config(image=tk_img)
        img_label.image = tk_img
        
        def update_image():
            nonlocal processed_img
            try:
                if HAS_MODERNGL and self.current_shader is not None:
                    processed_img = self.apply_shader(image, depth)
                    display_img = Image.fromarray(processed_img)
                    display_img = display_img.resize((min(800, display_img.width), 
                                                   min(600, display_img.height)))
                    tk_img = ImageTk.PhotoImage(display_img)
                    img_label.config(image=tk_img)
                    img_label.image = tk_img
            except Exception as e:
                print(f"Error updating image: {e}")
                # Show error in UI if possible
                ttk.Label(control_frame, text=f"Error: {str(e)}", foreground="red").pack()
        
        def update_uniforms():
            # Update uniform values from sliders
            for name, slider in sliders.items():
                self.shader_uniforms[name] = float(slider.get())
            update_image()
            
        def on_shader_change(event):
            # Clear existing sliders
            for widget in slider_frame.winfo_children():
                widget.destroy()
            sliders.clear()
            
            # Set new shader
            shader_name = shader_var.get()
            if self.set_shader(shader_name):
                # Create sliders for uniforms
                for name, default in self.shader_uniforms.items():
                    frame = ttk.Frame(slider_frame)
                    frame.pack(fill=tk.X, pady=2)
                    
                    ttk.Label(frame, text=name, width=15).pack(side=tk.LEFT)
                    
                    slider = ttk.Scale(
                        frame, 
                        from_=0.0, 
                        to=10.0, 
                        value=default,
                        command=lambda _: update_uniforms()
                    )
                    slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    
                    value_label = ttk.Label(frame, width=5)
                    value_label.pack(side=tk.LEFT)
                    
                    # Update value label when slider moves
                    def update_label(_, val=slider, lbl=value_label):
                        lbl.config(text=f"{val.get():.1f}")
                    
                    slider.config(command=lambda _: (update_label(_), update_uniforms()))
                    update_label(None)
                    
                    sliders[name] = slider
                
                update_image()
        
        shader_dropdown.bind("<<ComboboxSelected>>", on_shader_change)
        
        # Initial setup
        if self.shaders:
            on_shader_change(None)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            button_frame, 
            text="Save", 
            command=lambda: cv2.imwrite("shader_output.png", cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Close", 
            command=root.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
        # Start UI loop
        root.mainloop()
        
        return processed_img
