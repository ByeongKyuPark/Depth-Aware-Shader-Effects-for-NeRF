import numpy as np
import cv2
import os
import importlib.util
import inspect

class PostProcessingSystem:
    """Manages post-processing effects similar to shaders but using NumPy/OpenCV."""
    
    def __init__(self, effects_dir='effects'):
        """Initialize the post-processing system."""
        self.effects_dir = effects_dir
        self.effects = {}
        self.load_effects()
        
    def load_effects(self):
        """Load all effect modules from the effects directory."""
        if not os.path.exists(self.effects_dir):
            os.makedirs(self.effects_dir)
            self._create_default_effects()
            
        for filename in os.listdir(self.effects_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                effect_name = os.path.splitext(filename)[0]
                file_path = os.path.join(self.effects_dir, filename)
                
                # Load the module
                spec = importlib.util.spec_from_file_location(effect_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check if the module has process function and parameters
                if hasattr(module, 'process') and hasattr(module, 'parameters'):
                    self.effects[effect_name] = {
                        'module': module,
                        'parameters': module.parameters
                    }
    
    def _create_default_effects(self):
        """Create some default effect files."""
        # Implementation to create default effect files
        pass
    
    def get_available_effects(self):
        """Get list of available effects."""
        return list(self.effects.keys())
    
    def get_effect_parameters(self, effect_name):
        """Get parameters for a specific effect."""
        if effect_name in self.effects:
            return self.effects[effect_name]['parameters']
        return {}
    
    def apply_effect(self, image, depth=None, effect_name='default', parameters=None):
        """
        Apply the specified effect to the image.
        
        Args:
            image: RGB image as numpy array
            depth: Optional depth map as numpy array
            effect_name: Name of the effect to apply
            parameters: Optional parameter overrides
            
        Returns:
            Processed image
        """
        if effect_name not in self.effects:
            return image
            
        effect = self.effects[effect_name]['module']
        
        # Get default parameters and update with any overrides
        params = dict(self.effects[effect_name]['parameters'])
        if parameters:
            params.update(parameters)
            
        # Apply the effect
        try:
            return effect.process(image, depth, **params)
        except Exception as e:
            print(f"Error applying effect {effect_name}: {e}")
            return image
