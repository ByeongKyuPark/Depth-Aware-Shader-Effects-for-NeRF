# NeRF-W: Neural Radiance Fields with Depth-Aware Shader Effects

This project implements a simplified version of NeRF-W (Neural Radiance Fields in the Wild) for CS445 Computational Photography, with a special focus on depth-aware shader effects. It allows you to train neural radiance fields on multi-view image collections and render novel viewpoints of 3D scenes with creative visual effects.

## Video Demonstrations

See my shader effects in action in these video demonstrations:

[![Chair Model thumbnail](https://img.youtube.com/vi/DCMf-_8uxYM/hqdefault.jpg)](https://youtu.be/DCMf-_8uxYM)

[![Hotdog Model thumbnail](https://img.youtube.com/vi/WZiyH85AaJc/hqdefault.jpg)](https://youtu.be/WZiyH85AaJc)

[![Fog Effect thumbnail](https://img.youtube.com/vi/QJwHEvS-z2s/hqdefault.jpg)](https://youtu.be/QJwHEvS-z2s)


  
## Project Significance & Innovation

This project represents a forward-thinking approach to computational photography by combining neural rendering with traditional graphics techniques:

### Technical Innovation

- **Direct Neural Rendering**: We intentionally bypass mesh extraction to preserve the full fidelity and view-dependent effects (specular highlights, reflections, translucency) that are often lost in traditional 3D reconstruction
- **End-to-End Neural Graphics Pipeline**: The project demonstrates how classical graphics concepts can be reimagined with neural networks
- **Depth-Aware Neural Effects**: The implementation uniquely leverages the precise per-pixel depth maps that NeRF generates, enabling effects like edge detection that directly follow object boundaries rather than color changes, creating more accurate and compelling visual results

### Research Relevance

- **State-of-the-Art Approach**: NeRF is currently one of the most active research areas in computer vision and graphics (500+ papers since 2020)
- **Industry Alignment**: Major companies like NVIDIA, Meta, and Google are heavily investing in neural rendering technologies
- **Future-Facing Techniques**: This project explores where the graphics pipeline is heading, not where it's been

### Technical Depth

- **Cross-Domain Integration**: Successfully combines deep learning, volume rendering, computational photography, and shader programming
- **Novel View Synthesis**: Demonstrates scene understanding by generating photorealistic images from viewpoints never captured
- **Neural Representation Learning**: Shows how complex 3D environments can be encoded in neural network weights

### Educational Value

- **Interdisciplinary Learning**: Bridges the gap between traditional computer graphics and modern machine learning approaches
- **Implementation of Research**: Translates cutting-edge academic papers into working code
- **Visual Computing Pipeline**: Develops understanding of the full image synthesis and processing pipeline

## Overview

NeRF-W extends the original Neural Radiance Fields framework to handle real-world photo collections with varying lighting and exposure. This implementation includes:

- Core NeRF architecture with positional encoding
- Appearance embeddings for handling appearance variations
- Hierarchical sampling for improved rendering quality
- Training and evaluation pipelines for the NeRF synthetic dataset

## Project Structure

```
Depth-Aware Shader Effects for NeRF/
├── config.py                  # Core configuration parameters
├── run.py                     # Main entry point for training/rendering
├── render_aligned_spiral.py   # High-quality spiral rendering
├── create_video.py            # Video creation utility
├── check_cuda.py              # CUDA diagnostics
├── checkpoints_chair/         # Chair model checkpoints
├── checkpoints_hotdog/        # Hotdog model checkpoints
└── src/                       # Core modules directory
    ├── models.py              # Neural network architecture
    ├── dataset.py             # Dataset handling
    ├── ray_utils.py           # Ray generation utilities
    ├── render.py              # Volume rendering implementation
    ├── train.py               # Training pipeline
    ├── post_processor.py      # Image effects
    └── shader_system.py       # Shader system
```

## Setup Instructions

### 1. Requirements

Install the required dependencies:

```bash
pip install torch torchvision numpy pillow matplotlib tqdm moderngl
```

For GPU acceleration (recommended):

```bash
# For CUDA support (replace cu117 with your CUDA version)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
```

### 2. Dataset Setup

1. Download the NeRF synthetic dataset from the original sources:

The renders are from modified Blender models located on blendswap.com:

- Chair by 1DInc (CC-0): https://www.blendswap.com/blend/8261
- Hotdog by erickfree (CC-0): https://www.blendswap.com/blend/23962

2. Extract the dataset to a directory of your choice

3. Update the dataset path in `config.py`:

```python
dataset_path = 'data/nerf_synthetic'  # Change this to your actual path
```

The dataset should have the following structure:

```
nerf_synthetic/
├── chair/
│   ├── train/
│   │   ├── r_0.png
│   │   └── ...
│   ├── test/
│   ├── val/
│   ├── transforms_train.json  # JSON files are directly under each scene folder
│   ├── transforms_test.json
│   └── transforms_val.json
├── hotdog/
└── ...
```

## Usage

### Training

To train a NeRF model on the default scene:

```bash
python run.py --mode train
```

To train on a different scene:

```bash
python run.py --mode train --scene hotdog
```

**Important**: You must train a model for each scene before rendering it. The training process will:

1. Load the specified scene's training images and camera parameters
2. Train a neural network to represent the scene
3. Save checkpoints to `checkpoints_[scene_name]/checkpoint_[iteration].pt` and a final checkpoint to `checkpoints_[scene_name]/checkpoint_final.pt`
4. Display training progress and metrics

Training typically takes several hours depending on your hardware. To check training progress, look for checkpoint files in the appropriate checkpoint directory.

### Checking Available Checkpoints

To verify which models you've already trained:

```bash
ls checkpoints_chair/
ls checkpoints_hotdog/
```

If you see `checkpoint_final.pt` in a directory, you have a fully trained model for that scene.

## Rendering Commands

### Recommended Rendering Approach

Our recommended rendering approach uses the aligned spiral path for high-quality results:

```bash
# For chair model
python render_aligned_spiral.py --scene chair --checkpoint checkpoints_chair/checkpoint_final.pt --output_dir output/chair_spiral --frames 540 --fps 60 --loops 2 --rotation x

# For hotdog model
python render_aligned_spiral.py --scene hotdog --checkpoint checkpoints_hotdog/checkpoint_final.pt --output_dir output/hotdog_spiral --frames 540 --fps 60 --loops 2 --rotation x
```

The `--rotation` parameter tells the script which axis to rotate around:

- `--rotation x` - Rotate 90° around X-axis (default, good for chair)
- `--rotation z` - Rotate around Z-axis (sometimes better for hotdog)

You can adjust these parameters based on your needs:

- `--frames` - Number of frames to render (higher = smoother, but takes longer)
- `--loops` - How many complete rotations around the object
- `--fps` - Frames per second in the output video (60 gives smooth motion)

### Alternative Rendering Commands

For quick basic rendering:

```bash
python run.py --mode render --scene chair --checkpoint checkpoints_chair/checkpoint_final.pt
```

For interactive shader selection:

```bash
python run.py --mode render --scene chair --checkpoint checkpoints_chair/checkpoint_final.pt --use_shader
```

For custom camera paths:

```bash
# Spiral path
python run.py --mode render --scene chair --checkpoint checkpoints_chair/checkpoint_final.pt --camera_path spiral --frames 180

# Circle path with custom height
python run.py --mode render --scene hotdog --checkpoint checkpoints_hotdog/checkpoint_final.pt --camera_path circle --height 0.5
```

For faster preview rendering:

```bash
python run.py --mode render --scene chair --checkpoint checkpoints_chair/checkpoint_final.pt --width 400 --height 400 --quality preview
```

### Scene-Specific Considerations

- **Chair model**: Required X-axis rotation (90°) to appear upright
- **Hotdog model**: Required X-axis rotation (90°) to appear upright

### Creating Videos

After rendering frames to an output directory:

```bash
python create_video.py --input_dir output/chair_spiral --output output/chair_video.mp4 --pattern "frame_*.png" --fps 60
```

### GPU Acceleration

To check if your system supports GPU acceleration:

```bash
python check_cuda.py
```

If you have a compatible NVIDIA GPU, the rendering will use CUDA for significantly faster processing.

## Post-Processing Effects

Our implementation includes the following shader effects:

1. **Toon Shader**: Creates a cartoon/cel-shaded look with quantized colors and edge detection
2. **Color Boost**: Increases color saturation and vibrancy
3. **Sepia**: Applies a vintage sepia tone filter
4. **Bloom**: Adds a soft glow to bright areas of the image
5. **Vignette**: Darkens the corners/edges of the image
6. **Night Vision**: Creates a green-tinted night vision goggles effect
7. **Film Grain**: Adds noise to simulate photographic film grain
8. **Pencil Sketch**: Simulates a hand-drawn sketch effect
9. **Cross Processing**: Creates color shifts inspired by alternative film processing
10. **Posterize**: Reduces the number of colors for a poster-like effect
11. **Neon Glow**: Creates cyberpunk-style glowing edges that respond to scene geometry
12. **Hologram**: Transforms models into holographic projections with scanlines and glow
13. **Fog**: Creates realistic volumetric fog that accurately follows the 3D structure of the scene

### Depth-Aware Fog Demonstration

Our fog effect demonstrates a key advantage of NeRF - access to precise depth information without requiring explicit 3D modeling:

- **Watch the demonstration**: [Fog Effect Demonstration Video](https://youtu.be/QJwHEvS-z2s)

This effect is significant because:

1. **Direct Depth Access**: The fog uses depth maps generated directly during NeRF's volume rendering, without extracting meshes or point clouds
2. **Physically Accurate**: The fog follows the true 3D structure of the scene, creating realistic atmospheric effects where objects gradually fade with distance
3. **Impossible with Traditional Photography**: Standard photography would require specialized depth sensors to achieve similar results
4. **Simpler Than Traditional CG**: Traditional CG would require creating a complete 3D model first, while NeRF learns the geometry implicitly

To generate the fog effect:

```bash
# Generate fog video using only frames with depth maps
python apply_all_shaders.py --input_dir output/chair_spiral --output_dir output/effects --fog_only
```

The resulting effect shows how NeRF offers the best of both worlds: photorealistic imagery with the depth information traditionally only available in CG pipelines.

### Working with Depth-Aware Effects

To maximize the benefit of depth information in shader effects, our implementation processes only the frames that have depth maps for certain effects like fog:

```bash
# Generate fog video using frames with depth maps
python apply_all_shaders.py --input_dir output/chair_spiral --output_dir output/effects --fog_only
```

The dense fog effect demonstrates how depth information enables realistic atmospheric effects where objects gradually disappear into the fog based on their distance from the camera.

## NeRF vs. Traditional Methods

### Comparing NeRF, Traditional Photography, and Traditional CG

| Aspect                           | NeRF                                                                           | Traditional Photography                                                               | Traditional CG                                                             |
| -------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Depth Information**            | Inherently generates accurate per-pixel depth as part of the rendering process | Requires specialized hardware (depth sensors, stereo cameras, LiDAR) to capture depth | Has perfect depth information but requires manual modeling of all geometry |
| **Material and Lighting**        | Automatically learns complex material properties and lighting from photos      | Captures real-world materials and lighting but can't modify them later                | Requires explicit programming of materials and lighting models by artists  |
| **Post-Processing Capabilities** | Can apply depth-aware effects with accurate depth information                  | Limited to 2D effects unless depth is specially captured                              | Has full depth information but materials may not be photorealistic         |

### Effect Examples

#### Toon/Edge Rendering

- **NeRF/CG**: Can detect true object boundaries using depth
- **Photography**: Must rely on color edges which can be noisy

## Output Format Details

The renderer saves images in PNG format:

```
frame_0000.png, frame_0001.png, ... (Color images)
depth_0000.png, depth_0001.png, ... (Depth visualizations)
```

PNG is the standard format for rendered images because:

1. **Lossless Quality**: Unlike JPG, PNG doesn't introduce compression artifacts, preserving the exact pixel values from neural rendering
2. **Higher Bit Depth**: Supports 16-bit channels for more precise color representation
3. **Alpha Channel**: Provides transparency support (though not used in the current implementation)
4. **Better for Screenshots/UI**: Keeps text and sharp edges crisp without compression artifacts

## Configuration

Key parameters in `config.py`:

- `scene`: Which synthetic scene to use (options: chair, hotdog)
- `hidden_dim`: Dimensionality of the neural network layers
- `num_samples`: Number of samples per ray for the coarse network
- `num_importance`: Number of additional samples for the fine network
- `use_appearance`: Whether to use appearance embeddings
- `num_iterations`: Number of training iterations

## Implementation Details

The project is structured as follows:

- `config.py`: Configuration parameters
- `src/models.py`: Neural network architecture
- `src/ray_utils.py`: Ray generation and sampling utilities
- `src/render.py`: Volume rendering implementation
- `src/dataset.py`: Dataset handling and loading
- `src/train.py`: Training pipeline
- `run.py`: Main entry point for training and rendering

## Acknowledgements

This implementation is based on the following papers:

- "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (Mildenhall et al., 2020)
- "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections" (Martin-Brualla et al., 2021)
