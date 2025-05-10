#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D u_texture;  // Color texture
uniform sampler2D u_depth;    // Depth texture
uniform float u_levels = 4.0; // Number of color quantization levels
uniform float u_edge_threshold = 0.1; // Threshold for edge detection

void main()
{
    // Sample color texture
    vec4 color = texture(u_texture, TexCoord);
    
    // Quantize colors for toon effect
    color.rgb = floor(color.rgb * u_levels) / u_levels;
    
    // Edge detection using depth
    float depth = texture(u_depth, TexCoord).r;
    float depth_right = texture(u_depth, TexCoord + vec2(0.001, 0.0)).r;
    float depth_bottom = texture(u_depth, TexCoord + vec2(0.0, 0.001)).r;
    
    // Calculate edge based on depth discontinuities
    float edge = step(u_edge_threshold, abs(depth - depth_right) + abs(depth - depth_bottom));
    
    // Apply edge as black outline
    color.rgb *= (1.0 - edge);
    
    FragColor = color;
}
