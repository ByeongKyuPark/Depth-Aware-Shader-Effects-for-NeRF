#include "texture.h"

// Include stb_image for loading images
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

// Include stb_image_write for saving images
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <iostream>

Texture::Texture() : id(0), width(0), height(0), channels(0) {}

Texture::Texture(const std::string& path) : id(0), width(0), height(0), channels(0) {
    load(path);
}

Texture::~Texture() {
    if (id != 0) {
        glDeleteTextures(1, &id);
    }
}

void Texture::load(const std::string& path) {
    // Generate texture
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Load image
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    
    if (data) {
        // Determine format based on number of channels
        GLenum format;
        if (channels == 1)
            format = GL_RED;
        else if (channels == 3)
            format = GL_RGB;
        else if (channels == 4)
            format = GL_RGBA;
        else {
            std::cerr << "Unsupported number of channels: " << channels << std::endl;
            stbi_image_free(data);
            return;
        }
        
        // Upload texture data
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        
        // Free image data
        stbi_image_free(data);
    } else {
        std::cerr << "Failed to load texture: " << path << std::endl;
    }
}

void Texture::bind(unsigned int unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, id);
}

void Texture::saveImage(const std::string& path, int width, int height, unsigned char* data, int channels) {
    // Determine file extension
    std::string extension = path.substr(path.find_last_of(".") + 1);
    
    // Save using appropriate function
    if (extension == "png") {
        stbi_write_png(path.c_str(), width, height, channels, data, width * channels);
    } else if (extension == "jpg" || extension == "jpeg") {
        stbi_write_jpg(path.c_str(), width, height, channels, data, 95);
    } else if (extension == "bmp") {
        stbi_write_bmp(path.c_str(), width, height, channels, data);
    } else {
        std::cerr << "Unsupported file extension: " << extension << std::endl;
    }
}
