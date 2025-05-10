#pragma once
#include <glad/glad.h>
#include <string>

class Texture {
public:
    unsigned int id;
    int width;
    int height;
    int channels;
    
    Texture();
    Texture(const std::string& path);
    ~Texture();
    
    void load(const std::string& path);
    void bind(unsigned int unit = 0) const;
    
    static void saveImage(const std::string& path, int width, int height, unsigned char* data, int channels = 3);
};
