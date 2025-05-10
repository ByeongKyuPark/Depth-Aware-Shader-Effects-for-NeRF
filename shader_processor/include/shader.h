#pragma once
#include <glad/glad.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>

class Shader {
public:
    unsigned int ID;
    std::unordered_map<std::string, int> uniformLocations;
    
    Shader(const std::string& shaderPath);
    ~Shader();
    
    void use();
    void setBool(const std::string& name, bool value);
    void setInt(const std::string& name, int value);
    void setFloat(const std::string& name, float value);
    
private:
    void checkCompileErrors(unsigned int shader, const std::string& type);
};
