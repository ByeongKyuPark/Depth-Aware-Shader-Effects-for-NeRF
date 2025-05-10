#include "shader.h"

Shader::Shader(const std::string& shaderPath) {
    std::string vertexCode, fragmentCode;
    std::ifstream vShaderFile, fShaderFile;
    
    // Default vertex shader for full-screen quad
    vertexCode = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;
        
        out vec2 TexCoord;
        
        void main() {
            gl_Position = vec4(aPos, 1.0);
            TexCoord = aTexCoord;
        }
    )";
    
    // Load fragment shader from file
    try {
        fShaderFile.open(shaderPath);
        std::stringstream fShaderStream;
        fShaderStream << fShaderFile.rdbuf();
        fShaderFile.close();
        fragmentCode = fShaderStream.str();
    }
    catch(std::ifstream::failure& e) {
        std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << shaderPath << std::endl;
        throw;
    }
    
    // Compile shaders
    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();
    
    unsigned int vertex, fragment;
    
    // Vertex shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");
    
    // Fragment shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT");
    
    // Shader program
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");
    
    // Delete shaders as they're linked into the program now
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

Shader::~Shader() {
    glDeleteProgram(ID);
}

void Shader::use() {
    glUseProgram(ID);
}

void Shader::setBool(const std::string& name, bool value) {
    int location;
    if (uniformLocations.find(name) != uniformLocations.end()) {
        location = uniformLocations[name];
    } else {
        location = glGetUniformLocation(ID, name.c_str());
        uniformLocations[name] = location;
    }
    glUniform1i(location, (int)value);
}

void Shader::setInt(const std::string& name, int value) {
    int location;
    if (uniformLocations.find(name) != uniformLocations.end()) {
        location = uniformLocations[name];
    } else {
        location = glGetUniformLocation(ID, name.c_str());
        uniformLocations[name] = location;
    }
    glUniform1i(location, value);
}

void Shader::setFloat(const std::string& name, float value) {
    int location;
    if (uniformLocations.find(name) != uniformLocations.end()) {
        location = uniformLocations[name];
    } else {
        location = glGetUniformLocation(ID, name.c_str());
        uniformLocations[name] = location;
    }
    glUniform1f(location, value);
}

void Shader::checkCompileErrors(unsigned int shader, const std::string& type) {
    int success;
    char infoLog[1024];
    
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << std::endl;
            throw std::runtime_error("Shader compilation failed");
        }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << std::endl;
            throw std::runtime_error("Shader program linking failed");
        }
    }
}
