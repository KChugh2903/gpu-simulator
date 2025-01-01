
// visualization/shader_loader.cpp
#include "shader_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

std::unordered_map<std::string, GLuint> ShaderLoader::shaderCache;

GLuint ShaderLoader::createShaderProgram(const std::string& vertexPath, 
                                       const std::string& fragmentPath) {
    // Check cache first
    std::string cacheKey = vertexPath + ":" + fragmentPath;
    auto it = shaderCache.find(cacheKey);
    if (it != shaderCache.end()) {
        return it->second;
    }

    // Load and compile vertex shader
    GLuint vertexShader = loadShader(vertexPath, GL_VERTEX_SHADER);
    if (vertexShader == 0) {
        throw std::runtime_error("Failed to compile vertex shader");
    }

    // Load and compile fragment shader
    GLuint fragmentShader = loadShader(fragmentPath, GL_FRAGMENT_SHADER);
    if (fragmentShader == 0) {
        glDeleteShader(vertexShader);
        throw std::runtime_error("Failed to compile fragment shader");
    }

    // Create shader program
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    // Check for linking errors
    checkCompileErrors(program, "PROGRAM");

    // Clean up shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Cache the program
    shaderCache[cacheKey] = program;

    return program;
}

GLuint ShaderLoader::createShaderProgram(const std::string& vertexPath, 
                                       const std::string& geometryPath,
                                       const std::string& fragmentPath) {
    // Check cache first
    std::string cacheKey = vertexPath + ":" + geometryPath + ":" + fragmentPath;
    auto it = shaderCache.find(cacheKey);
    if (it != shaderCache.end()) {
        return it->second;
    }

    // Load and compile shaders
    GLuint vertexShader = loadShader(vertexPath, GL_VERTEX_SHADER);
    GLuint geometryShader = loadShader(geometryPath, GL_GEOMETRY_SHADER);
    GLuint fragmentShader = loadShader(fragmentPath, GL_FRAGMENT_SHADER);

    if (vertexShader == 0 || geometryShader == 0 || fragmentShader == 0) {
        glDeleteShader(vertexShader);
        glDeleteShader(geometryShader);
        glDeleteShader(fragmentShader);
        throw std::runtime_error("Failed to compile shaders");
    }
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, geometryShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    checkCompileErrors(program, "PROGRAM");

    glDeleteShader(vertexShader);
    glDeleteShader(geometryShader);
    glDeleteShader(fragmentShader);

    // Cache the program
    shaderCache[cacheKey] = program;

    return program;
}

GLuint ShaderLoader::loadShader(const std::string& path, GLenum shaderType) {
    try {
        // Read shader source
        std::string shaderCode = readShaderFile(path);
        GLuint shader = glCreateShader(shaderType);
        const char* code = shaderCode.c_str();
        glShaderSource(shader, 1, &code, nullptr);
        glCompileShader(shader);
        std::string type;
        switch (shaderType) {
            case GL_VERTEX_SHADER: type = "VERTEX"; break;
            case GL_FRAGMENT_SHADER: type = "FRAGMENT"; break;
            case GL_GEOMETRY_SHADER: type = "GEOMETRY"; break;
            default: type = "UNKNOWN"; break;
        }
        checkCompileErrors(shader, type);

        return shader;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading shader " << path << ": " << e.what() << std::endl;
        return 0;
    }
}

std::string ShaderLoader::readShaderFile(const std::string& path) {
    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    
    try {
        file.open(path);
        std::stringstream stream;
        stream << file.rdbuf();
        file.close();
        return stream.str();
    }
    catch (const std::ifstream::failure& e) {
        throw std::runtime_error("Failed to read shader file: " + path);
    }
}

void ShaderLoader::checkCompileErrors(GLuint shader, const std::string& type) {
    GLint success;
    GLchar infoLog[1024];

    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
            throw std::runtime_error("Shader compilation error of type " + type + ": " + infoLog);
        }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, nullptr, infoLog);
            throw std::runtime_error("Shader program linking error of type " + type + ": " + infoLog);
        }
    }
}

void ShaderLoader::cleanup() {
    // Delete all shader programs from cache
    for (const auto& pair : shaderCache) {
        glDeleteProgram(pair.second);
    }
    shaderCache.clear();
}