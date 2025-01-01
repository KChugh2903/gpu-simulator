#pragma once
#include <GL/glew.h>
#include <string>
#include <unordered_map>
#include <memory>

class ShaderLoader {
public:
    static GLuint createShaderProgram(const std::string& vertexPath, 
                                    const std::string& fragmentPath);
    
    // Optional: Create program with geometry shader
    static GLuint createShaderProgram(const std::string& vertexPath, 
                                    const std::string& geometryPath,
                                    const std::string& fragmentPath);
    
    // Cleanup shaders and programs
    static void cleanup();

private:
    static GLuint loadShader(const std::string& path, GLenum shaderType);
    static std::string readShaderFile(const std::string& path);
    static void checkCompileErrors(GLuint shader, const std::string& type);
    
    // Cache for shader programs
    static std::unordered_map<std::string, GLuint> shaderCache;
};