
// visualization/renderer.hpp
#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include "rocket.hpp"
#include "dynamics.hpp"

class Renderer {
public:
    Renderer(int width = 1920, int height = 1080);
    ~Renderer();

    void init();
    void render(const std::vector<Dynamics::State>& trajectory,
                const Rocket& rocket);
    void cleanup();

    bool shouldClose() const { return glfwWindowShouldClose(window); }
    void processInput();

private:
    GLFWwindow* window;
    int windowWidth, windowHeight;

    // Shader programs
    GLuint rocketShader;
    GLuint trajectoryShader;

    // Geometry
    GLuint rocketVAO, rocketVBO, rocketEBO;
    GLuint trajectoryVAO, trajectoryVBO;

    // Camera
    glm::vec3 cameraPos;
    glm::vec3 cameraFront;
    glm::vec3 cameraUp;
    float yaw, pitch;
    float lastX, lastY;
    bool firstMouse;

    // Methods
    void initShaders();
    void initGeometry();
    void initCamera();
    void updateCamera();
    void drawRocket(const Dynamics::State& state);
    void drawTrajectory(const std::vector<Dynamics::State>& trajectory);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
};
