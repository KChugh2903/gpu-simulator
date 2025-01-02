#include "renderer.hpp"
#include "shader_loader.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

Renderer::Renderer(int width, int height) 
    : windowWidth(width), windowHeight(height), window(nullptr),
      cameraPos(glm::vec3(0.0f, 0.0f, 3.0f)),
      cameraFront(glm::vec3(0.0f, 0.0f, -1.0f)),
      cameraUp(glm::vec3(0.0f, 1.0f, 0.0f)),
      yaw(-90.0f), pitch(0.0f),
      lastX(width/2.0f), lastY(height/2.0f),
      firstMouse(true) {
}

Renderer::~Renderer() {
    cleanup();
}

void Renderer::init() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(windowWidth, windowHeight, "Rocket Simulation", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        throw std::runtime_error("Failed to initialize GLEW");
    }
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
    initShaders();
    initGeometry();
    initCamera();
    glEnable(GL_DEPTH_TEST);
}

void Renderer::initShaders() {
    try {
        std::string shaderPath = "shaders/";
        
        rocketShader = ShaderLoader::createShaderProgram(
            shaderPath + "rocket.vert",
            shaderPath + "rocket.frag"
        );

        glUseProgram(rocketShader);
        glUniform3f(glGetUniformLocation(rocketShader, "lightPos"), 100.0f, 100.0f, 100.0f);
        glUniform3f(glGetUniformLocation(rocketShader, "lightColor"), 1.0f, 1.0f, 1.0f);
        glUniform3f(glGetUniformLocation(rocketShader, "objectColor"), 0.7f, 0.7f, 0.7f);
        glUniform3f(glGetUniformLocation(rocketShader, "viewPos"), 0.0f, 0.0f, 0.0f);

        trajectoryShader = ShaderLoader::createShaderProgram(
            shaderPath + "trajectory.vert",
            shaderPath + "trajectory.frag"
        );

        if (rocketShader == 0 || trajectoryShader == 0) {
            throw std::runtime_error("Failed to create shader programs");
        }
    } catch (const std::exception& e) {
        std::cerr << "Shader initialization failed: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::initGeometry() {
    std::vector<float> vertices = {};
    std::vector<unsigned int> indices = {};

    glGenVertexArrays(1, &rocketVAO);
    glGenBuffers(1, &rocketVBO);
    glGenBuffers(1, &rocketEBO);

    glBindVertexArray(rocketVAO);
    glBindBuffer(GL_ARRAY_BUFFER, rocketVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rocketEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glGenVertexArrays(1, &trajectoryVAO);
    glGenBuffers(1, &trajectoryVBO);
}

void Renderer::render(const std::vector<Dynamics::State>& trajectory,
                     const Rocket& rocket) {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    updateCamera();
    drawTrajectory(trajectory);
    if (!trajectory.empty()) {
        drawRocket(trajectory.back());
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Renderer::initCamera() {
   cameraPos = glm::vec3(0.0f, -10.0f, 5.0f);  // Start behind and slightly above rocket
   cameraFront = glm::vec3(0.0f, 1.0f, -0.2f);  // Look at rocket's initial position
   cameraUp = glm::vec3(0.0f, 0.0f, 1.0f);      // Z-up coordinate system
   
   yaw = -90.0f;    // Look along y-axis initially
   pitch = 0.0f;    // Horizontal view
      lastX = windowWidth / 2.0f;
   lastY = windowHeight / 2.0f;
   firstMouse = true;
   glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
   glm::mat4 projection = glm::perspective(glm::radians(45.0f),
       static_cast<float>(windowWidth) / windowHeight,
       0.1f, 10000.0f);

   glUseProgram(rocketShader);
   glUniformMatrix4fv(glGetUniformLocation(rocketShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
   glUniformMatrix4fv(glGetUniformLocation(rocketShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

   glUseProgram(trajectoryShader);
   glUniformMatrix4fv(glGetUniformLocation(trajectoryShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
   glUniformMatrix4fv(glGetUniformLocation(trajectoryShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
}

void Renderer::drawRocket(const Dynamics::State& state) {
    glUseProgram(rocketShader);
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(state.position.x(),
                                          state.position.y(),
                                          state.position.z()));
    glm::mat4 rotMat(1.0f);
    model *= rotMat;
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f),
                                          (float)windowWidth / windowHeight,
                                          0.1f, 10000.0f);

    glUniformMatrix4fv(glGetUniformLocation(rocketShader, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(rocketShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(rocketShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glBindVertexArray(rocketVAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
}

void Renderer::drawTrajectory(const std::vector<Dynamics::State>& trajectory) {
    if (trajectory.empty()) return;

    std::vector<float> trajectoryData;
    trajectoryData.reserve(trajectory.size() * 6);  // pos + color per point

    for (const auto& state : trajectory) {
        trajectoryData.push_back(state.position.x());
        trajectoryData.push_back(state.position.y());
        trajectoryData.push_back(state.position.z());
        float normalizedAltitude = state.position.z() / 1000.0f;  // Normalize to km
        trajectoryData.push_back(1.0f - normalizedAltitude);  // Red
        trajectoryData.push_back(0.0f);                       // Green
        trajectoryData.push_back(normalizedAltitude);         // Blue
    }

    glBindVertexArray(trajectoryVAO);
    glBindBuffer(GL_ARRAY_BUFFER, trajectoryVBO);
    glBufferData(GL_ARRAY_BUFFER, trajectoryData.size() * sizeof(float),
                trajectoryData.data(), GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                         (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glUseProgram(trajectoryShader);

    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f),
                                          (float)windowWidth / windowHeight,
                                          0.1f, 10000.0f);

    glUniformMatrix4fv(glGetUniformLocation(trajectoryShader, "view"), 1,
                      GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(trajectoryShader, "projection"), 1,
                      GL_FALSE, glm::value_ptr(projection));

    glDrawArrays(GL_LINE_STRIP, 0, trajectory.size());
}

void Renderer::cleanup() {
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}

void Renderer::updateCamera() {
    const float cameraSpeed = 0.05f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraUp;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraUp;
}

void Renderer::mouseCallback(GLFWwindow* window, double xposIn, double yposIn) {
    // Get the renderer instance from GLFW window user pointer
    auto* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (renderer->firstMouse) {
        renderer->lastX = xpos;
        renderer->lastY = ypos;
        renderer->firstMouse = false;
    }

    float xoffset = xpos - renderer->lastX;
    float yoffset = renderer->lastY - ypos; // reversed since y-coordinates go from bottom to top
    renderer->lastX = xpos;
    renderer->lastY = ypos;

    const float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    renderer->yaw += xoffset;
    renderer->pitch += yoffset;
    if (renderer->pitch > 89.0f)
        renderer->pitch = 89.0f;
    if (renderer->pitch < -89.0f)
        renderer->pitch = -89.0f;
    glm::vec3 front;
    front.x = cos(glm::radians(renderer->yaw)) * cos(glm::radians(renderer->pitch));
    front.y = sin(glm::radians(renderer->pitch));
    front.z = sin(glm::radians(renderer->yaw)) * cos(glm::radians(renderer->pitch));
    renderer->cameraFront = glm::normalize(front);
}

void Renderer::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    auto* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    static float fov = 45.0f;
    fov -= static_cast<float>(yoffset);
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 45.0f)
        fov = 45.0f;
    glm::mat4 projection = glm::perspective(glm::radians(fov),
        static_cast<float>(renderer->windowWidth) / renderer->windowHeight,
        0.1f, 10000.0f);
    glUseProgram(renderer->rocketShader);
    glUniformMatrix4fv(
        glGetUniformLocation(renderer->rocketShader, "projection"),
        1, GL_FALSE, glm::value_ptr(projection)
    );
    
    glUseProgram(renderer->trajectoryShader);
    glUniformMatrix4fv(
        glGetUniformLocation(renderer->trajectoryShader, "projection"),
        1, GL_FALSE, glm::value_ptr(projection)
    );
}