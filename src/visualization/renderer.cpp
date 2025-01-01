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
    // Initialize GLFW
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    window = glfwCreateWindow(windowWidth, windowHeight, "Rocket Simulation", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        throw std::runtime_error("Failed to initialize GLEW");
    }

    // Set callbacks
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);

    // Initialize components
    initShaders();
    initGeometry();
    initCamera();

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
}

void Renderer::initShaders() {
    try {
        // Get the executable path to find shader directory
        std::string shaderPath = "shaders/";
        
        // Load rocket shader program (for 3D rocket model)
        rocketShader = ShaderLoader::createShaderProgram(
            shaderPath + "rocket.vert",
            shaderPath + "rocket.frag"
        );

        // Set default lighting parameters for rocket shader
        glUseProgram(rocketShader);
        glUniform3f(glGetUniformLocation(rocketShader, "lightPos"), 100.0f, 100.0f, 100.0f);
        glUniform3f(glGetUniformLocation(rocketShader, "lightColor"), 1.0f, 1.0f, 1.0f);
        glUniform3f(glGetUniformLocation(rocketShader, "objectColor"), 0.7f, 0.7f, 0.7f);
        glUniform3f(glGetUniformLocation(rocketShader, "viewPos"), 0.0f, 0.0f, 0.0f);

        // Load trajectory shader program (for path visualization)
        trajectoryShader = ShaderLoader::createShaderProgram(
            shaderPath + "trajectory.vert",
            shaderPath + "trajectory.frag"
        );

        // Verify shader programs
        if (rocketShader == 0 || trajectoryShader == 0) {
            throw std::runtime_error("Failed to create shader programs");
        }
    } catch (const std::exception& e) {
        std::cerr << "Shader initialization failed: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::initGeometry() {
    // Create rocket geometry
    std::vector<float> vertices = {
        // ... rocket mesh vertices
    };

    std::vector<unsigned int> indices = {
        // ... rocket mesh indices
    };

    glGenVertexArrays(1, &rocketVAO);
    glGenBuffers(1, &rocketVBO);
    glGenBuffers(1, &rocketEBO);

    glBindVertexArray(rocketVAO);
    glBindBuffer(GL_ARRAY_BUFFER, rocketVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rocketEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Set vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Create trajectory geometry
    glGenVertexArrays(1, &trajectoryVAO);
    glGenBuffers(1, &trajectoryVBO);
}

void Renderer::render(const std::vector<Dynamics::State>& trajectory,
                     const Rocket& rocket) {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    updateCamera();

    // Draw trajectory
    drawTrajectory(trajectory);

    // Draw rocket at current position
    if (!trajectory.empty()) {
        drawRocket(trajectory.back());
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Renderer::initCamera() {
   // Initialize camera position and orientation
   cameraPos = glm::vec3(0.0f, -10.0f, 5.0f);  // Start behind and slightly above rocket
   cameraFront = glm::vec3(0.0f, 1.0f, -0.2f);  // Look at rocket's initial position
   cameraUp = glm::vec3(0.0f, 0.0f, 1.0f);      // Z-up coordinate system
   
   // Initialize camera angles
   yaw = -90.0f;    // Look along y-axis initially
   pitch = 0.0f;    // Horizontal view
   
   // Initialize mouse tracking
   lastX = windowWidth / 2.0f;
   lastY = windowHeight / 2.0f;
   firstMouse = true;

   // Set initial view matrix
   glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
   
   // Set initial projection matrix
   glm::mat4 projection = glm::perspective(glm::radians(45.0f),
       static_cast<float>(windowWidth) / windowHeight,
       0.1f, 10000.0f);

   // Set matrices in shaders
   glUseProgram(rocketShader);
   glUniformMatrix4fv(glGetUniformLocation(rocketShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
   glUniformMatrix4fv(glGetUniformLocation(rocketShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

   glUseProgram(trajectoryShader);
   glUniformMatrix4fv(glGetUniformLocation(trajectoryShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
   glUniformMatrix4fv(glGetUniformLocation(trajectoryShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
}

void Renderer::drawRocket(const Dynamics::State& state) {
    glUseProgram(rocketShader);

    // Set uniforms
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(state.position.x(),
                                          state.position.y(),
                                          state.position.z()));
    
    // Convert quaternion to rotation matrix
    glm::mat4 rotMat(1.0f);
    // ... quaternion to rotation matrix conversion

    model *= rotMat;

    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f),
                                          (float)windowWidth / windowHeight,
                                          0.1f, 10000.0f);

    glUniformMatrix4fv(glGetUniformLocation(rocketShader, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(rocketShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(rocketShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    // Draw rocket
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
        
        // Color based on altitude
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
    // Camera movement speed
    const float cameraSpeed = 0.05f;

    // Forward/Backward
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;

    // Left/Right strafing
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;

    // Up/Down
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

    // Constrain pitch to avoid camera flipping
    if (renderer->pitch > 89.0f)
        renderer->pitch = 89.0f;
    if (renderer->pitch < -89.0f)
        renderer->pitch = -89.0f;

    // Update camera front direction
    glm::vec3 front;
    front.x = cos(glm::radians(renderer->yaw)) * cos(glm::radians(renderer->pitch));
    front.y = sin(glm::radians(renderer->pitch));
    front.z = sin(glm::radians(renderer->yaw)) * cos(glm::radians(renderer->pitch));
    renderer->cameraFront = glm::normalize(front);
}

void Renderer::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    // Get the renderer instance
    auto* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    
    // Adjust field of view for zoom
    static float fov = 45.0f;
    fov -= static_cast<float>(yoffset);
    
    // Constrain FOV
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 45.0f)
        fov = 45.0f;
    
    // Update projection matrix with new FOV
    glm::mat4 projection = glm::perspective(glm::radians(fov),
        static_cast<float>(renderer->windowWidth) / renderer->windowHeight,
        0.1f, 10000.0f);
    
    // Update projection uniform in both shaders
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