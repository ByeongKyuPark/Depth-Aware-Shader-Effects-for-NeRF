cmake_minimum_required(VERSION 3.10)
project(NeRF_Shader_Processor)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/glad)

# Find GLFW library - fix path to lib directory
find_library(GLFW_LIBRARY
    NAMES glfw3 glfw
    PATHS ${PROJECT_SOURCE_DIR}/lib
    NO_DEFAULT_PATH
)

if (NOT GLFW_LIBRARY)
    message(FATAL_ERROR "GLFW library not found. Please place glfw3.lib in the shader_processor/lib directory.")
endif()

message(STATUS "Found GLFW: ${GLFW_LIBRARY}")

# Check for GLAD
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/glad/glad.h")
    message(STATUS "GLAD headers not found. Please download GLAD")
    message(STATUS "See setup_instructions.md for details")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/glad.c")
    message(STATUS "GLAD source not found. Please download GLAD")
    message(STATUS "See setup_instructions.md for details")
endif()

# Check for stb_image
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/stb/stb_image.h")
    message(STATUS "stb_image.h not found. Please download it.")
    message(STATUS "See setup_instructions.md for details")
endif()

# Add source files
set(SOURCES
    src/main.cpp
    src/shader.cpp
    src/texture.cpp
    src/glad.c
)

# Add executable
add_executable(shader_processor ${SOURCES})

# Link libraries - use the specific library path found above
target_link_libraries(shader_processor
    ${OPENGL_LIBRARIES}
    ${GLFW_LIBRARY}
    ${CMAKE_DL_LIBS}
)

# Copy shader files to build directory if they exist
if(EXISTS "${PROJECT_SOURCE_DIR}/../shaders")
    file(COPY ${PROJECT_SOURCE_DIR}/../shaders/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shaders)
endif()

# Create setup instructions file
file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/setup_instructions.md" "
# Setting up the Shader Processor

## Dependencies

1. **GLAD**: Download from https://glad.dav1d.de/
   - Generate with the following options:
     - Language: C/C++
     - Specification: OpenGL
     - API: gl Version 3.3
     - Profile: Core

2. Extract the files to these locations:
   - Create directory: `shader_processor/include/glad/`
   - Copy `glad.h` and `khrplatform.h` to `shader_processor/include/glad/`
   - Create directory: `shader_processor/src/`
   - Copy `glad.c` to `shader_processor/src/`

3. **GLFW**: Download from https://www.glfw.org/download.html
   - Option 1: Extract to `shader_processor/external/glfw/`
   - Option 2: Install via vcpkg: `vcpkg install glfw3`
   - Option 3: Download pre-compiled binaries

4. **stb_image**: Download from https://github.com/nothings/stb
   - Create directory: `shader_processor/include/stb/`
   - Download `stb_image.h` and `stb_image_write.h` to `shader_processor/include/stb/`

## Build Instructions

```bash
cd shader_processor
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
")
