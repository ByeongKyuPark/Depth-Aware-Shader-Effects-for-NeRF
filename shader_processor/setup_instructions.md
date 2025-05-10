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
