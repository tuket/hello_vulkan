cmake_minimum_required (VERSION 3.8)
project("hello_vulkan")
set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

add_subdirectory(libs/glm)
add_subdirectory(libs/glfw)

set(COMMON_LIBS
    glm
    glfw
    vulkan
)

add_executable(hello
    main.cpp
)
target_link_libraries(hello ${COMMON_LIBS})

add_executable(hello2
    hello2.cpp
)
target_link_libraries(hello2 ${COMMON_LIBS})

add_executable(tutorial
    tutorial.cpp
)
target_link_libraries(tutorial ${COMMON_LIBS})

add_executable(tutorial2
    tutorial2.cpp
)
target_link_libraries(tutorial2 ${COMMON_LIBS})

add_executable(original_tutorial2
    original_tutorial2.cpp
)
target_link_libraries(original_tutorial2 ${COMMON_LIBS})

# SHADERS
find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

## find all the shader files under the shaders folder
file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.frag.glsl"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert.glsl"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp.glsl"
)

## iterate each shader
foreach(GLSL ${GLSL_SOURCE_FILES})
    message(STATUS "BUILDING SHADER")
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.spv")
    message(STATUS ${GLSL})
    ##execute glslang command to compile that specific shader
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
        DEPENDS ${GLSL}
    )
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    Shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
)