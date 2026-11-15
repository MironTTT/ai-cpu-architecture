cmake_minimum_required(VERSION 3.16)
project(graph_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(graph_evolve graph_evolve.cpp)

if(MSVC)
    target_compile_options(graph_evolve PRIVATE /W4 /utf-8 /O2)
else()
    target_compile_options(graph_evolve PRIVATE -Wall -Wextra -O3)
endif()

# Бинарник кладём рядом с исходником, а не в build/
set_target_properties(graph_evolve PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}")
