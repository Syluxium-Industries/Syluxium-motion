# Copyright (c) 2026 Syluxium Industries.
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.22)
project(SyluxiumMotion VERSION 0.2.0 LANGUAGES CXX)

if(NOT TARGET Vulkan::Headers)
    find_package(VulkanHeaders CONFIG QUIET)
    if(NOT TARGET Vulkan::Headers)
        find_package(Vulkan REQUIRED)
    endif()
endif()
find_program(SYLUXIUM_MOTION_GLSLC NAMES glslc REQUIRED)

set(SM_SPV ${CMAKE_CURRENT_BINARY_DIR}/spv)
file(MAKE_DIRECTORY ${SM_SPV})
set(SM_EMBED ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedSpirv.cmake)
set(SM_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/shaders/flow_common.glsl)
set(SM_EMBEDDED)
foreach(spec "luma;;luma" "down;;down" "search;;search" "search_wide;-DWIDE;search" "prop;;prop"
             "smooth;;smooth" "refine;;refine" "interp;;interp" "still;;still")
    list(GET spec 0 name)
    list(GET spec 1 flags)
    list(GET spec 2 srcname)
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/shaders/flow_${srcname}.comp)
    add_custom_command(
        OUTPUT ${SM_SPV}/flow_${name}_spv.hpp
        COMMAND ${SYLUXIUM_MOTION_GLSLC} --target-env=vulkan1.2 -O ${flags} ${src} -o ${SM_SPV}/flow_${name}.spv
        COMMAND ${CMAKE_COMMAND} -DIN=${SM_SPV}/flow_${name}.spv -DOUT=${SM_SPV}/flow_${name}_spv.hpp
                -DNAME=kFlow_${name} -P ${SM_EMBED}
        DEPENDS ${src} ${SM_COMMON} ${SM_EMBED}
        COMMENT "Syluxium Motion: compiling flow_${name}")
    list(APPEND SM_EMBEDDED ${SM_SPV}/flow_${name}_spv.hpp)
endforeach()
add_custom_target(syluxium-motion-shaders DEPENDS ${SM_EMBEDDED})

add_library(syluxium-motion STATIC
    syluxium_motion/vkfn.cpp syluxium_motion/device.cpp syluxium_motion/context.cpp)
add_library(SyluxiumMotion::Motion ALIAS syluxium-motion)
add_dependencies(syluxium-motion syluxium-motion-shaders)
target_compile_features(syluxium-motion PUBLIC cxx_std_20)
set_target_properties(syluxium-motion PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(syluxium-motion PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${SM_SPV})
target_compile_definitions(syluxium-motion PUBLIC VK_NO_PROTOTYPES)
target_link_libraries(syluxium-motion PUBLIC Vulkan::Headers ${CMAKE_DL_LIBS})
