#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

cmake_minimum_required(VERSION 3.10.2)

project(vk_cooperative_matrix_perf)

set(TARGET_NAME "vk_cooperative_matrix_perf")

set(BENCHMARK_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin CACHE STRING "Binary Dir" FORCE) #force update even if it in cache
message(STATUS "BENCHMARK_BINARY_DIR='${TUTORIAL_BINARY_DIR}' ")

# Create the directory if it does not exist
file(MAKE_DIRECTORY ${BENCHMARK_BINARY_DIR})

find_package(Vulkan REQUIRED)

include_directories(${Vulkan_INCLUDE_DIR})

file( GLOB SHADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp )
foreach(shader ${SHADER_FILES})
    set_source_files_properties( ${shader} PROPERTIES HEADER_FILE_ONLY TRUE )
endforeach()

add_executable(${TARGET_NAME} src/vk_cooperative_matrix_perf.cpp ${SHADER_FILES})

target_link_libraries(${TARGET_NAME} ${Vulkan_LIBRARY})

set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BENCHMARK_BINARY_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BENCHMARK_BINARY_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${BENCHMARK_BINARY_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${BENCHMARK_BINARY_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${TARGET_NAME} PROPERTIES RELWITHDEBINFO_POSTFIX "RelWithDebInfo")

file( GLOB SPV_FILES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.spv )
file( COPY ${SPV_FILES}  DESTINATION ${BENCHMARK_BINARY_DIR}/shaders/)
message(STATUS "copy SPIR-V file into ${BENCHMARK_BINARY_DIR}/shaders/...") 
