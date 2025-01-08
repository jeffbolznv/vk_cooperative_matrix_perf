
vk_cooperative_matrix_perf is a sample/benchmark demonstrating performance of
using the VK_KHR_cooperative_matrix and VK_NV_cooperative_matrix2 Vulkan
extensions, and the associated GL_KHR_cooperative_matrix and
GL_NV_cooperative_matrix2 GLSL extensions.

The benchmark queries the supported matrix multiply sizes and precisions from
the Vulkan implementation, and runs a few different shaders at various
tiles sizes and reports the performance in teraflops.

Running this application requires an NVIDIA Turing or newer GPU, and a recent
driver that supports the VK_KHR_cooperative_matrix extension.

Modifying and rebuilding the shaders requires a glslangValidator.exe with
GL_KHR_cooperative_matrix and GL_NV_cooperative_matrix2 support.

Run the benchmark from the base directory, e.g. build\RelWithDebInfo\vk_cooperative_matrix_perf.exe

For more serious usage of cooperative matrix, check out
https://github.com/ggerganov/llama.cpp.
