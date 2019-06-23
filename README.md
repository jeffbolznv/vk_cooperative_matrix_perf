
vk_cooperative_matrix_perf is a sample/benchmark demonstrating performance of
using the VK_NV_cooperative_matrix Vulkan extension, and the associated
GL_NV_cooperative_matrix GLSL extension.

The benchmark queries the supported matrix multiply sizes and precisions from
the Vulkan implementation, and runs a couple different shaders at various
tiles sizes and reports the performance in teraflops. All matrices are
row-major in memory.

Running this application requires an NVIDIA Turing GPU, and a recent driver
that supports the VK_NV_cooperative_matrix extension (available from
https://developer.nvidia.com/vulkan-driver).

Modifying and rebuilding the shaders requires a glslangValidator.exe with
GL_NV_cooperative_matrix support. This is included in Vulkan SDK versions
1.1.106.0 and newer.
