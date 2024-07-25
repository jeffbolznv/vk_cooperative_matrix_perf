
vk_cooperative_matrix_perf is a sample/benchmark demonstrating performance of
using the VK_KHR_cooperative_matrix Vulkan extension, and the associated
GL_KHR_cooperative_matrix GLSL extension.

The benchmark queries the supported matrix multiply sizes and precisions from
the Vulkan implementation, and runs a couple different shaders at various
tiles sizes and reports the performance in teraflops.

Running this application requires an NVIDIA Turing or newer GPU, and a recent
driver that supports the VK_KHR_cooperative_matrix extension.

Modifying and rebuilding the shaders requires a glslangValidator.exe with
GL_KHR_cooperative_matrix support. This is included Vulkan SDK version
1.3.261.0 and newer.
