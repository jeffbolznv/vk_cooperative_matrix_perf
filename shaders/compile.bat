@echo off
REM
REM SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: MIT
REM
REM Permission is hereby granted, free of charge, to any person obtaining a
REM copy of this software and associated documentation files (the "Software"),
REM to deal in the Software without restriction, including without limitation
REM the rights to use, copy, modify, merge, publish, distribute, sublicense,
REM and/or sell copies of the Software, and to permit persons to whom the
REM Software is furnished to do so, subject to the following conditions:
REM
REM The above copyright notice and this permission notice shall be included in
REM all copies or substantial portions of the Software.
REM
REM THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
REM IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
REM FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
REM THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
REM LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
REM FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
REM DEALINGS IN THE SOFTWARE.
REM
@echo on
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=bfloat16_t              -DC_TYPE=float32_t -V workgroup.comp -o workgroupbf16_fp32.spv
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=float16_t               -DC_TYPE=float16_t -V workgroup.comp -o workgroupfp16_fp16.spv
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=float16_t               -DC_TYPE=float32_t -V workgroup.comp -o workgroupfp16_fp32.spv
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=uint8_t                 -DC_TYPE=uint32_t  -V workgroup.comp -o workgroupu8_u32.spv
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=int8_t                  -DC_TYPE=int32_t   -V workgroup.comp -o workgroups8_s32.spv
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=floate5m2_t             -DC_TYPE=float16_t -V workgroup.comp -o workgroupe5m2_fp16.spv
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=floate4m3_t             -DC_TYPE=float16_t -V workgroup.comp -o workgroupe4m3_fp16.spv
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=floate5m2_t             -DC_TYPE=float32_t -V workgroup.comp -o workgroupe5m2_fp32.spv
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=floate4m3_t             -DC_TYPE=float32_t -V workgroup.comp -o workgroupe4m3_fp32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=16 -DA_TYPE=float16_t   -DC_TYPE=float16_t -V tiled.comp -o tiledfp16_fp16.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=16 -DA_TYPE=bfloat16_t  -DC_TYPE=float32_t -V tiled.comp -o tiledbf16_fp32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=16 -DA_TYPE=float16_t   -DC_TYPE=float32_t -V tiled.comp -o tiledfp16_fp32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=16 -DA_TYPE=float16_t   -DC_TYPE=float16_t -V shmem.comp -o shmemfp16_fp16.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=16 -DA_TYPE=bfloat16_t  -DC_TYPE=float32_t -V shmem.comp -o shmembf16_fp32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=16 -DA_TYPE=float16_t   -DC_TYPE=float32_t -V shmem.comp -o shmemfp16_fp32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=uint8_t     -DC_TYPE=uint32_t  -V tiled.comp -o tiledu8_u32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=uint8_t     -DC_TYPE=uint32_t  -V shmem.comp -o shmemu8_u32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=int8_t      -DC_TYPE=int32_t   -V tiled.comp -o tileds8_s32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=int8_t      -DC_TYPE=int32_t   -V shmem.comp -o shmems8_s32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=floate5m2_t -DC_TYPE=float16_t -V tiled.comp -o tilede5m2_fp16.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=floate4m3_t -DC_TYPE=float16_t -V tiled.comp -o tilede4m3_fp16.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=floate5m2_t -DC_TYPE=float32_t -V tiled.comp -o tilede5m2_fp32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=floate4m3_t -DC_TYPE=float32_t -V tiled.comp -o tilede4m3_fp32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=floate5m2_t -DC_TYPE=float16_t -V shmem.comp -o shmeme5m2_fp16.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=floate4m3_t -DC_TYPE=float16_t -V shmem.comp -o shmeme4m3_fp16.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=floate5m2_t -DC_TYPE=float32_t -V shmem.comp -o shmeme5m2_fp32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=floate4m3_t -DC_TYPE=float32_t -V shmem.comp -o shmeme4m3_fp32.spv
