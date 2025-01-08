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
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=float16_t -DC_TYPE=float16_t -V workgroup.comp -o workgroupfp16.spv
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=float16_t -DC_TYPE=float -V workgroup.comp -o workgroupfp32.spv
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=uint8_t -DC_TYPE=uint32_t -V workgroup.comp -o workgroupu8.spv
glslangValidator.exe --target-env spirv1.6 -DA_TYPE=int8_t -DC_TYPE=int32_t -V workgroup.comp -o workgroups8.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=16 -DA_TYPE=float16_t -DC_BITS=16 -DC_TYPE=float16_t -V tiled.comp -o tiledfp16.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=16 -DA_TYPE=float16_t -DC_BITS=32 -DC_TYPE=float     -V tiled.comp -o tiledfp32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=16 -DA_TYPE=float16_t -DC_BITS=16 -DC_TYPE=float16_t -V shmem.comp -o shmemfp16.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=16 -DA_TYPE=float16_t -DC_BITS=32 -DC_TYPE=float     -V shmem.comp -o shmemfp32.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=uint8_t   -DC_BITS=32 -DC_TYPE=uint32_t  -V tiled.comp -o tiledu8.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=uint8_t   -DC_BITS=32 -DC_TYPE=uint32_t  -V shmem.comp -o shmemu8.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=int8_t    -DC_BITS=32 -DC_TYPE=int32_t   -V tiled.comp -o tileds8.spv
glslangValidator.exe --target-env spirv1.3 -DA_BITS=8  -DA_TYPE=int8_t    -DC_BITS=32 -DC_TYPE=int32_t   -V shmem.comp -o shmems8.spv
