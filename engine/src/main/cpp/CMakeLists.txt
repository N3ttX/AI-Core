cmake_minimum_required(VERSION 3.22.1)
project(aicore_engine C CXX)

# Optimized build even for debug APKs (see engine/build.gradle.kts).
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

include(FetchContent)

# Pinned llama.cpp release. Update together with llama_jni.cpp when the C API changes.
set(LLAMA_CPP_TAG b11852)

# Static llama/ggml linked into one JNI library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
# ARMv8.2 with dot product and fp16: every 64-bit phone SoC since ~2019.
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)

FetchContent_Declare(
    llama_cpp
    URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMA_CPP_TAG}.tar.gz
)
FetchContent_MakeAvailable(llama_cpp)

add_library(aicore_engine SHARED llama_jni.cpp)
target_compile_features(aicore_engine PRIVATE cxx_std_17)
target_link_libraries(aicore_engine PRIVATE llama ggml android log)
