cmake_minimum_required(VERSION 2.8)
project(LibtorchAssist)

set(CMAKE_CXX_STANDARD 17)

# 编译版本
#SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_BUILD_TYPE "Debug")

MESSAGE(STATUS "platform: ${CMAKE_SYSTEM_NAME}")

# 设置库文件输出目录
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
# 可执行文件输出目录
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)


if (UNIX AND NOT APPLE)
    # include Pytorch
    set(Torch_DIR "/home/tdl/software/libtorch/share/cmake/Torch")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    include_directories(${TORCH_INCLUDE_DIRS})

    # OpenCV
    set(OpenCV_DIR "/home/tdl/software/OpenCV/build_debug_x64/install/share/OpenCV/")
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

elseif (WIN32)
    MESSAGE(STATUS "windows")
    # Pytorch
    SET(Torch_DIR "D:\\libtorch-win-shared-with-deps-debug-1.5.1+cu101\\libtorch\\share\\cmake\\Torch")
    FIND_PACKAGE(Torch REQUIRED)
    INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})

    #配置 OpenCV
    set(OpenCV_DIR "D:\\opencv-4.3.0\\build_debug_x64\\install")
    find_package(OpenCV REQUIRED)
elseif (APPLE)
    MESSAGE(STATUS "mac os")
    FIND_PACKAGE(Torch REQUIRED)
    find_package(OpenCV REQUIRED)
endif ()



ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)





