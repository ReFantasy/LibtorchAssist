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
    # OpenCV
    set(OpenCV_DIR "/home/tdl/software/OpenCV/build_debug_x64/install/share/OpenCV/")
elseif (WIN32)
	IF(CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "None")      
        SET(Torch_DIR "D:\\Libtorch\\Debug\\share\\cmake\\Torch")
		   
    ELSEIF (CMAKE_BUILD_TYPE MATCHES "Release")
        SET(Torch_DIR "D:\\Libtorch\\Release\\share\\cmake\\Torch")
	ENDIF()
elseif (APPLE)
    
endif ()

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Torch REQUIRED)    
#INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)





