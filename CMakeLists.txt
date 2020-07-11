cmake_minimum_required(VERSION 2.8)
project(LibtorchAssist)

set(CMAKE_CXX_STANDARD 14)

# 配置依赖
MESSAGE(STATUS "platform: ${CMAKE_SYSTEM_NAME}")


FIND_PACKAGE(Torch REQUIRED)
#find_package(OpenCV REQUIRED)


ADD_EXECUTABLE(LibtorchAssist LibtorchAssist.cpp LibtorchAssist.h)
TARGET_LINK_LIBRARIES(LibtorchAssist ${TORCH_LIBRARIES} )
#${OpenCV_LIBS}