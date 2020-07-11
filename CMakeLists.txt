cmake_minimum_required(VERSION 2.8)
project(LibtorchAssist)

set(CMAKE_CXX_STANDARD 14)

MESSAGE(STATUS "platform: ${CMAKE_SYSTEM_NAME}")

# 设置库文件输出目录
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

ADD_SUBDIRECTORY(src)





