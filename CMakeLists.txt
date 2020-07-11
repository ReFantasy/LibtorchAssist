cmake_minimum_required(VERSION 2.8)
project(LibtorchAssist)

set(CMAKE_CXX_STANDARD 14)

# 配置依赖
MESSAGE(STATUS "platform: ${CMAKE_SYSTEM_NAME}")
FIND_PACKAGE(Torch REQUIRED)
#find_package(OpenCV REQUIRED)


# 设置库文件输出目录
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)


ADD_LIBRARY(LibtorchAssist STATIC LibtorchAssist.cpp LibtorchAssist.h array_to_tensor.cpp array_to_tensor.h)
TARGET_LINK_LIBRARIES(LibtorchAssist ${TORCH_LIBRARIES} )
#${OpenCV_LIBS}

# 拷贝include文件到生成目录
FILE(GLOB HEADER_FILE ${PROJECT_SOURCE_DIR}/*.h ${PROJECT_SOURCE_DIR}/*.hpp) #所有头文件存入变量
FOREACH(F ${HEADER_FILE})
    file(COPY ${F} DESTINATION ${PROJECT_BINARY_DIR}/include)
ENDFOREACH(F)
