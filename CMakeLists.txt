cmake_minimum_required(VERSION 3.10)
project(aim)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# 设置头文件目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

include_directories(${CMAKE_SOURCE_DIR}/MindVisionSdk)

# 收集源文件
set(SOURCES
    src/Config.cpp
    src/CameraDriver.cpp          # 仅相机版需要
    src/KalmanTracker.cpp
    src/PnPSolver.cpp
    src/PreProcess.cpp
    src/SerialPort.cpp
    src/UdpLogger.cpp
    src/main.cpp                   # 或 main_test.cpp
)

# 生成可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接库
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBRARIES}
    ${CMAKE_SOURCE_DIR}/MindVisionSdk/lib/x64/libMVSDK.so
    pthread
)

# 设置运行时输出目录
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)