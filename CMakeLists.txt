cmake_minimum_required(VERSION 3.10)
project(aim)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置可执行文件输出目录为项目根目录下的 bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# 包含头文件目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/MindVisionSdk
)

# 收集所有源文件
set(SOURCES
    src/Config.cpp
    src/CameraDriver.cpp
    src/EKF.cpp
    src/KalmanTracker.cpp
    src/PnPSolver.cpp
    src/PreProcess.cpp
    src/SerialPort.cpp
    src/UdpLogger.cpp
    src/main.cpp
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