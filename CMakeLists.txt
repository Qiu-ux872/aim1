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
    ${OpenCV_INCLUDE_DIRS}cmake_minimum_required(VERSION 3.10)

    project(aim)
    
    add_library(config Config.cpp)
    target_link_libraries(config yaml-cpp)
    ${YAML_CPP_INCLUDE_DIR}
)

# 添加 MindVision SDK 头文件路径（假设 SDK 头文件位于 MindVisionSdk 目录）
include_directories(${CMAKE_SOURCE_DIR}/MindVisionSdk)

# 收集源文件
set(SOURCES
    src/Config.cpp
    src/CameraDriver.cpp
    src/KalmanTracker.cpp
    src/PnPSolver.cpp
    src/PreProcess.cpp
    src/SerialPort.cpp
    src/main.cpp
)

# 生成可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接库
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBRARIES}
    # MindVision SDK 库（根据实际库名修改）
    ${CMAKE_SOURCE_DIR}/MindVisionSdk/libMVSDK.so   # 如果 SDK 是静态库或动态库，按实际情况指定路径
    pthread   # 有时需要线程库
)

# 如果 MindVision SDK 库是系统路径安装的，可以用 find_library 或直接指定 -lMVSDK
# 这里假设 SDK 库位于项目 MindVisionSdk 目录下，请根据实际调整

# 设置运行时输出目录（可选）
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)