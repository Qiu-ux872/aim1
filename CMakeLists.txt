cmake_minimum_required(VERSION 3.10)
project(aim)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/MindVisionSdk/include
)

set(SOURCES
    src/Config.cpp
    src/CameraDriver.cpp
    src/EKF.cpp
    src/PnPSolver.cpp
    src/PreProcess.cpp
    src/SerialPort.cpp
    src/plotter.cpp
    src/TargetSelect.cpp
    src/main_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBRARIES}
    ${CMAKE_SOURCE_DIR}/MindVisionSdk/lib/x64/libMVSDK.so
    nlohmann_json::nlohmann_json
    pthread
)
