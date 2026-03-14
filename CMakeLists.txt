cmake_minimum_required(VERSION 3.10)

project(aim)

add_library(config Config.cpp)
target_link_libraries(config yaml-cpp)