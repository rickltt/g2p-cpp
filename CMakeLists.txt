cmake_minimum_required(VERSION 3.18)

project(G2P)

set(CMAKE_CXX_STANDARD 17) 

find_package(glog 0.7.1 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

FILE(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

SET(ALL_SRCS ${SOURCE_FILES})

add_library(g2p SHARED ${ALL_SRCS})
target_link_libraries(g2p PUBLIC glog::glog dl)

add_executable(debug ${CMAKE_SOURCE_DIR}/debug.cpp)

target_link_libraries(debug g2p)
