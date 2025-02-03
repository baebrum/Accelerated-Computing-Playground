cmake_minimum_required(VERSION 3.5)

project(p01)

find_package(OpenMP REQUIRED)

add_executable(p01 main.cpp)

if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    target_link_libraries(p01 PRIVATE OpenMP::OpenMP_CXX)
endif()
