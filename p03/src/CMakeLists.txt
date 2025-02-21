cmake_minimum_required(VERSION 3.5)

project(main)

find_package(OpenMP REQUIRED)

add_executable(main
    main.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
endif()
