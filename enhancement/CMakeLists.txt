cmake_minimum_required(VERSION 3.1)
project(imgEnhancement)
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(DSIHE DSIHE.cpp)
add_executable(BBHE BBHE.cpp)
add_executable(contrast contrast.cpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

target_link_libraries(DSIHE ${OpenCV_LIBS})
target_link_libraries(BBHE ${OpenCV_LIBS})
target_link_libraries(contrast ${OpenCV_LIBS})