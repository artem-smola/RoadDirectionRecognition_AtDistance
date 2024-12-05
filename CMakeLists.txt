set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

cmake_minimum_required(VERSION 2.8.3)
project(RoadRecognition)


set(ONNXRUNTIME_ROOTDIR "/home/artem/onnxruntime-linux-x64-1.20.0")  # Замените на ваш путь

include_directories(${ONNXRUNTIME_ROOTDIR}/include)

link_directories(${ONNXRUNTIME_ROOTDIR}/lib)


# Options
option(ENABLE_CUDA "Enable CUDA" OFF)

if(ENABLE_CUDA)
    message(STATUS "CUDA is enabled")
    add_definitions(-DENABLE_CUDA)
endif()

find_package(OpenCV REQUIRED)

set(HEADER_FILES
TwinLiteNet-onnxruntime/include/twinlitenet_onnxruntime.hpp
src/road_recognition/recognition.hpp)

set(SOURCE_FILES
    main.cpp
    TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp
    src/road_recognition/recognition.cpp
    )

# general Includes
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

# cuda includes
if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(
        ${CUDA_INCLUDE_DIRS}
    )
endif()

add_executable(main ${SOURCE_FILES} ${HEADER_FILES})

target_include_directories(main PRIVATE 
    ${ONNXRUNTIME_ROOTDIR}/include 
    TwinLiteNet-onnxruntime/include 
    src/road_recognition)

target_link_libraries(main
    ${OpenCV_LIBS})

target_link_libraries(main onnxruntime) 

# cuda libs
if(ENABLE_CUDA)
    target_link_libraries(main
        ${CUDA_LIBRARIES})
endif()