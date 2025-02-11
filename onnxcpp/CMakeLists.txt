

cmake_minimum_required(VERSION 3.13)
project(onnx_c++)


find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(argparse REQUIRED)

# Source files
set(SOURCES
    main.cpp
)


add_executable(onnx_c++ main.cpp)
target_link_libraries(onnx_c++ PRIVATE onnxruntime ${OpenCV_LIBS})