cmake_minimum_required(VERSION 3.5)

# Options.
set(CMAKE_CXX_STANDARD 11)

# Project name.
project(gcn LANGUAGES CXX)

# Include ZeroMQ.
set(ZMQ_INC /usr/local/include ../../src/common)
set(ZMQ_LIB /usr/local/lib/libzmq.a)
include_directories(${ZMQ_INC})

# Include Threads.
find_package(ZLIB)
find_package(Threads REQUIRED)

# Include AWS Lambda runtime.
find_package(aws-lambda-runtime REQUIRED)

# Include AWS SDK for Json Utils
find_package(AWSSDK REQUIRED COMPONENTS core)

# Specify the OpenBLAS library and include directories
set(OpenBLAS_INCLUDE_DIR "/usr/lib/x86_64-linux-gnu/openblas-pthread/cmake/openblas")
set(OpenBLAS_LIBRARIES "/usr/lib/x86_64-linux-gnu/libopenblas.so")

# Add the OpenBLAS include directory to the include path
include_directories(${OpenBLAS_INCLUDE_DIR})

add_subdirectory(../../common common)
add_subdirectory(ops)

# Create the executable target.
add_executable(${PROJECT_NAME} "main.cpp")

# Link against libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ops PUBLIC common Threads::Threads ${AWSSDK_LINK_LIBRARIES} AWS::aws-lambda-runtime ${OBLIB} ${CBLIB} ${ZMQ_LIB} ${OpenBLAS_LIBRARIES})

# Create the packaging target.
aws_lambda_package_target(${PROJECT_NAME})
