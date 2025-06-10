cmake_minimum_required(VERSION 3.12)

project(ParSTLTests)

find_package(TBB REQUIRED)
set(CMAKE_CXX_COMPILER icpx)

if(TARGET_GPU)
  #To build for Intel® Data Center GPU Max 1550 or 1100
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-pstl-offload=gpu -D GPU")
else()
  #To build for Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D CPU ")
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable (ParSTLTest main.cpp)

target_link_libraries(${PROJECT_NAME} tbb)
target_link_libraries(${PROJECT_NAME} oneDPL)

if(WIN32)
  # Add custom target for running
  add_custom_target(run ${PROJECT_NAME}.exe)
else()
  # Add custom target for running
  add_custom_target(run ./${PROJECT_NAME})
endif()
add_test(NAME ${PROJECT_NAME} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME})
