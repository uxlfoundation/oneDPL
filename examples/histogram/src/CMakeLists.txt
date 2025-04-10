# Add an executable target from source files
add_executable(${PROJECT_NAME} main.cpp)

if (NOT CMAKE_CXX_COMPILER MATCHES ".*dpcpp(-cl)?(.exe)?$")
  target_compile_options(${PROJECT_NAME} PRIVATE -fsycl)
  target_link_libraries(${PROJECT_NAME} sycl)
  target_link_libraries(${PROJECT_NAME} oneDPL)
endif()

# Add custom target for running
add_custom_target(run ./${PROJECT_NAME})
