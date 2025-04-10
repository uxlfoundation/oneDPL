# Add an executable target from source files
add_executable(${PROJECT_NAME} main.cpp)
# Specify libraries to link with
target_link_libraries(${PROJECT_NAME} tbb)
target_link_libraries(${PROJECT_NAME} oneDPL)

if(WIN32)
  # Add custom target for running
  add_custom_target(run ${PROJECT_NAME}.exe)
  add_test(NAME ${PROJECT_NAME} COMMAND ./${PROJECT_NAME}.exe)
else()
  # Add custom target for running
  add_custom_target(run ./${PROJECT_NAME})
  add_test(NAME ${PROJECT_NAME} COMMAND ./${PROJECT_NAME})
endif()
