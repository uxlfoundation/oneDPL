if(TARGET_GPU)
  #To build for Intel® Data Center GPU Max 1550 or 1100
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-pstl-offload=gpu -D GPU")
else()
  #To build for Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D CPU ")
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

target_link_libraries(${PROJECT_NAME} tbb)
target_link_libraries(${PROJECT_NAME} oneDPL)

add_executable (FileWordCount main.cpp)

add_custom_target (run_fwc0
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/FileWordCount/ "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/FileWordCount$<IF:$<PLATFORM_ID:Windows>,.exe,>" GutenbergBooks 0
)
add_custom_target (run_fwc1
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/FileWordCount/ "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/FileWordCount$<IF:$<PLATFORM_ID:Windows>,.exe,>" GutenbergBooks 1
)

add_custom_target (FileWordCount ALL
    DEPENDS run_fwc0 run_fwc1
)

add_test(NAME ${PROJECT_NAME} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME})
