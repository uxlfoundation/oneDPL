add_executable(FileWordCount main.cpp)
target_link_libraries(FileWordCount tbb)
target_link_libraries(FileWordCount oneDPL)

add_custom_target (run_fwc0
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/FileWordCount/ "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/FileWordCount$<IF:$<PLATFORM_ID:Windows>,.exe,>" GutenbergBooks 0
)
add_custom_target (run_fwc1
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/FileWordCount/ "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/FileWordCount$<IF:$<PLATFORM_ID:Windows>,.exe,>" GutenbergBooks 1
)

add_custom_target (FileWordCount ALL
    DEPENDS run_fwc0 run_fwc1
)

add_test(NAME FileWordCount COMMAND ${CMAKE_CURRENT_BINARY_DIR}/FileWordCount)
