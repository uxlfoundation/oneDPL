add_executable(WordCount main.cpp)
target_link_libraries(WordCount tbb)
target_link_libraries(WordCount oneDPL)

add_custom_target (run_wc COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/WordCount/ && "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/WordCount$<IF:$<PLATFORM_ID:Windows>,.exe,>")

add_test(NAME WordCount COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target run_wc)