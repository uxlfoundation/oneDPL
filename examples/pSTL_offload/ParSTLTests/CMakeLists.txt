
add_executable (ParSTLTest main.cpp)

target_link_libraries(ParSTLTest tbb)
target_link_libraries(ParSTLTest oneDPL)

add_custom_target(run_pstl COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/ParSTLTest/ "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ParSTLTest$<IF:$<PLATFORM_ID:Windows>,.exe,>"
add_test(NAME ParSTLTest COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target run_pstl)