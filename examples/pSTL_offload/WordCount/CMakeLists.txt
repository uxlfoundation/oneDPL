project (WordCount)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} tbb)
target_link_libraries(${PROJECT_NAME} oneDPL)

add_test(NAME WordCount COMMAND $<TARGET_FILE:WordCount>)
