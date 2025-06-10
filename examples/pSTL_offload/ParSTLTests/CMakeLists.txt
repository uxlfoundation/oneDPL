project (ParSTLTests)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} tbb)
target_link_libraries(${PROJECT_NAME} oneDPL)

add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)