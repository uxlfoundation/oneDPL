project (FileWordCount)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} tbb)
target_link_libraries(${PROJECT_NAME} oneDPL)

add_test(NAME ${PROJECT_NAME}1 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME} GutenbergBooks 0)
add_test(NAME ${PROJECT_NAME}2 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME} GutenbergBooks 2)
