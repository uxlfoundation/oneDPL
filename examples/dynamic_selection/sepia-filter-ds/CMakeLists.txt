project(sepia-filter-ds)

add_executable(1_sepia_sycl src/1_sepia_sycl.cpp)
add_executable (2_sepia_policies src/2_sepia_policies.cpp)

target_link_libraries(1_sepia_sycl tbb)
target_link_libraries(1_sepia_sycl sycl)
target_link_libraries(2_sepia_policies tbb)
target_link_libraries(2_sepia_policies oneDPL)

add_test(NAME sepia_sycl COMMAND ${CMAKE_CURRENT_BINARY_DIR}/1_sepia_sycl 1000 2)
add_test(NAME sepia_policies COMMAND ${CMAKE_CURRENT_BINARY_DIR}/2_sepia_policies 1000 2 4)
