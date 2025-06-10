project(sepia-filter-ds)

add_executable(1_sepia_sycl src/1_sepia_sycl.cpp)
add_executable (2_sepia_policies src/2_sepia_policies.cpp)

target_link_libraries(1_sepia_sycl tbb)
target_link_libraries(1_sepia_sycl oneDPL)
target_link_libraries(2_sepia_policies tbb)
target_link_libraries(2_sepia_policies oneDPL)

add_test(NAME sepia_sycl
	COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_FILE:1_sepia_sycl> 1000 2)
add_test(NAME sepia_policies
	COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_FILE:2_sepia_policies> 1000 2 1)
