project(sepia-filter-ds)

add_executable(sepia_sycl src/sepia_sycl.cpp)
add_executable (sepia_policies src/sepia_policies.cpp)

target_link_libraries(sepia_sycl oneDPL)
target_link_libraries(sepia_policies oneDPL)

add_test(NAME sepia_sycl
	COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_FILE:sepia_sycl> 1000 2)
add_test(NAME sepia_policies
	COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_FILE:sepia_policies> 1000 2 1)
