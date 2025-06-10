project(nstream)

set(CMAKE_CXX_FLAGS "-Wall")

add_executable(1_nstreams_sycl ./src/1_nstreams_sycl.cpp)
add_executable(2_nstreams_policies ./src/2_nstreams_policies.cpp)

target_link_libraries(1_nstreams_sycl tbb)
target_link_libraries(1_nstreams_sycl oneDPL)
target_link_libraries(2_nstreams_policies tbb)
target_link_libraries(2_nstreams_policies oneDPL)

add_test(NAME nstreams_sycl COMMAND ${CMAKE_CURRENT_BINARY_DIR}/1_nstreams_sycl 1000)
add_test(NAME nstreams_policies COMMAND ${CMAKE_CURRENT_BINARY_DIR}/2_nstreams_policies 1000 5)