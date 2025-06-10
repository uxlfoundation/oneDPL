project(nstream)

set(CMAKE_CXX_FLAGS "-Wall")

add_executable(1_nstreams_sycl ./src/1_nstreams_sycl.cpp)
add_executable(2_nstreams_policies ./src/2_nstreams_policies.cpp)

target_link_libraries(1_nstreams_sycl tbb)
target_link_libraries(1_nstreams_sycl oneDPL)
target_link_libraries(2_nstreams_policies tbb)
target_link_libraries(2_nstreams_policies oneDPL)

add_test(NAME 1_nstreams_sycl COMMAND $<TARGET_FILE:1_nstreams_sycl> 1000)
add_test(NAME 2_nstreams_policies COMMAND $<TARGET_FILE:2_nstreams_policies> 1000 1)