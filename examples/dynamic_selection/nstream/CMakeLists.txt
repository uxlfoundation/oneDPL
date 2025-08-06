project(nstream)

set(CMAKE_CXX_FLAGS "-Wall")

add_executable(nstreams_sycl ./src/nstreams_sycl.cpp)
add_executable(nstreams_policies ./src/nstreams_policies.cpp)

target_link_libraries(nstreams_sycl oneDPL)
target_link_libraries(nstreams_policies oneDPL)

add_test(NAME nstreams_sycl COMMAND $<TARGET_FILE:nstreams_sycl> 1000)
add_test(NAME nstreams_policies COMMAND $<TARGET_FILE:nstreams_policies> 1000 1)
