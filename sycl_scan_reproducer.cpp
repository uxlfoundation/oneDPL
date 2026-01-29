// Reproducer for sycl::inclusive_scan_over_group bug
// Issue: inclusive_scan_over_group fails with mask values (0,1) and std::plus on Linux GPU release builds
// Observed with: Intel(R) oneAPI DPC++/C++ Compiler 2025.2 and nightly builds
// Devices affected: Intel GPU (OpenCL and Level Zero backends)
// Build configuration: Release builds on Linux only
//
// Key insight from oneDPL code (unseq_backend_sycl.h lines 962-972):
// 1. Data is loaded into local memory
// 2. A unary op reads from local memory to produce the value to scan
// 3. There's a barrier BEFORE the scan
// 4. The scan uses an init parameter that carries over from previous iterations

#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>

int main() {
    constexpr size_t work_group_size = 1024;
    constexpr size_t elements_per_wg = 1024;  // Process 1024 elements per work group
    constexpr size_t iters_per_wg = elements_per_wg / work_group_size;  // 4 iterations
    constexpr size_t n_work_groups = 4;
    constexpr size_t n_elements = elements_per_wg * n_work_groups;
    constexpr int num_test_patterns = 1000000;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 1);

    try {
        sycl::queue q{sycl::gpu_selector_v};
        std::cout << "Running on device: "
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";
        std::cout << "Work group size: " << work_group_size << "\n";
        std::cout << "Elements per work group: " << elements_per_wg << "\n";
        std::cout << "Iterations per work group: " << iters_per_wg << "\n";
        std::cout << "Number of work groups: " << n_work_groups << "\n";
        std::cout << "Total elements: " << n_elements << "\n";
        std::cout << "Running " << num_test_patterns << " random test patterns...\n\n";

        int total_tests = 0;
        int failed_tests = 0;
        int first_failure = -1;

        std::vector<int> input_data(n_elements);
        std::vector<int> output_data(n_elements);
        std::vector<int> expected_data(n_elements);

        for (int test = 0; test < num_test_patterns; ++test) {
            // Generate random mask pattern: 0 or 1
            for (size_t i = 0; i < n_elements; ++i) {
                input_data[i] = distrib(gen);
            }

            // Calculate expected results serially - cumulative sum within each work group
            for (size_t wg = 0; wg < n_work_groups; ++wg) {
                int cumsum = 0;
                for (size_t i = 0; i < elements_per_wg; ++i) {
                    size_t global_idx = wg * elements_per_wg + i;
                    cumsum += input_data[global_idx];
                    expected_data[global_idx] = cumsum;
                }
            }

            // Run on GPU - EXACT pattern from oneDPL unseq_backend_sycl.h:962-983
            {
                sycl::buffer<int> input_buf(input_data.data(), sycl::range<1>(n_elements));
                sycl::buffer<int> output_buf(output_data.data(), sycl::range<1>(n_elements));

                q.submit([&](sycl::handler& cgh) {
                    auto input_acc = input_buf.get_access<sycl::access::mode::read>(cgh);
                    auto output_acc = output_buf.get_access<sycl::access::mode::write>(cgh);
                    sycl::local_accessor<int, 1> local_acc(sycl::range<1>(work_group_size), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<1>(n_work_groups * work_group_size, work_group_size),
                        [=](sycl::nd_item<1> item) {
                            auto group = item.get_group();
                            auto local_id = item.get_local_id(0);
                            auto group_id = item.get_group(0);

                            // Identity for std::plus<int>
                            int adder = 0;

                            size_t adjusted_global_id = local_id + elements_per_wg * group_id;

                            // Multi-iteration loop - matches unseq_backend_sycl.h:960
                            for (size_t iter = 0; iter < iters_per_wg;
                                 ++iter, adjusted_global_id += work_group_size) {

                                // Load into local memory - matches line 962-965
                                if (adjusted_global_id < n_elements)
                                    local_acc[local_id] = input_acc[adjusted_global_id];
                                else
                                    local_acc[local_id] = 0;  // identity

                                // Apply unary op - matches line 968
                                // In oneDPL this reads from local_acc and produces mask value (0 or 1)
                                // For simplicity we just read the value which is already 0 or 1
                                int old_value = local_acc[local_id];

                                // CRITICAL: Barrier BEFORE scan - matches line 969
                                sycl::group_barrier(group);

                                // BUG LOCATION: inclusive_scan_over_group with init - matches line 972
                                local_acc[local_id] = sycl::inclusive_scan_over_group(
                                    group,
                                    old_value,
                                    std::plus<int>(),
                                    adder  // <-- Carry-over from previous iteration!
                                );

                                // Barrier after scan - matches line 981
                                sycl::group_barrier(group);

                                // Update adder for next iteration - matches line 983
                                adder = local_acc[work_group_size - 1];

                                // Store result
                                if (adjusted_global_id < n_elements) {
                                    output_acc[adjusted_global_id] = local_acc[local_id];
                                }
                            }
                        }
                    );
                }).wait();
            }

            // Verify results
            bool test_passed = true;
            size_t first_error_idx = 0;
            size_t first_error_wg = 0;

            for (size_t wg = 0; wg < n_work_groups; ++wg) {
                for (size_t i = 0; i < elements_per_wg; ++i) {
                    size_t global_idx = wg * elements_per_wg + i;

                    if (output_data[global_idx] != expected_data[global_idx]) {
                        if (test_passed && first_failure < 0) {
                            first_failure = test;
                            first_error_idx = global_idx;
                            first_error_wg = wg;

                            std::cout << "\n=== First failure on test " << test << " ===\n";
                            std::cout << "Failed in work group " << wg << ", global index " << global_idx << "\n";
                            std::cout << "Showing first 40 elements of failing work group:\n";
                            std::cout << "Idx | Input | Expected | Got\n";
                            std::cout << "----|-------|----------|-----\n";

                            size_t wg_start = wg * elements_per_wg;
                            for (size_t j = 0; j < std::min(size_t(40), elements_per_wg); ++j) {
                                size_t idx = wg_start + j;
                                std::cout << std::setw(3) << j << " | "
                                          << std::setw(5) << input_data[idx] << " | "
                                          << std::setw(8) << expected_data[idx] << " | "
                                          << std::setw(4) << output_data[idx];
                                if (output_data[idx] != expected_data[idx]) {
                                    std::cout << " <- MISMATCH";
                                }
                                std::cout << "\n";
                            }
                            std::cout << "...\n";
                        }
                        test_passed = false;
                        break;
                    }
                }
                if (!test_passed) break;
            }

            total_tests++;
            if (!test_passed) {
                failed_tests++;
            }

            // Progress indicator
            if ((test + 1) % 1000 == 0) {
                std::cout << "Completed " << (test + 1) << " tests, "
                          << failed_tests << " failures so far\n";
            }
        }

        std::cout << "\n=== Summary ===\n";
        std::cout << "Total tests: " << total_tests << "\n";
        std::cout << "Passed: " << (total_tests - failed_tests) << "\n";
        std::cout << "Failed: " << failed_tests << "\n";

        if (failed_tests == 0) {
            std::cout << "PASSED: All tests successful\n";
            return 0;
        } else {
            std::cout << "FAILED: " << failed_tests << " out of " << total_tests
                      << " tests failed (" << (100.0 * failed_tests / total_tests) << "%)\n";
            std::cout << "First failure at test: " << first_failure << "\n";
            return 1;
        }

    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << "\n";
        return 2;
    }
}
