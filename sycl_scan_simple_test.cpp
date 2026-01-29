// Simplest possible test for inclusive_scan_over_group with init parameter
// Tests if the SYCL implementation matches expected semantics
// Runs 10000 random test patterns

#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

int main() {
    constexpr size_t wg_size = 256;
    constexpr int num_test_patterns = 10000;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> mask_distrib(0, 1);
    std::uniform_int_distribution<> init_distrib(0, 100);

    sycl::queue q{sycl::gpu_selector_v};
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Work group size: " << wg_size << "\n";
    std::cout << "Running " << num_test_patterns << " random test patterns...\n\n";

    int total_tests = 0;
    int sycl_failed = 0;
    int custom_failed = 0;
    int first_sycl_failure = -1;
    int first_custom_failure = -1;

    std::vector<int> input(wg_size);
    std::vector<int> output_sycl(wg_size);
    std::vector<int> output_custom(wg_size);
    std::vector<int> expected(wg_size);

    for (int test = 0; test < num_test_patterns; ++test) {
        // Generate random mask pattern (0 or 1) and random init value
        for (size_t i = 0; i < wg_size; ++i) {
            input[i] = mask_distrib(gen);
        }
        int init_value = init_distrib(gen);

        // Test SYCL inclusive_scan_over_group with init
        {
            sycl::buffer<int> in_buf(input.data(), sycl::range<1>(wg_size));
            sycl::buffer<int> out_buf(output_sycl.data(), sycl::range<1>(wg_size));

            q.submit([&](sycl::handler& h) {
                auto in = in_buf.get_access<sycl::access::mode::read>(h);
                auto out = out_buf.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::nd_range<1>(wg_size, wg_size), [=](sycl::nd_item<1> item) {
                    auto lid = item.get_local_id(0);
                    int value = in[lid];

                    // SYCL scan with init
                    int result = sycl::inclusive_scan_over_group(
                        item.get_group(),
                        value,
                        std::plus<int>(),
                        init_value
                    );

                    out[lid] = result;
                });
            }).wait();
        }

        // Test custom implementation (matching oneDPL's #else branch)
        {
            sycl::buffer<int> in_buf(input.data(), sycl::range<1>(wg_size));
            sycl::buffer<int> out_buf(output_custom.data(), sycl::range<1>(wg_size));

            q.submit([&](sycl::handler& h) {
                auto in = in_buf.get_access<sycl::access::mode::read>(h);
                auto out = out_buf.get_access<sycl::access::mode::write>(h);
                sycl::local_accessor<int, 1> local_acc(sycl::range<1>(wg_size), h);

                h.parallel_for(sycl::nd_range<1>(wg_size, wg_size), [=](sycl::nd_item<1> item) {
                    auto lid = item.get_local_id(0);
                    int value = in[lid];

                    // Store in local memory
                    local_acc[lid] = value;
                    sycl::group_barrier(item.get_group());

                    // Custom scan implementation (from oneDPL #else branch)
                    int scan_result = value;
                    for (size_t i = lid; i > 0; --i) {
                        scan_result = std::plus<int>()(local_acc[i - 1], scan_result);
                    }
                    sycl::group_barrier(item.get_group());

                    // Add init value
                    int result = std::plus<int>()(init_value, scan_result);
                    out[lid] = result;
                });
            }).wait();
        }

        // Calculate expected results
        int cumsum = init_value;
        for (size_t i = 0; i < wg_size; ++i) {
            cumsum += input[i];
            expected[i] = cumsum;
        }

        // Check SYCL results
        bool sycl_test_passed = true;
        for (size_t i = 0; i < wg_size; ++i) {
            if (output_sycl[i] != expected[i]) {
                if (sycl_test_passed && first_sycl_failure < 0) {
                    first_sycl_failure = test;
                    std::cout << "\n=== First SYCL failure on test " << test << " ===\n";
                    std::cout << "Init value: " << init_value << "\n";
                    std::cout << "Idx | Input | Expected | SYCL\n";
                    std::cout << "----|-------|----------|-----\n";
                    for (size_t j = 0; j < std::min(size_t(30), wg_size); ++j) {
                        std::cout << std::setw(3) << j << " | "
                                  << std::setw(5) << input[j] << " | "
                                  << std::setw(8) << expected[j] << " | "
                                  << std::setw(4) << output_sycl[j];
                        if (output_sycl[j] != expected[j]) {
                            std::cout << " <- MISMATCH";
                        }
                        std::cout << "\n";
                    }
                    std::cout << "...\n";
                }
                sycl_test_passed = false;
                break;
            }
        }

        // Check custom results
        bool custom_test_passed = true;
        for (size_t i = 0; i < wg_size; ++i) {
            if (output_custom[i] != expected[i]) {
                if (custom_test_passed && first_custom_failure < 0) {
                    first_custom_failure = test;
                    std::cout << "\n=== First CUSTOM failure on test " << test << " ===\n";
                    std::cout << "Init value: " << init_value << "\n";
                    std::cout << "Idx | Input | Expected | Custom\n";
                    std::cout << "----|-------|----------|-------\n";
                    for (size_t j = 0; j < std::min(size_t(30), wg_size); ++j) {
                        std::cout << std::setw(3) << j << " | "
                                  << std::setw(5) << input[j] << " | "
                                  << std::setw(8) << expected[j] << " | "
                                  << std::setw(6) << output_custom[j];
                        if (output_custom[j] != expected[j]) {
                            std::cout << " <- MISMATCH";
                        }
                        std::cout << "\n";
                    }
                    std::cout << "...\n";
                }
                custom_test_passed = false;
                break;
            }
        }

        total_tests++;
        if (!sycl_test_passed) sycl_failed++;
        if (!custom_test_passed) custom_failed++;

        // Progress indicator
        if ((test + 1) % 1000 == 0) {
            std::cout << "Completed " << (test + 1) << " tests, "
                      << "SYCL failures: " << sycl_failed
                      << ", Custom failures: " << custom_failed << "\n";
        }
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "Total tests: " << total_tests << "\n";
    std::cout << "SYCL passed: " << (total_tests - sycl_failed) << "\n";
    std::cout << "SYCL failed: " << sycl_failed << "\n";
    std::cout << "Custom passed: " << (total_tests - custom_failed) << "\n";
    std::cout << "Custom failed: " << custom_failed << "\n";

    if (sycl_failed == 0 && custom_failed == 0) {
        std::cout << "\nPASSED: Both implementations work correctly\n";
        return 0;
    } else {
        if (sycl_failed > 0) {
            std::cout << "\nFAILED: SYCL implementation has bugs ("
                      << (100.0 * sycl_failed / total_tests) << "% failure rate)\n";
            std::cout << "First SYCL failure at test: " << first_sycl_failure << "\n";
        }
        if (custom_failed > 0) {
            std::cout << "\nFAILED: Custom implementation has bugs ("
                      << (100.0 * custom_failed / total_tests) << "% failure rate)\n";
            std::cout << "First custom failure at test: " << first_custom_failure << "\n";
        }
        return 1;
    }
}
