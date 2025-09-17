// Simple USM shared memory reproducer with oneDPL permutation iterator
// Allocates USM shared data and uses permutation iterator with counting iterator as map

#include <iostream>
#include <vector>
#include <cassert>

#include <sycl/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

int main() {
    try {
        // Get the default SYCL queue
        sycl::queue q{sycl::default_selector_v};
        
        std::cout << "Running on device: " 
                  << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        
        const size_t N = 1000;
        
        // Allocate USM shared memory
        int* data = sycl::malloc_shared<int>(N, q);
        if (!data) {
            std::cerr << "Failed to allocate USM shared memory" << std::endl;
            return 1;
        }
        
        // Initialize data on host
        std::cout << "Initializing " << N << " elements..." << std::endl;
        for (size_t i = 0; i < N; ++i) {
            data[i] = static_cast<int>(i);
        }
        
        // Create a permutation iterator using counting iterator as map
        auto counting_iter = oneapi::dpl::counting_iterator<size_t>(0);
        auto perm_iter = oneapi::dpl::make_permutation_iterator(data, counting_iter);
        
        // Print first few initial values via permutation iterator
        std::cout << "Initial values (via permutation iterator): ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << *(perm_iter + i) << " ";
        }
        std::cout << "..." << std::endl;
        
        // Increment each value by 1 on the host using permutation iterator
        std::cout << "Incrementing values via permutation iterator..." << std::endl;
        for (size_t i = 0; i < N; ++i) {
            *(perm_iter + i) += 1;
        }
        
        // Print first few modified values via permutation iterator
        std::cout << "After increment (via permutation iterator): ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << *(perm_iter + i) << " ";
        }
        std::cout << "..." << std::endl;
        
        // Verify all values were incremented correctly
        bool success = true;
        for (size_t i = 0; i < N; ++i) {
            if (data[i] != static_cast<int>(i + 1)) {
                std::cerr << "Error at index " << i << ": expected " 
                          << (i + 1) << ", got " << data[i] << std::endl;
                success = false;
                break;
            }
        }
        
        if (success) {
            std::cout << "SUCCESS: All " << N << " values incremented correctly via permutation iterator!" << std::endl;
        } else {
            std::cout << "FAILED: Value mismatch detected" << std::endl;
        }
        
        // Clean up
        sycl::free(data, q);
        
        return success ? 0 : 1;
        
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return 1;
    }
}