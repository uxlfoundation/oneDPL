// SYCL USM shared memory reproducer with queue.copy and TBB parallel_for
// Allocates USM shared memory, copies from vector via queue.copy, then uses TBB to increment

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <thread>
#include <sycl/sycl.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

int main() {
    try {
        // Get the default SYCL queue
        sycl::queue q{sycl::default_selector_v};
        
        std::cout << "Running on device: " 
                  << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        
        const size_t N = 1000;
        
        // Create source vector
        std::vector<int> source_data(N);
        for (size_t i = 0; i < N; ++i) {
            source_data[i] = static_cast<int>(i * 10);  // Initialize with 0, 10, 20, 30, ...
        }
        
        // Allocate USM shared memory
        int* usm_data = sycl::malloc_shared<int>(N, q);
        if (!usm_data) {
            std::cerr << "Failed to allocate USM shared memory" << std::endl;
            return 1;
        }
        
        std::cout << "Starting 5 iterations of copy + increment cycle..." << std::endl;
        
        // Repeat the process 5 times
        for (int iteration = 0; iteration < 5; ++iteration) {
            std::cout << "\n--- Iteration " << (iteration + 1) << " ---" << std::endl;
            
            // Copy data from vector to USM shared memory using queue.copy
            std::cout << "Copying data from vector to USM shared memory..." << std::endl;
            q.copy(source_data.data(), usm_data, N).wait();
            
#if 0 // turning this section on would make it work by loading the data to host before parallel section
            // Print first few values before increment
            std::cout << "Before increment: ";
            for (size_t i = 0; i < 5; ++i) {
                std::cout << usm_data[i] << " ";
            }
            std::cout << "..." << std::endl;
#endif
#if 0 // turning this section on also makes it work... I dont know why, 
      // but perhaps the copy operation is still doing something past wait() on its event
            // sleep for 50 ms
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
#endif

            // Immediately increment each element using TBB parallel_for
            std::cout << "Incrementing USM data with TBB parallel_for..." << std::endl;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, N), 
                [usm_data](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        usm_data[i] += 1;
                    }
                });
            
            // Print first few values after increment
            std::cout << "After increment:  ";
            for (size_t i = 0; i < 5; ++i) {
                std::cout << usm_data[i] << " ";
            }
            std::cout << "..." << std::endl;
            
            // Verify the increment was applied correctly
            bool success = true;
            for (size_t i = 0; i < N; ++i) {
                int expected = source_data[i] + 1;
                if (usm_data[i] != expected) {
                    std::cerr << "Error at index " << i << ": expected " 
                              << expected << ", got " << usm_data[i] << std::endl;
                    success = false;
                    break;
                }
            }
            
            if (success) {
                std::cout << "Iteration " << (iteration + 1) << " successful!" << std::endl;
            } else {
                std::cout << "Iteration " << (iteration + 1) << " failed!" << std::endl;
                // Clean up and exit on failure
                sycl::free(usm_data, q);
                return 1;
            }
        }
        
        std::cout << "\n SUCCESS: All 5 iterations completed successfully!" << std::endl;
        std::cout << "USM shared memory, queue.copy, and TBB parallel_for all working correctly." << std::endl;
        
        // Clean up
        sycl::free(usm_data, q);
        
        return 0;
        
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return 1;
    }
}