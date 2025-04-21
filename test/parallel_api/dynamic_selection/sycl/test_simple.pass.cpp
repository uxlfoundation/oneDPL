// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include "oneapi/dpl/dynamic_selection"
#include <iostream>
#include "support/test_dynamic_load_utils.h"
#include "support/utils.h"
#if TEST_DYNAMIC_SELECTION_AVAILABLE

static inline void
build_dl_universe(std::vector<sycl::queue>& u)
{
    try
    {
        auto device_cpu1 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu1_queue(device_cpu1);
        u.push_back(cpu1_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
    try
    {
        auto device_cpu2 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu2_queue(device_cpu2);
        u.push_back(cpu2_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
}
#endif

int
main()
{
    auto device_cpu1 = sycl::device(sycl::cpu_selector_v);
    sycl::queue cpu1_queue(device_cpu1);

    

    constexpr size_t N = 1000; // Number of vectors
    constexpr size_t D = 100;  // Dimension of each vector
    std::vector<int> a(N * D);
    std::vector<int> b(N * D);

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1, 10);

    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < D; ++j)
        {
            a[i * D + j] = distribution(generator);
            b[i * D + j] = distribution(generator);
        }
    }

    std::vector<int> resultMatrix(N * N);
    sycl::buffer<int, 1> bufferA(a.data(), sycl::range<1>(N * D));
    sycl::buffer<int, 1> bufferB(b.data(), sycl::range<1>(N * D));
    sycl::buffer<int, 1> bufferResultMatrix(resultMatrix.data(), sycl::range<1>(N * N));

    std::atomic<int> probability = 0;


    auto e2 = cpu1_queue.submit([&](sycl::handler& cgh) {
        auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
        auto accessorB = bufferB.get_access<sycl::access::mode::read>(cgh);
        auto accessorResultMatrix = bufferResultMatrix.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<TestUtils::unique_kernel_name<class load2, 0>>(
            sycl::range<1>(N), [=](sycl::item<1> item) {
                for (size_t j = 0; j < N; ++j)
                {
                    int dotProduct = 0;
                    for (size_t i = 0; i < D; ++i)
                    {
                        dotProduct += accessorA[item* D + i] * accessorB[item* D + i];
                    }
                    accessorResultMatrix[item * N + j] = dotProduct;
                }
            });
        });
    e2.wait();

    return TestUtils::done();
}
