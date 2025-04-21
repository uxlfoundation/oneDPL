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


int
main()
{
    auto device_cpu1 = sycl::device(sycl::cpu_selector_v);
    sycl::queue cpu1_queue(device_cpu1);

    

    constexpr size_t N = 1000; // Number of vectors
    constexpr size_t D = 100;  // Dimension of each vector


    std::vector<int> resultMatrix(N * N);
    sycl::buffer<int, 1> bufferResultMatrix(resultMatrix.data(), sycl::range<1>(N * N));

    std::atomic<int> probability = 0;


    auto e2 = cpu1_queue.submit([&](sycl::handler& cgh) {
        auto accessorResultMatrix = bufferResultMatrix.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<TestUtils::unique_kernel_name<class load2, 0>>(
            sycl::range<1>(N), [=](sycl::item<1> item) {
                for (size_t j = 0; j < N; ++j)
                {
                    accessorResultMatrix[item * N + j] = 1;
                }
            });
        });
    e2.wait();

    return TestUtils::done();
}
