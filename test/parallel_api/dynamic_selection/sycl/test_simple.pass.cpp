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

    std::vector<int> resultMatrix(N);
    sycl::buffer<int, 1> bufferResultMatrix(resultMatrix.data(), sycl::range<1>(N));


    auto e2 = cpu1_queue.submit([&](sycl::handler& cgh) {
        auto accessorResultMatrix = bufferResultMatrix.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<class load2>(
            sycl::range<1>(N), [=](sycl::item<1> item) {
                accessorResultMatrix[item] = 1;
            });
        });
    e2.wait();

    return TestUtils::done();
}
