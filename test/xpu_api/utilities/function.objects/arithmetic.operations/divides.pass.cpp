// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include <oneapi/dpl/functional>

#include "support/utils.h"

class KernelDividesTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);

    int div_array[2] = {10, 5};
    sycl::range<1> numOfItems2{2};
    sycl::buffer<std::int32_t, 1> div_buffer(div_array, numOfItems2);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        auto div_access = div_buffer.get_access<sycl::access::mode::read>(cgh);
        cgh.single_task<class KernelDividesTest>([=]() {
            const dpl::divides<int> f1;
            ret_access[0] = (f1(36, 4) == 9);
            ret_access[0] &= (f1(div_access[0], div_access[1]) == 2);

            const dpl::divides<float> f2;
            ret_access[0] &= (f2(18, 4.0f) == 4.5f);
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with dpl::divides");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
