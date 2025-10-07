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
#include "support/utils_invoke.h"

class KernelPlusTest1;
class KernelPlusTest2;

void
kernel_test1(sycl::queue& deviceQueue)
{
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelPlusTest1>([=]() {
            const dpl::plus<int> f1;
            ret_access[0] = (f1(3, 7) == 10);

            const dpl::plus<float> f2;
            ret_access[0] &= (f2(3, 2.5f) == 5.5f);
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with dpl::plus(kernel_test1)");
}

void
kernel_test2(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelPlusTest2>([=]() {
            const dpl::plus<double> f3;
            ret_access[0] &= (f3(3.4, 2.5) == 5.9);
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with dpl::plus (kernel_test2)");
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test1(deviceQueue);

    const auto device = deviceQueue.get_device();
    if (TestUtils::has_type_support<double>(device))
    {
        kernel_test2(deviceQueue);
    }

    return TestUtils::done();
}
