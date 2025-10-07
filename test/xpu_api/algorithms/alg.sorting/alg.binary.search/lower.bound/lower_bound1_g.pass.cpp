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

#include <oneapi/dpl/algorithm>

#include <iostream>

#include "support/utils.h"

using dpl::lower_bound;

bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    bool check = false;
    int array[] = {0, 0, 0, 0, 1, 1, 1, 1};
    auto tmp = array;
    const int N = sizeof(array) / sizeof(array[0]);
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};

    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<bool, 1> buffer2(&check, item1);
        sycl::buffer<int, 1> buffer3(array, itemN);
        deviceQueue
            .submit([&](sycl::handler& cgh) {
                auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
                auto check_access = buffer2.get_access<sycl::access::mode::write>(cgh);
                auto access = buffer3.get_access<sycl::access::mode::write>(cgh);
                cgh.single_task<class KernelTest1>([=]() {
                    int arr[] = {0, 0, 0, 0, 1, 1, 1, 1};
                    // check if there is change after data transfer
                    check_access[0] = TestUtils::check_data(&access[0], arr, N);
                    if (check_access[0])
                    {
                        auto ret = true;
                        for (int i = 0; i < 5; ++i)
                        {
                            for (int j = 4; j < 7; ++j)
                            {
                                ret &= (lower_bound(&access[0] + i, &access[0] + j, 1) == &access[0] + 4);
                            }
                        }
                        ret_access[0] = ret;
                    }
                });
            })
            .wait();
    }
    // check if there is change after executing kernel function
    check &= TestUtils::check_data(tmp, array, N);
    if (!check)
        return false;
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of lower_bound in kernel_test");

    return TestUtils::done();
}
