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
#include <oneapi/dpl/utility>

#include <iostream>

#include "support/utils.h"

using dpl::equal_range;

bool
kernel_test1()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    int array[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    const int N = sizeof(array) / sizeof(array[0]);
    auto tmp = array;
    bool ret = false;
    bool check = false;
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
                    auto ret = true;
                    int arr[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
                    // check if there is change after data transfer
                    check_access[0] = TestUtils::check_data(&access[0], arr, N);
                    if (check_access[0])
                    {
                        for (int i = 0; i < 6; ++i)
                        {
                            for (int j = 6; j < 12; ++j)
                            {
                                ret &= (equal_range(&access[0] + i, &access[0] + j, 1).first ==
                                        &access[0] + std::max(i, 4));
                                ret &= (equal_range(&access[0] + i, &access[0] + j, 1).second ==
                                        &access[0] + std::min(j, 8));
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

bool
kernel_test2()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    int array[] = {0, 0, 2, 2, 2};
    const int N = sizeof(array) / sizeof(array[0]);
    auto tmp = array;
    bool ret = false;
    bool check = false;
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
                cgh.single_task<class KernelTest2>([=]() {
                    int arr[] = {0, 0, 2, 2, 2};
                    // check if there is change after data transfer
                    check_access[0] = TestUtils::check_data(&access[0], arr, N);
                    if (check_access[0])
                    {
                        ret_access[0] = (equal_range(&access[0], &access[0] + 5, 1).first == &access[0] + 2);
                        ret_access[0] &= (equal_range(&access[0], &access[0] + 5, 1).second == &access[0] + 2);
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
    auto ret = kernel_test1();
    ret &= kernel_test2();
    EXPECT_TRUE(ret, "Wrong result of equal_range in kernel_test1 or in kernel_test2");

    return TestUtils::done();
}
