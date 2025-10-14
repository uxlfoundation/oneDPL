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

using dpl::binary_search;

bool
kernel_test1()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    int array[] = {0};
    bool ret = false;
    bool transferCheck = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        sycl::buffer<bool, 1> buffer2(&transferCheck, numOfItems);
        sycl::buffer<int, 1> buffer3(array, numOfItems);
        deviceQueue
            .submit([&](sycl::handler& cgh) {
                auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
                auto check_access = buffer2.get_access<sycl::access::mode::write>(cgh);
                auto acc_arr = buffer3.get_access<sycl::access::mode::write>(cgh);
                cgh.single_task<class KernelTest1>([=]() {
                    // check if there is change after data transfer
                    check_access[0] = (acc_arr[0] == 0);
                    if (check_access[0])
                    {
                        ret_access[0] = (!binary_search(&acc_arr[0], &acc_arr[0], 1));
                    }
                });
            })
            .wait();
    }
    // check if there is change after executing kernel function
    transferCheck &= (array[0] == 0);
    if (!transferCheck)
        return false;
    return ret;
}

bool
kernel_test2()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    int array[] = {0, 2, 4, 6, 8};
    int tmp[] = {0, 2, 4, 6, 8};
    const int N = sizeof(array) / sizeof(array[0]);
    bool ret = false;
    bool check = false;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<int, 1> buffer1(array, itemN);
        sycl::buffer<bool, 1> buffer2(&ret, item1);
        sycl::buffer<bool, 1> buffer3(&check, item1);
        deviceQueue
            .submit([&](sycl::handler& cgh) {
                auto access1 = buffer1.get_access<sycl::access::mode::write>(cgh);
                auto ret_access = buffer2.get_access<sycl::access::mode::write>(cgh);
                auto check_access = buffer3.get_access<sycl::access::mode::write>(cgh);
                cgh.single_task<class KernelTest2>([=]() {
                    int tmp[] = {0, 2, 4, 6, 8};
                    // check if there is change after data transfer
                    check_access[0] = TestUtils::check_data(&access1[0], tmp, N);
                    if (check_access[0])
                    {
                        ret_access[0] = (binary_search(&access1[0], &access1[0] + N, 0));

                        for (int i = 2; i < 10; i += 2)
                        {
                            ret_access[0] &= (binary_search(&access1[0], &access1[0] + N, i));
                        }

                        for (int i = -1; i < 11; i += 2)
                        {
                            ret_access[0] &= (!binary_search(&access1[0], &access1[0] + N, i));
                        }
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
    EXPECT_TRUE(ret, "Wrong result of binary_search in kernel_test1 or kernel_test2");

    return TestUtils::done();
}
