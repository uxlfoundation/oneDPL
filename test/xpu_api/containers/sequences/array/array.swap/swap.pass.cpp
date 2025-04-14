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

// <array>
// void swap(array& a);
// namespace std { void swap(array<T, N> &x, array<T, N> &y);

// In Windows, as a temporary workaround, disable vector algorithm calls to avoid calls within sycl kernels
#if defined(_MSC_VER)
#    define _USE_STD_VECTOR_ALGORITHMS 0
#endif

#include "support/test_config.h"

#include <oneapi/dpl/array>
#include <oneapi/dpl/utility>

#include "support/utils.h"

class KernelTest1;

bool
kernel_test()
{

    bool ret = true;
    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{1});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto ret_acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelTest1>([=]() {
                {
                    typedef int T;
                    typedef dpl::array<T, 3> C;
                    C c1 = {1, 2, 35};
                    C c2 = {4, 5, 65};
                    c1.swap(c2);
                    ret_acc[0] &= (c1.size() == 3);
                    ret_acc[0] &= (c1[0] == 4);
                    ret_acc[0] &= (c1[1] == 5);
                    ret_acc[0] &= (c1[2] == 65);
                    ret_acc[0] &= (c2.size() == 3);
                    ret_acc[0] &= (c2[0] == 1);
                    ret_acc[0] &= (c2[1] == 2);
                    ret_acc[0] &= (c2[2] == 35);
                }
                {
                    typedef int T;
                    typedef dpl::array<T, 3> C;
                    C c1 = {1, 2, 35};
                    C c2 = {4, 5, 65};
                    dpl::swap(c1, c2);
                    ret_acc[0] &= (c1.size() == 3);
                    ret_acc[0] &= (c1[0] == 4);
                    ret_acc[0] &= (c1[1] == 5);
                    ret_acc[0] &= (c1[2] == 65);
                    ret_acc[0] &= (c2.size() == 3);
                    ret_acc[0] &= (c2[0] == 1);
                    ret_acc[0] &= (c2[1] == 2);
                    ret_acc[0] &= (c2[2] == 35);
                }
                {
                    typedef int T;
                    typedef dpl::array<T, 0> C;
                    C c1 = {};
                    C c2 = {};
                    c1.swap(c2);
                    ret_acc[0] &= (c1.size() == 0);
                    ret_acc[0] &= (c2.size() == 0);
                }
                {
                    typedef int T;
                    typedef dpl::array<T, 0> C;
                    C c1 = {};
                    C c2 = {};
                    dpl::swap(c1, c2);
                    ret_acc[0] &= (c1.size() == 0);
                    ret_acc[0] &= (c2.size() == 0);
                }
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of work with dpl::swap");

    return TestUtils::done();
}
