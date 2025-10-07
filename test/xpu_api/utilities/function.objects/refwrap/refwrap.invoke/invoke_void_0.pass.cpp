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
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

struct A_void_0
{
    int* count_A;
    A_void_0(int* count) { count_A = count; }
    void
    operator()()
    {
        ++(*count_A);
    }
};

class KernelInvokeVoid0Test;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelInvokeVoid0Test>([=]() {
            int count = 0;
            int save_count = count;
            {
                A_void_0 a0(&count);
                dpl::reference_wrapper<A_void_0> r1(a0);
                r1();
                ret_access[0] = (count == save_count + 1);
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with invoke (void)");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
