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

// In Windows, as a temporary workaround, disable vector algorithm calls to avoid calls within sycl kernels
#if defined(_MSC_VER)
#    define _USE_STD_VECTOR_ALGORITHMS 0
#endif

#include "support/test_config.h"

#include <oneapi/dpl/array>

#include "support/utils.h"

#if !_PSTL_TEST_COMPARISON_BROKEN
bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                const size_t len = 5;
                typedef dpl::array<int, len> array_type;
                array_type a = {{0, 1, 2, 3, 4}};
                array_type b = {{0, 1, 2, 3, 4}};
                array_type c = {{0, 1, 2, 3, 7}};
                ret_access[0] = (!(a > b));
                ret_access[0] &= (c > a);
            });
        });
    }
    return ret;
}
#endif // !_PSTL_TEST_COMPARISON_BROKEN

int
main()
{
    bool bProcessed = false;

#if !_PSTL_TEST_COMPARISON_BROKEN
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of work with dpl::array and '>' in kernel_test");
    bProcessed = true;
#endif // !_PSTL_TEST_COMPARISON_BROKEN

    return TestUtils::done(bProcessed);
}
