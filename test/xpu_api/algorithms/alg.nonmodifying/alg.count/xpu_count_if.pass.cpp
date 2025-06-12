//===-- xpu_count_if.pass.cpp ---------------------------------------------===//
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

#include <oneapi/dpl/algorithm>

#include "support/utils.h"
#include "support/test_iterators.h"

#include <cassert>

void
kernel_test(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
                const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                ret_acc[0] &=
                    (dpl::count_if(input_iterator<const int*>(ia), input_iterator<const int*>(ia + sa), TestUtils::IsEqualTo<int>{2}) == 3);
                ret_acc[0] &=
                    (dpl::count_if(input_iterator<const int*>(ia), input_iterator<const int*>(ia + sa), TestUtils::IsEqualTo<int>{7}) == 0);
                ret_acc[0] &=
                    (dpl::count_if(input_iterator<const int*>(ia), input_iterator<const int*>(ia), TestUtils::IsEqualTo<int>{2}) == 0);
            });
        });
    }
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test(deviceQueue);
    return TestUtils::done();
}
