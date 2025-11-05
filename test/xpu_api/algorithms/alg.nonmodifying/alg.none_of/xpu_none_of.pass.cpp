//===-- xpu_none_of.pass.cpp ----------------------------------------------===//
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

using oneapi::dpl::none_of;

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
                {
                    int ia[] = {2, 4, 6, 8};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        !none_of(input_iterator<const int*>(ia), input_iterator<const int*>(ia + sa), TestUtils::IsEven<int>());
                    ret_acc[0] &= none_of(input_iterator<const int*>(ia), input_iterator<const int*>(ia), TestUtils::IsEven<int>());
                }
                {
                    const int ia[] = {2, 4, 5, 8};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        !none_of(input_iterator<const int*>(ia), input_iterator<const int*>(ia + sa), TestUtils::IsEven<int>());
                    ret_acc[0] &= none_of(input_iterator<const int*>(ia), input_iterator<const int*>(ia), TestUtils::IsEven<int>());
                }
                {
                    const int ia[] = {1, 3, 5, 7};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &= none_of(input_iterator<const int*>(ia), input_iterator<const int*>(ia + sa), TestUtils::IsEven<int>());
                    ret_acc[0] &= none_of(input_iterator<const int*>(ia), input_iterator<const int*>(ia), TestUtils::IsEven<int>());
                }
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
