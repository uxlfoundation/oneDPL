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

class KernelTypeCtorTest;

class functor1
{
};

template <class T>
bool
test(T& t)
{
    dpl::reference_wrapper<T> r(t);
    return (&r.get() == &t);
}

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTypeCtorTest>([=]() {
                functor1 f1;
                ret_access[0] = test(f1);

                int i = 0;
                ret_access[0] &= test(i);
                const int j = 0;
                ret_access[0] &= test(j);
            });
        });
    }

    EXPECT_TRUE(ret, "Error in work with type constructor");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
