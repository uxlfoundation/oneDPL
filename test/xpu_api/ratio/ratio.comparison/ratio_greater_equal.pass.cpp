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

#include <oneapi/dpl/ratio>

#include "support/test_macros.h"
#include "support/utils.h"

template <class Rat1, class Rat2, bool result, class KernelName>
bool
test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> item1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelName>([=]() {
                static_assert(result == dpl::ratio_greater_equal<Rat1, Rat2>::value);
                ret_acc[0] = true;
            });
        });
    }
    return ret;
}

class T1;
class T2;
class T3;
class T4;
class T5;
class T6;
class T7;
class T8;

bool
kernel_test()
{
    auto ret = true;
    {
        typedef dpl::ratio<1, 1> R1;
        typedef dpl::ratio<1, 1> R2;
        ret &= test<R1, R2, true, T1>();
    }
    {
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
        ret &= test<R1, R2, true, T2>();
    }
    {
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
        ret &= test<R1, R2, true, T3>();
    }
    {
        typedef dpl::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
        typedef dpl::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R2;
        ret &= test<R1, R2, true, T4>();
    }
    {
        typedef dpl::ratio<1, 1> R1;
        typedef dpl::ratio<1, -1> R2;
        ret &= test<R1, R2, true, T5>();
    }
    {
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
        ret &= test<R1, R2, true, T6>();
    }
    {
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
        ret &= test<R1, R2, false, T7>();
    }
    {
        typedef dpl::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
        typedef dpl::ratio<1, -0x7FFFFFFFFFFFFFFFLL> R2;
        ret &= test<R1, R2, true, T8>();
    }

    return ret;
}

int
main()
{
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::ratio_greater_equal check");

    return TestUtils::done();
}
