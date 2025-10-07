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
void
test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::range<1> item1{1};
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<KernelName>([=]() { static_assert(result == dpl::ratio_less<Rat1, Rat2>::value); });
        });
    }
}

class T1;
class T2;
class T3;
class T4;
class T5;
class T6;
class T7;
class T8;
class T9;
class T10;
class T11;
class T12;
class T13;
class T14;

void
kernel_test()
{
    {
        typedef dpl::ratio<1, 1> R1;
        typedef dpl::ratio<1, 1> R2;
        test<R1, R2, false, T1>();
    }
    {
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
        test<R1, R2, false, T2>();
    }
    {
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
        test<R1, R2, false, T3>();
    }
    {
        typedef dpl::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
        typedef dpl::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R2;
        test<R1, R2, false, T4>();
    }
    {
        typedef dpl::ratio<1, 1> R1;
        typedef dpl::ratio<1, -1> R2;
        test<R1, R2, false, T5>();
    }
    {
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
        test<R1, R2, false, T6>();
    }
    {
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
        test<R1, R2, true, T7>();
    }
    {
        typedef dpl::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
        typedef dpl::ratio<1, -0x7FFFFFFFFFFFFFFFLL> R2;
        test<R1, R2, false, T8>();
    }
    {
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R1;
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R2;
        test<R1, R2, true, T9>();
    }
    {
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R1;
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R2;
        test<R1, R2, false, T10>();
    }
    {
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R1;
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R2;
        test<R1, R2, true, T11>();
    }
    {
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R1;
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFELL, 0x7FFFFFFFFFFFFFFDLL> R2;
        test<R1, R2, true, T12>();
    }
    {
        typedef dpl::ratio<641981, 1339063> R1;
        typedef dpl::ratio<1291640, 2694141LL> R2;
        test<R1, R2, false, T13>();
    }
    {
        typedef dpl::ratio<1291640, 2694141LL> R1;
        typedef dpl::ratio<641981, 1339063> R2;
        test<R1, R2, true, T14>();
    }
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
