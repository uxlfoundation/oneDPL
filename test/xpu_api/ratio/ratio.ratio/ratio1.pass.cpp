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

template <long long N, long long D, long long eN, long long eD, class KernelName>
void
test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::range<1> item1{1};
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<KernelName>([=]() {
                static_assert(dpl::ratio<N, D>::num == eN);
                static_assert(dpl::ratio<N, D>::den == eD);
            });
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
class T15;
class T16;
class T17;
class T18;
class T19;
class T20;

int
main()
{
    test<1, 1, 1, 1, T1>();
    test<1, 10, 1, 10, T2>();
    test<10, 10, 1, 1, T3>();
    test<10, 1, 10, 1, T4>();
    test<12, 4, 3, 1, T5>();
    test<12, -4, -3, 1, T6>();
    test<-12, 4, -3, 1, T7>();
    test<-12, -4, 3, 1, T8>();
    test<4, 12, 1, 3, T9>();
    test<4, -12, -1, 3, T10>();
    test<-4, 12, -1, 3, T11>();
    test<-4, -12, 1, 3, T12>();
    test<222, 333, 2, 3, T13>();
    test<222, -333, -2, 3, T14>();
    test<-222, 333, -2, 3, T15>();
    test<-222, -333, 2, 3, T16>();
    test<0x7FFFFFFFFFFFFFFFLL, 127, 72624976668147841LL, 1, T17>();
    test<-0x7FFFFFFFFFFFFFFFLL, 127, -72624976668147841LL, 1, T18>();
    test<0x7FFFFFFFFFFFFFFFLL, -127, -72624976668147841LL, 1, T19>();
    test<-0x7FFFFFFFFFFFFFFFLL, -127, 72624976668147841LL, 1, T20>();

    return TestUtils::done();
}
