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


void
test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::range<1> item1{1};
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                {
                    typedef dpl::ratio<1, 1> R1;
                    typedef dpl::ratio<1, 1> R2;
                    typedef dpl::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == 1 && R::den == 1);
                }
                {
                    typedef dpl::ratio<1, 2> R1;
                    typedef dpl::ratio<1, 1> R2;
                    typedef dpl::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == 1 && R::den == 2);
                }
                {
                    typedef dpl::ratio<-1, 2> R1;
                    typedef dpl::ratio<1, 1> R2;
                    typedef dpl::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == -1 && R::den == 2);
                }
                {
                    typedef dpl::ratio<1, -2> R1;
                    typedef dpl::ratio<1, 1> R2;
                    typedef dpl::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == -1 && R::den == 2);
                }
                {
                    typedef dpl::ratio<1, 2> R1;
                    typedef dpl::ratio<-1, 1> R2;
                    typedef dpl::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == -1 && R::den == 2);
                }
                {
                    typedef dpl::ratio<1, 2> R1;
                    typedef dpl::ratio<1, -1> R2;
                    typedef dpl::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == -1 && R::den == 2);
                }
                {
                    typedef dpl::ratio<56987354, 467584654> R1;
                    typedef dpl::ratio<544668, 22145> R2;
                    typedef dpl::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == 15519594064236LL && R::den == 5177331081415LL);
                }
            });
        });
    }
}


int
main()
{
    test();


    return TestUtils::done();
}
