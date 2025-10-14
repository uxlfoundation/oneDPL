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

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

struct NonAssignable
{
    NonAssignable&
    operator=(NonAssignable const&) = delete;
    NonAssignable&
    operator=(NonAssignable&&) = delete;
};
struct CopyAssignable
{
    CopyAssignable() = default;
    CopyAssignable(CopyAssignable const&) = default;
    CopyAssignable&
    operator=(CopyAssignable const&) = default;
};
struct MoveAssignable
{
    MoveAssignable() = default;
    MoveAssignable&
    operator=(MoveAssignable const&) = delete;
    MoveAssignable&
    operator=(MoveAssignable&&) = default;
};

class KernelPairTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelPairTest>([=]() {
            {
                typedef dpl::pair<CopyAssignable, short> P;
                const P p1(CopyAssignable(), 4);
                P p2;
                p2 = p1;
                ret_access[0] = (p2.second == 4);
            }

            {
                using P = dpl::pair<int, NonAssignable>;
                static_assert(!dpl::is_copy_assignable<P>::value);
            }

            {
                using P = dpl::pair<int, MoveAssignable>;
                static_assert(!dpl::is_copy_assignable<P>::value);
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::pair assign check");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
