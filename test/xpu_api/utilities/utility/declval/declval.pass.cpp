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

#include "support/test_macros.h"
#include "support/utils.h"

struct A
{
    A(const A&) = delete;
    A&
    operator=(const A&) = delete;
};

void
kernel_test()
{
    static_assert(dpl::is_same_v<decltype(dpl::declval<A>()), A&&>);
    static_assert(dpl::is_same_v<decltype(dpl::declval<A&>()), A&>);
}

class KernelTest;

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::range<1> numOfItems{1};
    {
        deviceQueue.submit([&](sycl::handler& cgh) { cgh.single_task<class KernelTest>([=]() { kernel_test(); }); });
    }

    return TestUtils::done();
}
