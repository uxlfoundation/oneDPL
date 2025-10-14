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

// size_t should:
//
//  1. be in namespace std.
//  2. be the same sizeof as void*.
//  3. be an unsigned integral.

#include "support/test_config.h"

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/cstddef>

#include "support/test_macros.h"
#include "support/utils.h"

int
main()
{
    const dpl::size_t N = 1;
    bool ret = true;

    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{N});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                static_assert(sizeof(dpl::size_t) == sizeof(void*), "sizeof(dpl::size_t) == sizeof(void*)");
                static_assert(dpl::is_unsigned<dpl::size_t>::value, "spl::is_unsigned<dpl::size_t>::value");
                static_assert(dpl::is_integral<dpl::size_t>::value, "spl::is_integral<dpl::size_t>::value");
                acc[0] &= (sizeof(dpl::size_t) == sizeof(void*));
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with size_t in Kernel");

    return TestUtils::done();
}
