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

// template<class E> class initializer_list;
//
// initializer_list();

#include "support/test_config.h"

#include <oneapi/dpl/cstddef>

#include "support/test_macros.h"
#include "support/utils.h"

#include <initializer_list>

struct A
{
};

int
main()
{
    const dpl::size_t N = 1;
    bool rs[N] = {false};

    {
        sycl::buffer<bool, 1> buf(rs, sycl::range<1>{N});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                std::initializer_list<A> il;
                acc[0] = (il.size() == 0);
            });
        });
    }

    for (dpl::size_t i = 0; i < N; ++i)
    {
        EXPECT_TRUE(rs[i], "Wrong result of work with default initializer list in Kernel");
    }

    return TestUtils::done();
}
