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

//
// <initializer_list>
//
// template<class E> const E* begin(initializer_list<E> il);

#include "support/test_config.h"

#include <oneapi/dpl/cstddef>

#include "support/test_macros.h"
#include "support/utils.h"

#include <initializer_list>

struct A
{
    A(std::initializer_list<int> il)
    {
        const int* b = begin(il);
        const int* e = end(il);
        size = il.size();
        int i = 0;
        while (b < e)
        {
            data[i++] = *b++;
        }
    }

    dpl::size_t size;
    int data[10];
};

int
main()
{
    const dpl::size_t N = 4;
    bool rs[N] = {false};
    {
        sycl::buffer<bool, 1> buf(rs, sycl::range<1>{N});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                A test1 = {3, 2, 1};
                acc[0] = (test1.size == 3);
                acc[1] = (test1.data[0] == 3);
                acc[2] = (test1.data[1] == 2);
                acc[3] = (test1.data[2] == 1);
            });
        });
    }

    for (dpl::size_t i = 0; i < N; ++i)
    {
        EXPECT_TRUE(rs[i], "Wrong result of work with begin / end in Kernel");
    }

    return TestUtils::done();
}
