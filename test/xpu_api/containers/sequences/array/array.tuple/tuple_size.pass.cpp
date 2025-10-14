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

// <array>
// tuple_size<array<T, N> >::value

#include "support/test_config.h"

#include <oneapi/dpl/array>

#include "support/utils.h"

template <class T, std::size_t N>
void
test()
{
    {
        typedef dpl::array<T, N> C;
        static_assert(dpl::tuple_size<C>::value == N);
    }
    {
        typedef dpl::array<T const, N> C;
        static_assert(dpl::tuple_size<C>::value == N);
    }
    {
        typedef dpl::array<T volatile, N> C;
        static_assert(dpl::tuple_size<C>::value == N);
    }
    {
        typedef dpl::array<T const volatile, N> C;
        static_assert(dpl::tuple_size<C>::value == N);
    }
}

int
main()
{
    bool ret = false;
    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{1});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto ret_acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                test<float, 0>();
                test<float, 3>();
                test<float, 5>();
                ret_acc[0] = true;
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with dpl::tuple_size");

    return TestUtils::done();
}
