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

#include <oneapi/dpl/tuple>
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

// make_tuple
void
test_make_tuple()
{
    {
        constexpr auto p1 = dpl::make_tuple(22, 22.222f);
        static_assert(dpl::get<0>(p1) == 22);
    }

    {
        constexpr auto p1 = dpl::make_tuple(22, 22.222f, 77799);
        static_assert(dpl::get<0>(p1) == 22);
    }
}

// forward_as_tuple
void
test_forward_as_tuple()
{
    {
        static const int i = 22;
        static const float f = 22.222f;

        typedef dpl::tuple<const int&, const float&&> tuple_type;
        constexpr tuple_type p1 = dpl::forward_as_tuple(i, dpl::move(f));
        static_assert(dpl::get<0>(p1) == i);
    }

    {
        static const int i = 22;
        static const float f = 22.222f;
        static const int ii = 77799;

        typedef dpl::tuple<const int&, const float&, const int&&> tuple_type;
        constexpr tuple_type p1 = dpl::forward_as_tuple(i, f, dpl::move(ii));
        static_assert(dpl::get<0>(p1) == i);
    }
}

// tie
void
test_tie()
{
    {
        static const int i = 22;
        static const float f = 22.222f;

        typedef dpl::tuple<const int&, const float&> tuple_type;
        constexpr tuple_type p1 = dpl::tie(i, f);
        static_assert(dpl::get<0>(p1) == i);
    }

    {
        static const int i = 22;
        static const float f = 22.222f;
        static const int ii = 77799;

        typedef dpl::tuple<const int&, const float&, const int&> tuple_type;
        constexpr tuple_type p1 = dpl::tie(i, f, ii);
        static_assert(dpl::get<0>(p1) == i);
    }
}

// tuple_cat
void
test_tuple_cat()
{
    typedef dpl::tuple<int, float> tuple_type1;
    typedef dpl::tuple<int, int, float> tuple_type2;

    constexpr tuple_type1 t1{55, 77.77f};
    constexpr tuple_type2 t2{55, 99, 77.77f};
    constexpr auto cat1 = dpl::tuple_cat(t1, t2);
    static_assert(std::tuple_size_v<decltype(cat1)> == 5);
    static_assert(dpl::get<0>(cat1) == 55);
}

// ignore
void
test_ignore()
{
    [[maybe_unused]] constexpr auto ign1 = dpl::ignore;
    [[maybe_unused]] constexpr auto ign2 = dpl::make_tuple(dpl::ignore);
}

class KernelTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                test_make_tuple();
                test_forward_as_tuple();
                test_tie();
                test_tuple_cat();
                test_ignore();
            });
        });
    }
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
