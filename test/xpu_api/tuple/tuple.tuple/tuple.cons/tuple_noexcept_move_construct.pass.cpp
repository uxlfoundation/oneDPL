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

#include "support/utils.h"

struct NoexceptMoveConsClass
{
};

struct NonNoexceptMoveConsClass
{
    NonNoexceptMoveConsClass&
    operator=(NonNoexceptMoveConsClass&&) noexcept(false);
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                typedef dpl::tuple<int> tt1;
                typedef dpl::tuple<int, float> tt2;
                typedef dpl::tuple<short, float, int> tt3;
                typedef dpl::tuple<short, NoexceptMoveConsClass, float> tt4;
                typedef dpl::tuple<NoexceptMoveConsClass, NoexceptMoveConsClass, float> tt5;
                typedef dpl::tuple<NoexceptMoveConsClass, NoexceptMoveConsClass, NoexceptMoveConsClass> tt6;

                typedef dpl::tuple<short, NonNoexceptMoveConsClass, float> tt4n;
                typedef dpl::tuple<NonNoexceptMoveConsClass, NonNoexceptMoveConsClass, float> tt5n;
                typedef dpl::tuple<NonNoexceptMoveConsClass, NonNoexceptMoveConsClass, NonNoexceptMoveConsClass> tt6n;

                static_assert(std::is_nothrow_move_constructible<tt1>::value);
                static_assert(std::is_nothrow_move_constructible<tt2>::value);
                static_assert(std::is_nothrow_move_constructible<tt3>::value);
                static_assert(std::is_nothrow_move_constructible<tt4>::value);
                static_assert(std::is_nothrow_move_constructible<tt5>::value);
                static_assert(std::is_nothrow_move_constructible<tt6>::value);

                static_assert(!std::is_nothrow_move_constructible<tt4n>::value);
                static_assert(!std::is_nothrow_move_constructible<tt5n>::value);
                static_assert(!std::is_nothrow_move_constructible<tt6n>::value);
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
