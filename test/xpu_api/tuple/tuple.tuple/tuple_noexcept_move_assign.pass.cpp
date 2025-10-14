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
#include "support/utils_invoke.h"

struct NoexceptMoveAssignClass
{
};

struct NonNoexceptMoveAssignClass
{
    NonNoexceptMoveAssignClass&
    operator=(NonNoexceptMoveAssignClass&&) noexcept(false);
};

void
kernel_test1(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest1>([=]() {
            typedef dpl::tuple<int> tt1;
            typedef dpl::tuple<int, float> tt2;
            typedef dpl::tuple<short, float, int> tt3;
            typedef dpl::tuple<short, NoexceptMoveAssignClass, float> tt4;
            typedef dpl::tuple<NoexceptMoveAssignClass, NoexceptMoveAssignClass, float> tt5;
            typedef dpl::tuple<NoexceptMoveAssignClass, NoexceptMoveAssignClass, NoexceptMoveAssignClass> tt6;

            static_assert(std::is_nothrow_move_assignable<tt1>::value);
            static_assert(std::is_nothrow_move_assignable<tt2>::value);
            static_assert(std::is_nothrow_move_assignable<tt3>::value);
            static_assert(std::is_nothrow_move_assignable<tt4>::value);
            static_assert(std::is_nothrow_move_assignable<tt5>::value);
            static_assert(std::is_nothrow_move_assignable<tt6>::value);

            typedef dpl::tuple<short, NonNoexceptMoveAssignClass, float> tt4n;
            typedef dpl::tuple<NonNoexceptMoveAssignClass, NonNoexceptMoveAssignClass, float> tt5n;
            typedef dpl::tuple<NonNoexceptMoveAssignClass, NonNoexceptMoveAssignClass, NonNoexceptMoveAssignClass> tt6n;

            static_assert(!std::is_nothrow_move_assignable<tt4n>::value);
            static_assert(!std::is_nothrow_move_assignable<tt5n>::value);
            static_assert(!std::is_nothrow_move_assignable<tt6n>::value);
        });
    });
}

void
kernel_test2(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest2>([=]() {
            typedef dpl::tuple<int, double> tt2;
            typedef dpl::tuple<short, double, int> tt3;
            typedef dpl::tuple<short, NoexceptMoveAssignClass, double> tt4;
            typedef dpl::tuple<NoexceptMoveAssignClass, NoexceptMoveAssignClass, double> tt5;

            typedef dpl::tuple<short, NonNoexceptMoveAssignClass, double> tt4n;
            typedef dpl::tuple<NonNoexceptMoveAssignClass, NonNoexceptMoveAssignClass, double> tt5n;

            static_assert(std::is_nothrow_move_assignable<tt2>::value);
            static_assert(std::is_nothrow_move_assignable<tt3>::value);
            static_assert(std::is_nothrow_move_assignable<tt4>::value);
            static_assert(std::is_nothrow_move_assignable<tt5>::value);

            static_assert(!std::is_nothrow_move_assignable<tt4n>::value);
            static_assert(!std::is_nothrow_move_assignable<tt5n>::value);
        });
    });
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test1(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        kernel_test2(deviceQueue);
    }

    return TestUtils::done();
}
