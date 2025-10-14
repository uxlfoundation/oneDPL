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
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

struct NoexceptMoveAssignClass
{
};

struct NonNoexceptMoveAssignClass
{
    NonNoexceptMoveAssignClass&
    operator=(NonNoexceptMoveAssignClass&&) noexcept(false);
};

struct NoexceptMoveConsClass
{
};

struct NonNoexceptMoveConsClass
{
    NonNoexceptMoveConsClass&
    operator=(NonNoexceptMoveConsClass&&) noexcept(false);
};

struct NoexceptMoveConsNoexceptMoveAssignClass
{
};

struct NonNoexceptMoveConsNoexceptMoveAssignClass
{
    NonNoexceptMoveConsNoexceptMoveAssignClass(const NonNoexceptMoveConsNoexceptMoveAssignClass&&) noexcept(false);

    NonNoexceptMoveConsNoexceptMoveAssignClass&
    operator=(const NonNoexceptMoveConsNoexceptMoveAssignClass&&) noexcept(false);
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
                typedef dpl::tuple<short, NoexceptMoveAssignClass, float> tt4;
                typedef dpl::tuple<short, NonNoexceptMoveAssignClass, float> tt4n;
                typedef dpl::tuple<short, NoexceptMoveConsClass, float> tt5;
                typedef dpl::tuple<short, NonNoexceptMoveConsClass, float> tt5n;
                typedef dpl::tuple<NoexceptMoveConsClass> tt6;
                typedef dpl::tuple<NonNoexceptMoveConsClass> tt6n;
                typedef dpl::tuple<NoexceptMoveConsNoexceptMoveAssignClass> tt7;
                typedef dpl::tuple<NonNoexceptMoveConsNoexceptMoveAssignClass> tt7n;
                typedef dpl::tuple<NoexceptMoveConsNoexceptMoveAssignClass, float> tt8;
                typedef dpl::tuple<NonNoexceptMoveConsNoexceptMoveAssignClass, float> tt8n;
                typedef dpl::tuple<float, NoexceptMoveConsNoexceptMoveAssignClass, short> tt9;
                typedef dpl::tuple<float, NonNoexceptMoveConsNoexceptMoveAssignClass, short> tt9n;
                typedef dpl::tuple<NoexceptMoveConsNoexceptMoveAssignClass, NoexceptMoveConsNoexceptMoveAssignClass,
                                   char>
                    tt10;
                typedef dpl::tuple<NonNoexceptMoveConsNoexceptMoveAssignClass,
                                   NonNoexceptMoveConsNoexceptMoveAssignClass, char>
                    tt10n;
                typedef dpl::tuple<NoexceptMoveConsNoexceptMoveAssignClass, NoexceptMoveConsNoexceptMoveAssignClass,
                                   NoexceptMoveConsNoexceptMoveAssignClass>
                    tt11;
                typedef dpl::tuple<NonNoexceptMoveConsNoexceptMoveAssignClass,
                                   NonNoexceptMoveConsNoexceptMoveAssignClass,
                                   NonNoexceptMoveConsNoexceptMoveAssignClass>
                    tt11n;

                static_assert(std::is_nothrow_swappable_v<tt1&>);
                static_assert(std::is_nothrow_swappable_v<tt2&>);
                static_assert(std::is_nothrow_swappable_v<tt3&>);

                static_assert(std::is_nothrow_swappable_v<tt4&>);
                static_assert(std::is_nothrow_swappable_v<tt5&>);
                static_assert(std::is_nothrow_swappable_v<tt6&>);
                static_assert(std::is_nothrow_swappable_v<tt7&>);
                static_assert(std::is_nothrow_swappable_v<tt8&>);
                static_assert(std::is_nothrow_swappable_v<tt9&>);
                static_assert(std::is_nothrow_swappable_v<tt10&>);
                static_assert(std::is_nothrow_swappable_v<tt11&>);

                static_assert(!std::is_nothrow_swappable_v<tt4n&>);
                static_assert(!std::is_nothrow_swappable_v<tt5n&>);
                static_assert(!std::is_nothrow_swappable_v<tt6n&>);
                static_assert(!std::is_nothrow_swappable_v<tt7n&>);
                static_assert(!std::is_nothrow_swappable_v<tt8n&>);
                static_assert(!std::is_nothrow_swappable_v<tt9n&>);
                static_assert(!std::is_nothrow_swappable_v<tt10n&>);
                static_assert(!std::is_nothrow_swappable_v<tt11n&>);
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
