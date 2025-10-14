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

#include "support/utils.h"

struct NoexceptMoveAssignClass
{
};

struct NonNoexceptMoveAssignClass
{
    NonNoexceptMoveAssignClass(NonNoexceptMoveAssignClass&&) noexcept(false);
    NonNoexceptMoveAssignClass&
    operator=(NonNoexceptMoveAssignClass&&) noexcept(false);
};

template <typename T>
void
test_nothrow_move_assignable()
{
    static_assert(std::is_nothrow_move_assignable<T>::value);
    static_assert(std::is_nothrow_move_assignable_v<T>);
}

template <typename T>
void
test_non_nothrow_move_assignable()
{
    static_assert(!std::is_nothrow_move_assignable<T>::value);
    static_assert(!std::is_nothrow_move_assignable_v<T>);
}

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                typedef dpl::pair<int, int> tt1;
                typedef dpl::pair<int, float> tt2;
                typedef dpl::pair<NoexceptMoveAssignClass, NoexceptMoveAssignClass> tt3;
                typedef dpl::pair<NonNoexceptMoveAssignClass, NonNoexceptMoveAssignClass> tt3n;

                test_nothrow_move_assignable<tt1>();
                test_nothrow_move_assignable<tt2>();
                test_nothrow_move_assignable<tt3>();

                test_non_nothrow_move_assignable<tt3n>();
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
