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

#include <oneapi/dpl/utility>

#include "support/utils.h"

struct NoexceptMoveAssignClass
{
};

struct NoexceptMoveConsClass
{
};

struct NoexceptMoveConsNoexceptMoveAssignClass
{
};

struct NonNoexceptCopyAssignClass
{
    NonNoexceptCopyAssignClass(const NonNoexceptCopyAssignClass&) noexcept(false);
    NonNoexceptCopyAssignClass&
    operator==(const NonNoexceptCopyAssignClass&) noexcept(false);
};

struct NonNoexceptMoveAssignClass
{
    NonNoexceptMoveAssignClass(NonNoexceptMoveAssignClass&&) noexcept(false);
    NonNoexceptMoveAssignClass&
    operator==(NonNoexceptMoveAssignClass&&) noexcept(false);
};

template <typename T>
void
test_is_nothrow_swappable()
{
    static_assert(std::is_nothrow_swappable_v<T>);
}

template <typename T>
void
test_is_non_nothrow_swappable()
{
    static_assert(!std::is_nothrow_swappable<T>::value);
    static_assert(!std::is_nothrow_swappable_v<T>);

    static_assert(!std::is_nothrow_swappable_with<T, T>::value);
    static_assert(!std::is_nothrow_swappable_with_v<T, T>);
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
                typedef dpl::pair<short, NoexceptMoveAssignClass> tt3;
                typedef dpl::pair<short, NoexceptMoveConsClass> tt4;
                typedef dpl::pair<NoexceptMoveConsClass, NoexceptMoveConsClass> tt5;
                typedef dpl::pair<NoexceptMoveConsNoexceptMoveAssignClass, NoexceptMoveConsNoexceptMoveAssignClass> tt6;
                typedef dpl::pair<NoexceptMoveConsNoexceptMoveAssignClass, float> tt7;
                typedef dpl::pair<NoexceptMoveConsNoexceptMoveAssignClass, NoexceptMoveConsNoexceptMoveAssignClass> tt8;

                typedef dpl::pair<short, NonNoexceptCopyAssignClass> tt1n1;
                typedef dpl::pair<NonNoexceptCopyAssignClass, NonNoexceptCopyAssignClass> tt1n2;
                typedef dpl::pair<NonNoexceptCopyAssignClass, float> tt1n3;

                typedef dpl::pair<short, NonNoexceptMoveAssignClass> tt2n1;
                typedef dpl::pair<NonNoexceptMoveAssignClass, NonNoexceptMoveAssignClass> tt2n2;
                typedef dpl::pair<NonNoexceptMoveAssignClass, float> tt2n3;

                test_is_nothrow_swappable<tt1&>();
                test_is_nothrow_swappable<tt2&>();
                test_is_nothrow_swappable<tt3&>();
                test_is_nothrow_swappable<tt4&>();
                test_is_nothrow_swappable<tt5&>();
                test_is_nothrow_swappable<tt6&>();
                test_is_nothrow_swappable<tt7&>();
                test_is_nothrow_swappable<tt8&>();

                test_is_non_nothrow_swappable<tt1n1&>();
                test_is_non_nothrow_swappable<tt1n2&>();
                test_is_non_nothrow_swappable<tt1n3&>();

                test_is_non_nothrow_swappable<tt2n1&>();
                test_is_non_nothrow_swappable<tt2n2&>();
                test_is_non_nothrow_swappable<tt2n3&>();
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
