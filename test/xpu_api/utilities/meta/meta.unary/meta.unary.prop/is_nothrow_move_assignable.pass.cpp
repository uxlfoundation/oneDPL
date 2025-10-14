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

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

template <class KernelTest, class T>
void
test_has_nothrow_assign(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(dpl::is_nothrow_move_assignable<T>::value);
            static_assert(dpl::is_nothrow_move_assignable_v<T>);
        });
    });
}

template <class KernelTest, class T>
void
test_has_not_nothrow_assign(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!dpl::is_nothrow_move_assignable<T>::value);
            static_assert(!dpl::is_nothrow_move_assignable_v<T>);
        });
    });
}

class Empty
{
};

union Union
{
};

struct bit_zero
{
    int : 0;
};

struct A
{
    A&
    operator=(const A&);
};

struct ANT
{
    ANT&
    operator=(const ANT&) noexcept;
};

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;
class KernelTest6;
class KernelTest7;
class KernelTest8;
class KernelTest9;
class KernelTest10;
class KernelTest11;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_has_nothrow_assign<KernelTest1, int&>(deviceQueue);
    test_has_nothrow_assign<KernelTest2, Union>(deviceQueue);
    test_has_nothrow_assign<KernelTest3, Empty>(deviceQueue);
    test_has_nothrow_assign<KernelTest4, int>(deviceQueue);
    test_has_nothrow_assign<KernelTest5, int*>(deviceQueue);
    test_has_nothrow_assign<KernelTest6, const int*>(deviceQueue);
    test_has_nothrow_assign<KernelTest7, bit_zero>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_has_nothrow_assign<KernelTest8, double>(deviceQueue);
    }

    test_has_not_nothrow_assign<KernelTest9, void>(deviceQueue);
    test_has_not_nothrow_assign<KernelTest10, A>(deviceQueue);

    test_has_nothrow_assign<KernelTest11, ANT>(deviceQueue);
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
