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
test_is_nothrow_copy_constructible(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(dpl::is_nothrow_copy_constructible<T>::value);
            static_assert(dpl::is_nothrow_copy_constructible<const T>::value);
            static_assert(dpl::is_nothrow_copy_constructible_v<T>);
            static_assert(dpl::is_nothrow_copy_constructible_v<const T>);
        });
    });
}

template <class KernelTest, class T>
void
test_has_not_nothrow_copy_constructor(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!dpl::is_nothrow_copy_constructible<T>::value);
            static_assert(!dpl::is_nothrow_copy_constructible<const T>::value);
            static_assert(!dpl::is_nothrow_copy_constructible<volatile T>::value);
            static_assert(!dpl::is_nothrow_copy_constructible<const volatile T>::value);
            static_assert(!dpl::is_nothrow_copy_constructible_v<T>);
            static_assert(!dpl::is_nothrow_copy_constructible_v<const T>);
            static_assert(!dpl::is_nothrow_copy_constructible_v<volatile T>);
            static_assert(!dpl::is_nothrow_copy_constructible_v<const volatile T>);
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
    A(const A&);
};

struct ANT
{
    ANT(const ANT&) noexcept;
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
    test_has_not_nothrow_copy_constructor<KernelTest1, void>(deviceQueue);
    test_has_not_nothrow_copy_constructor<KernelTest2, A>(deviceQueue);

    test_is_nothrow_copy_constructible<KernelTest3, int&>(deviceQueue);
    test_is_nothrow_copy_constructible<KernelTest4, Union>(deviceQueue);
    test_is_nothrow_copy_constructible<KernelTest5, Empty>(deviceQueue);
    test_is_nothrow_copy_constructible<KernelTest6, int>(deviceQueue);
    test_is_nothrow_copy_constructible<KernelTest7, int*>(deviceQueue);
    test_is_nothrow_copy_constructible<KernelTest8, const int*>(deviceQueue);
    test_is_nothrow_copy_constructible<KernelTest9, bit_zero>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_nothrow_copy_constructible<KernelTest10, double>(deviceQueue);
    }
    test_is_nothrow_copy_constructible<KernelTest11, ANT>(deviceQueue);
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
