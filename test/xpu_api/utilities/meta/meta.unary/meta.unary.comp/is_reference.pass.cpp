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
#include <oneapi/dpl/cstddef> // for dpl::nullptr_t

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

template <class KernelTest, class T>
void
test_is_reference(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(dpl::is_reference<T>::value);
            static_assert(dpl::is_reference<const T>::value);
            static_assert(dpl::is_reference<volatile T>::value);
            static_assert(dpl::is_reference<const volatile T>::value);

            static_assert(dpl::is_reference_v<T>);
            static_assert(dpl::is_reference_v<const T>);
            static_assert(dpl::is_reference_v<volatile T>);
            static_assert(dpl::is_reference_v<const volatile T>);
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_reference(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!dpl::is_reference<T>::value);
            static_assert(!dpl::is_reference<const T>::value);
            static_assert(!dpl::is_reference<volatile T>::value);
            static_assert(!dpl::is_reference<const volatile T>::value);

            static_assert(!dpl::is_reference_v<T>);
            static_assert(!dpl::is_reference_v<const T>);
            static_assert(!dpl::is_reference_v<volatile T>);
            static_assert(!dpl::is_reference_v<const volatile T>);
        });
    });
}

class incomplete_type;

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

enum Enum
{
    zero,
    one
};

typedef void (*FunctionPtr)();

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
class KernelTest12;
class KernelTest13;
class KernelTest14;
class KernelTest15;
class KernelTest16;
class KernelTest17;
class KernelTest18;
class KernelTest19;
class KernelTest20;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_reference<KernelTest1, int&>(deviceQueue);
    test_is_reference<KernelTest2, int&&>(deviceQueue);

    test_is_not_reference<KernelTest3, dpl::nullptr_t>(deviceQueue);
    test_is_not_reference<KernelTest4, void>(deviceQueue);
    test_is_not_reference<KernelTest5, int>(deviceQueue);
    test_is_not_reference<KernelTest6, char[3]>(deviceQueue);
    test_is_not_reference<KernelTest7, char[]>(deviceQueue);
    test_is_not_reference<KernelTest8, void*>(deviceQueue);
    test_is_not_reference<KernelTest9, FunctionPtr>(deviceQueue);
    test_is_not_reference<KernelTest10, Union>(deviceQueue);
    test_is_not_reference<KernelTest11, incomplete_type>(deviceQueue);
    test_is_not_reference<KernelTest12, Empty>(deviceQueue);
    test_is_not_reference<KernelTest13, bit_zero>(deviceQueue);
    test_is_not_reference<KernelTest14, int*>(deviceQueue);
    test_is_not_reference<KernelTest15, const int*>(deviceQueue);
    test_is_not_reference<KernelTest16, Enum>(deviceQueue);
    test_is_not_reference<KernelTest17, int(int)>(deviceQueue);
    test_is_not_reference<KernelTest18, int Empty::*>(deviceQueue);
    test_is_not_reference<KernelTest19, void (Empty::*)(int)>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_not_reference<KernelTest20, double>(deviceQueue);
    }
}

int
main()
{
    kernel_test();

    return 0;
}
