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
#include <oneapi/dpl/cstddef>

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

template <class KernelTest, class T>
void
test_is_arithmetic(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(dpl::is_arithmetic<T>::value);
            static_assert(dpl::is_arithmetic<const T>::value);
            static_assert(dpl::is_arithmetic<volatile T>::value);
            static_assert(dpl::is_arithmetic<const volatile T>::value);

            static_assert(dpl::is_arithmetic_v<T>);
            static_assert(dpl::is_arithmetic_v<const T>);
            static_assert(dpl::is_arithmetic_v<volatile T>);
            static_assert(dpl::is_arithmetic_v<const volatile T>);
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_arithmetic(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!dpl::is_arithmetic<T>::value);
            static_assert(!dpl::is_arithmetic<const T>::value);
            static_assert(!dpl::is_arithmetic<volatile T>::value);
            static_assert(!dpl::is_arithmetic<const volatile T>::value);

            static_assert(!dpl::is_arithmetic_v<T>);
            static_assert(!dpl::is_arithmetic_v<const T>);
            static_assert(!dpl::is_arithmetic_v<volatile T>);
            static_assert(!dpl::is_arithmetic_v<const volatile T>);
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
class KernelTest21;
class KernelTest22;
class KernelTest23;
class KernelTest24;
class KernelTest25;
class KernelTest26;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_arithmetic<KernelTest1, short>(deviceQueue);
    test_is_arithmetic<KernelTest2, unsigned short>(deviceQueue);
    test_is_arithmetic<KernelTest3, int>(deviceQueue);
    test_is_arithmetic<KernelTest4, unsigned int>(deviceQueue);
    test_is_arithmetic<KernelTest5, long>(deviceQueue);
    test_is_arithmetic<KernelTest6, unsigned long>(deviceQueue);
    test_is_arithmetic<KernelTest7, bool>(deviceQueue);
    test_is_arithmetic<KernelTest8, char>(deviceQueue);
    test_is_arithmetic<KernelTest9, signed char>(deviceQueue);
    test_is_arithmetic<KernelTest10, unsigned char>(deviceQueue);
    test_is_arithmetic<KernelTest11, wchar_t>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_arithmetic<KernelTest12, double>(deviceQueue);
    }

    test_is_not_arithmetic<KernelTest13, dpl::nullptr_t>(deviceQueue);
    test_is_not_arithmetic<KernelTest14, void>(deviceQueue);
    test_is_not_arithmetic<KernelTest15, int&>(deviceQueue);
    test_is_not_arithmetic<KernelTest16, int&&>(deviceQueue);
    test_is_not_arithmetic<KernelTest17, int*>(deviceQueue);
    test_is_not_arithmetic<KernelTest18, const int*>(deviceQueue);
    test_is_not_arithmetic<KernelTest19, char[3]>(deviceQueue);
    test_is_not_arithmetic<KernelTest20, char[]>(deviceQueue);
    test_is_not_arithmetic<KernelTest21, Union>(deviceQueue);
    test_is_not_arithmetic<KernelTest22, Enum>(deviceQueue);
    test_is_not_arithmetic<KernelTest23, FunctionPtr>(deviceQueue);
    test_is_not_arithmetic<KernelTest24, Empty>(deviceQueue);
    test_is_not_arithmetic<KernelTest25, incomplete_type>(deviceQueue);
    test_is_not_arithmetic<KernelTest26, bit_zero>(deviceQueue);
}

int
main()
{
    kernel_test();

    return 0;
}
