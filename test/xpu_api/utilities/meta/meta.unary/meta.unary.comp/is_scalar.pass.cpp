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
test_is_scalar(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(dpl::is_scalar<T>::value);
            static_assert(dpl::is_scalar<const T>::value);
            static_assert(dpl::is_scalar<volatile T>::value);
            static_assert(dpl::is_scalar<const volatile T>::value);

            static_assert(dpl::is_scalar_v<T>);
            static_assert(dpl::is_scalar_v<const T>);
            static_assert(dpl::is_scalar_v<volatile T>);
            static_assert(dpl::is_scalar_v<const volatile T>);
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_scalar(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!dpl::is_scalar<T>::value);
            static_assert(!dpl::is_scalar<const T>::value);
            static_assert(!dpl::is_scalar<volatile T>::value);
            static_assert(!dpl::is_scalar<const volatile T>::value);

            static_assert(!dpl::is_scalar_v<T>);
            static_assert(!dpl::is_scalar_v<const T>);
            static_assert(!dpl::is_scalar_v<volatile T>);
            static_assert(!dpl::is_scalar_v<const volatile T>);
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
class KernelTest27;
class KernelTest28;
class KernelTest29;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_scalar<KernelTest1, dpl::nullptr_t>(deviceQueue);
    test_is_scalar<KernelTest2, short>(deviceQueue);
    test_is_scalar<KernelTest3, unsigned short>(deviceQueue);
    test_is_scalar<KernelTest4, int>(deviceQueue);
    test_is_scalar<KernelTest5, unsigned int>(deviceQueue);
    test_is_scalar<KernelTest6, long>(deviceQueue);
    test_is_scalar<KernelTest7, unsigned long>(deviceQueue);
    test_is_scalar<KernelTest8, bool>(deviceQueue);
    test_is_scalar<KernelTest9, char>(deviceQueue);
    test_is_scalar<KernelTest10, signed char>(deviceQueue);
    test_is_scalar<KernelTest11, unsigned char>(deviceQueue);
    test_is_scalar<KernelTest12, wchar_t>(deviceQueue);
    test_is_scalar<KernelTest13, int*>(deviceQueue);
    test_is_scalar<KernelTest14, const int*>(deviceQueue);
    test_is_scalar<KernelTest15, int Empty::*>(deviceQueue);
    test_is_scalar<KernelTest16, void (Empty::*)(int)>(deviceQueue);
    test_is_scalar<KernelTest17, Enum>(deviceQueue);
    test_is_scalar<KernelTest18, FunctionPtr>(deviceQueue);

    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_scalar<KernelTest19, double>(deviceQueue);
    }
    test_is_not_scalar<KernelTest20, void>(deviceQueue);
    test_is_not_scalar<KernelTest21, int&>(deviceQueue);
    test_is_not_scalar<KernelTest22, int&&>(deviceQueue);
    test_is_not_scalar<KernelTest23, char[3]>(deviceQueue);
    test_is_not_scalar<KernelTest24, char[]>(deviceQueue);
    test_is_not_scalar<KernelTest25, Union>(deviceQueue);
    test_is_not_scalar<KernelTest26, Empty>(deviceQueue);
    test_is_not_scalar<KernelTest27, incomplete_type>(deviceQueue);
    test_is_not_scalar<KernelTest28, bit_zero>(deviceQueue);
    test_is_not_scalar<KernelTest29, int(int)>(deviceQueue);
}

int
main()
{
    kernel_test();

    return 0;
}
