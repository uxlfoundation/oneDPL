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
test_is_member_pointer(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(dpl::is_member_pointer<T>::value);
            static_assert(dpl::is_member_pointer<const T>::value);
            static_assert(dpl::is_member_pointer<volatile T>::value);
            static_assert(dpl::is_member_pointer<const volatile T>::value);

            static_assert(dpl::is_member_pointer_v<T>);
            static_assert(dpl::is_member_pointer_v<const T>);
            static_assert(dpl::is_member_pointer_v<volatile T>);
            static_assert(dpl::is_member_pointer_v<const volatile T>);
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_member_pointer(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!dpl::is_member_pointer<T>::value);
            static_assert(!dpl::is_member_pointer<const T>::value);
            static_assert(!dpl::is_member_pointer<volatile T>::value);
            static_assert(!dpl::is_member_pointer<const volatile T>::value);

            static_assert(!dpl::is_member_pointer_v<T>);
            static_assert(!dpl::is_member_pointer_v<const T>);
            static_assert(!dpl::is_member_pointer_v<volatile T>);
            static_assert(!dpl::is_member_pointer_v<const volatile T>);
        });
    });
}

class incomplete_type;

class Empty
{
};

class NotEmpty
{
    virtual ~NotEmpty();
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

void
kernel_test()
{

    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_not_member_pointer<KernelTest1, dpl::nullptr_t>(deviceQueue);
    test_is_not_member_pointer<KernelTest2, void>(deviceQueue);
    test_is_not_member_pointer<KernelTest3, void*>(deviceQueue);
    test_is_not_member_pointer<KernelTest4, int>(deviceQueue);
    test_is_not_member_pointer<KernelTest5, int*>(deviceQueue);
    test_is_not_member_pointer<KernelTest6, const int*>(deviceQueue);
    test_is_not_member_pointer<KernelTest7, int&>(deviceQueue);
    test_is_not_member_pointer<KernelTest8, int&&>(deviceQueue);
    test_is_not_member_pointer<KernelTest9, char[3]>(deviceQueue);
    test_is_not_member_pointer<KernelTest10, char[]>(deviceQueue);
    test_is_not_member_pointer<KernelTest11, Union>(deviceQueue);
    test_is_not_member_pointer<KernelTest12, Empty>(deviceQueue);
    test_is_not_member_pointer<KernelTest13, incomplete_type>(deviceQueue);
    test_is_not_member_pointer<KernelTest14, bit_zero>(deviceQueue);
    test_is_not_member_pointer<KernelTest15, NotEmpty>(deviceQueue);
    test_is_not_member_pointer<KernelTest16, int(int)>(deviceQueue);
    test_is_not_member_pointer<KernelTest17, Enum>(deviceQueue);
    test_is_not_member_pointer<KernelTest18, FunctionPtr>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_not_member_pointer<KernelTest19, double>(deviceQueue);
    }
    test_is_member_pointer<KernelTest20, int Empty::*>(deviceQueue);
    test_is_member_pointer<KernelTest21, void (Empty::*)(int)>(deviceQueue);
}

int
main()
{
    kernel_test();

    return 0;
}
