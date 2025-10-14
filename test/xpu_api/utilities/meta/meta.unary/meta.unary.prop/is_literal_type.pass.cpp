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

/*
  Warning  'is_literal_type<std::nullptr_t>' is deprecated: warning STL4013: std::is_literal_type and std::is_literal_type_v are deprecated in C++17.
  You can define _SILENCE_CXX17_IS_LITERAL_TYPE_DEPRECATION_WARNING or _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS to suppress this warning.
 */
#define _SILENCE_CXX17_IS_LITERAL_TYPE_DEPRECATION_WARNING

#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "support/test_config.h"

#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

// dpl::is_literal_type is removed since C++20
#if TEST_STD_VER == 17
template <class KernelTest, class T>
void
test_is_literal_type(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(dpl::is_literal_type<T>::value);
            static_assert(dpl::is_literal_type<const T>::value);
            static_assert(dpl::is_literal_type<volatile T>::value);
            static_assert(dpl::is_literal_type<const volatile T>::value);
            static_assert(dpl::is_literal_type_v<T>);
            static_assert(dpl::is_literal_type_v<const T>);
            static_assert(dpl::is_literal_type_v<volatile T>);
            static_assert(dpl::is_literal_type_v<const volatile T>);
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_literal_type(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!dpl::is_literal_type<T>::value);
            static_assert(!dpl::is_literal_type<const T>::value);
            static_assert(!dpl::is_literal_type<volatile T>::value);
            static_assert(!dpl::is_literal_type<const volatile T>::value);
            static_assert(!dpl::is_literal_type_v<T>);
            static_assert(!dpl::is_literal_type_v<const T>);
            static_assert(!dpl::is_literal_type_v<volatile T>);
            static_assert(!dpl::is_literal_type_v<const volatile T>);
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

enum Enum
{
    zero,
    one
};

typedef void (*FunctionPtr)();

class KernelTest1;
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

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_literal_type<KernelTest1, std::nullptr_t>(deviceQueue);

    test_is_literal_type<KernelTest3, void>(deviceQueue);
    test_is_literal_type<KernelTest4, int>(deviceQueue);
    test_is_literal_type<KernelTest5, int*>(deviceQueue);
    test_is_literal_type<KernelTest6, const int*>(deviceQueue);
    test_is_literal_type<KernelTest7, int&>(deviceQueue);
    test_is_literal_type<KernelTest8, int&&>(deviceQueue);
    test_is_literal_type<KernelTest9, char[3]>(deviceQueue);
    test_is_literal_type<KernelTest10, char[]>(deviceQueue);
    test_is_literal_type<KernelTest11, Empty>(deviceQueue);
    test_is_literal_type<KernelTest12, bit_zero>(deviceQueue);
    test_is_literal_type<KernelTest13, Union>(deviceQueue);
    test_is_literal_type<KernelTest14, Enum>(deviceQueue);
    test_is_literal_type<KernelTest15, FunctionPtr>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_literal_type<KernelTest16, double>(deviceQueue);
    }
}
#endif // TEST_STD_VER

int
main()
{
#if TEST_STD_VER == 17
    kernel_test();
#endif // TEST_STD_VER

    return TestUtils::done(TEST_STD_VER == 17);
}

#ifdef __clang__
#    pragma clang diagnostic pop
#endif
