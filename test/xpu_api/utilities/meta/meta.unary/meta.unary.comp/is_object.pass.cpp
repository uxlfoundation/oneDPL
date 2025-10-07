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

#include <oneapi/dpl/cstddef> // for dpl::nullptr_t
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

template <class T>
void
test_is_object()
{
    static_assert(dpl::is_object<T>::value);
    static_assert(dpl::is_object<const T>::value);
    static_assert(dpl::is_object<volatile T>::value);
    static_assert(dpl::is_object<const volatile T>::value);

    static_assert(dpl::is_object_v<T>);
    static_assert(dpl::is_object_v<const T>);
    static_assert(dpl::is_object_v<volatile T>);
    static_assert(dpl::is_object_v<const volatile T>);
}

template <class T>
void
test_is_not_object()
{
    static_assert(!dpl::is_object<T>::value);
    static_assert(!dpl::is_object<const T>::value);
    static_assert(!dpl::is_object<volatile T>::value);
    static_assert(!dpl::is_object<const volatile T>::value);

    static_assert(!dpl::is_object_v<T>);
    static_assert(!dpl::is_object_v<const T>);
    static_assert(!dpl::is_object_v<volatile T>);
    static_assert(!dpl::is_object_v<const volatile T>);
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

class Abstract
{
    virtual ~Abstract() = 0;
};

enum Enum
{
    zero,
    one
};

typedef void (*FunctionPtr)();

bool
kernel_test()
{
    // An object type is a (possibly cv-qualified) type that is not a function
    // type, not a reference type, and not a void type.
    test_is_object<dpl::nullptr_t>();
    test_is_object<void*>();
    test_is_object<char[3]>();
    test_is_object<char[]>();
    test_is_object<int>();
    test_is_object<int*>();
    test_is_object<Union>();
    test_is_object<int*>();
    test_is_object<const int*>();
    test_is_object<Enum>();
    test_is_object<incomplete_type>();
    test_is_object<bit_zero>();
    test_is_object<NotEmpty>();
    test_is_object<Abstract>();
    test_is_object<FunctionPtr>();
    test_is_object<int Empty::*>();
    test_is_object<void (Empty::*)(int)>();

    test_is_not_object<void>();
    test_is_not_object<int&>();
    test_is_not_object<int&&>();
    test_is_not_object<int(int)>();
    return true;
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();

    bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of `is_object` check");

    return 0;
}
