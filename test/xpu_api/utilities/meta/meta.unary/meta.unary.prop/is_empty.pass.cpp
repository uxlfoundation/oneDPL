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

// T is a non-union class type with:
//  no non-static data members,
//  no unnamed bit-fields of non-zero length,
//  no virtual member functions,
//  no virtual base classes,
//  and no base class B for which is_empty_v<B> is false.

template <class T>
void
test_is_empty(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(dpl::is_empty<T>::value);
            static_assert(dpl::is_empty<const T>::value);
            static_assert(dpl::is_empty<volatile T>::value);
            static_assert(dpl::is_empty<const volatile T>::value);
            static_assert(dpl::is_empty_v<T>);
            static_assert(dpl::is_empty_v<const T>);
            static_assert(dpl::is_empty_v<volatile T>);
            static_assert(dpl::is_empty_v<const volatile T>);
        });
    });
}

template <class T>
void
test_is_not_empty(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!dpl::is_empty<T>::value);
            static_assert(!dpl::is_empty<const T>::value);
            static_assert(!dpl::is_empty<volatile T>::value);
            static_assert(!dpl::is_empty<const volatile T>::value);
            static_assert(!dpl::is_empty_v<T>);
            static_assert(!dpl::is_empty_v<const T>);
            static_assert(!dpl::is_empty_v<volatile T>);
            static_assert(!dpl::is_empty_v<const volatile T>);
        });
    });
}

class Empty
{
};
struct NotEmpty
{
    int foo;
};

union Union
{
};

struct EmptyBase : public Empty
{
};
struct NotEmptyBase : public NotEmpty
{
};

struct NonStaticMember
{
    int foo;
};

struct bit_zero
{
    int : 0;
};

struct bit_one
{
    int : 1;
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_not_empty<void>(deviceQueue);
    test_is_not_empty<int&>(deviceQueue);
    test_is_not_empty<int>(deviceQueue);
    test_is_not_empty<int*>(deviceQueue);
    test_is_not_empty<const int*>(deviceQueue);
    test_is_not_empty<char[3]>(deviceQueue);
    test_is_not_empty<char[]>(deviceQueue);
    test_is_not_empty<Union>(deviceQueue);
    test_is_not_empty<NotEmpty>(deviceQueue);
    test_is_not_empty<NotEmptyBase>(deviceQueue);
    test_is_not_empty<NonStaticMember>(deviceQueue);

    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_not_empty<double>(deviceQueue);
    }

    test_is_empty<Empty>(deviceQueue);
    test_is_empty<EmptyBase>(deviceQueue);
    test_is_empty<bit_zero>(deviceQueue);
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
