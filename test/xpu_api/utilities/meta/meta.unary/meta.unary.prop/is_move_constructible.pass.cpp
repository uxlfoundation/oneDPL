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

template <class T>
void
test_is_move_constructible(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(dpl::is_move_constructible<T>::value);
            static_assert(dpl::is_move_constructible_v<T>);
        });
    });
}

template <class T>
void
test_is_not_move_constructible(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!dpl::is_move_constructible<T>::value);
            static_assert(!dpl::is_move_constructible_v<T>);
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

struct B
{
    B(B&&);
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_not_move_constructible<char[3]>(deviceQueue);
    test_is_not_move_constructible<char[]>(deviceQueue);
    test_is_not_move_constructible<void>(deviceQueue);

    test_is_move_constructible<A>(deviceQueue);
    test_is_move_constructible<int&>(deviceQueue);
    test_is_move_constructible<Union>(deviceQueue);
    test_is_move_constructible<Empty>(deviceQueue);
    test_is_move_constructible<int>(deviceQueue);
    test_is_move_constructible<int*>(deviceQueue);
    test_is_move_constructible<const int*>(deviceQueue);
    test_is_move_constructible<bit_zero>(deviceQueue);
    test_is_move_constructible<B>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_move_constructible<double>(deviceQueue);
    }
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
