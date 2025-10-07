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

template <class T>
void
test_is_nothrow_constructible()
{
    static_assert(dpl::is_nothrow_constructible<T>::value);
    static_assert(dpl::is_nothrow_constructible_v<T>);
}

template <class T, class A0>
void
test_is_nothrow_constructible()
{
    static_assert(dpl::is_nothrow_constructible<T, A0>::value);
    static_assert(dpl::is_nothrow_constructible_v<T, A0>);
}

template <class T>
void
test_is_not_nothrow_constructible()
{
    static_assert(!dpl::is_nothrow_constructible<T>::value);
    static_assert(!dpl::is_nothrow_constructible_v<T>);
}

template <class T, class A0>
void
test_is_not_nothrow_constructible()
{
    static_assert(!dpl::is_nothrow_constructible<T, A0>::value);
    static_assert(!dpl::is_nothrow_constructible_v<T, A0>);
}

template <class T, class A0, class A1>
void
test_is_not_nothrow_constructible()
{
    static_assert(!dpl::is_nothrow_constructible<T, A0, A1>::value);
    static_assert(!dpl::is_nothrow_constructible_v<T, A0, A1>);
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
    ANT() noexcept;
    ANT(const ANT&) noexcept;
};

struct C
{
    C(C&); // not const
    void
    operator=(C&); // not const
};

struct CNT
{
    CNT() noexcept;
    CNT(CNT&) noexcept; // not const
    void
    operator=(CNT&) noexcept; // not const
};

struct Tuple
{
    Tuple(Empty&&) noexcept {}
};

bool
kernel_test()
{
    test_is_nothrow_constructible<int>();
    test_is_nothrow_constructible<int, const int&>();
    test_is_nothrow_constructible<Empty>();
    test_is_nothrow_constructible<Empty, const Empty&>();
    test_is_nothrow_constructible<ANT>();
    test_is_nothrow_constructible<ANT, const ANT&>();
    test_is_nothrow_constructible<CNT>();
    test_is_nothrow_constructible<CNT, CNT&>();
    test_is_nothrow_constructible<Tuple&&, Empty>();

    test_is_not_nothrow_constructible<A, int>();
    test_is_not_nothrow_constructible<A, int, float>();
    test_is_not_nothrow_constructible<A>();
    test_is_not_nothrow_constructible<A, const A&>();
    test_is_not_nothrow_constructible<C>();
    test_is_not_nothrow_constructible<C, C&>();

    static_assert(!dpl::is_constructible<Tuple&, Empty>::value);
    test_is_not_nothrow_constructible<Tuple&, Empty>();

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

    EXPECT_TRUE(ret, "Wrong result of dpl::is_nothrow_constructible check");

    return TestUtils::done();
}
