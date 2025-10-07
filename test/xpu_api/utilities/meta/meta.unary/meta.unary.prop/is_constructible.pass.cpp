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

struct A
{
    explicit A(int);
    A(int, double);
    A(int, long, double);

  private:
    A(char);
};

struct Base
{
};
struct Derived : public Base
{
};

struct PrivateDtor
{
    PrivateDtor(int) {}

  private:
    ~PrivateDtor() {}
};

struct S
{
    template <class T>
    explicit operator T() const;
};

template <class To>
struct ImplicitTo
{
    operator To();
};

template <class To>
struct ExplicitTo
{
    explicit operator To();
};

template <class T>
void
test_is_constructible()
{
    static_assert(dpl::is_constructible<T>::value);
    static_assert(dpl::is_constructible_v<T>);
}

template <class T, class A0>
void
test_is_constructible()
{
    static_assert(dpl::is_constructible<T, A0>::value);
    static_assert(dpl::is_constructible_v<T, A0>);
}

template <class T, class A0, class A1>
void
test_is_constructible()
{
    static_assert(dpl::is_constructible<T, A0, A1>::value);
    static_assert(dpl::is_constructible_v<T, A0, A1>);
}

template <class T, class A0, class A1, class A2>
void
test_is_constructible()
{
    static_assert(dpl::is_constructible<T, A0, A1, A2>::value);
    static_assert(dpl::is_constructible_v<T, A0, A1, A2>);
}

template <class T>
void
test_is_not_constructible()
{
    static_assert(!dpl::is_constructible<T>::value);
    static_assert(!dpl::is_constructible_v<T>);
}

template <class T, class A0>
void
test_is_not_constructible()
{
    static_assert(!dpl::is_constructible<T, A0>::value);
    static_assert(!dpl::is_constructible_v<T, A0>);
}

void
kernel_test1(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest1>([=]() {
            test_is_constructible<int>();
            test_is_constructible<int, const int>();
            test_is_constructible<int&, int&>();

            test_is_not_constructible<int, void()>();
            test_is_not_constructible<int, void (&)()>();
            test_is_not_constructible<int, void() const>();
            test_is_not_constructible<int&, void>();
            test_is_not_constructible<int&, void()>();
            test_is_not_constructible<int&, void() const>();
            test_is_not_constructible<int&, void (&)()>();

            test_is_not_constructible<void>();
            test_is_not_constructible<const void>(); // LWG 2738
            test_is_not_constructible<volatile void>();
            test_is_not_constructible<const volatile void>();
            test_is_not_constructible<int&>();
            test_is_constructible<int, S>();
            test_is_not_constructible<int&, S>();

            test_is_constructible<void (&)(), void (&)()>();
            test_is_constructible<void (&)(), void()>();
            test_is_constructible<void(&&)(), void(&&)()>();
            test_is_constructible<void(&&)(), void()>();
            test_is_constructible<void(&&)(), void (&)()>();

            test_is_constructible<int const&, int>();
            test_is_constructible<int const&, int&&>();

            test_is_constructible<void (&)(), void(&&)()>();

            test_is_not_constructible<int&, int>();
            test_is_not_constructible<int&, int const&>();
            test_is_not_constructible<int&, int&&>();

            test_is_constructible<int&&, int>();
            test_is_constructible<int&&, int&&>();
            test_is_not_constructible<int&&, int&>();
            test_is_not_constructible<int&&, int const&&>();

            test_is_constructible<Base, Derived>();
            test_is_constructible<Base&, Derived&>();
            test_is_not_constructible<Derived&, Base&>();
            test_is_constructible<Base const&, Derived const&>();
            test_is_not_constructible<Derived const&, Base const&>();
            test_is_not_constructible<Derived const&, Base>();

            test_is_constructible<Base&&, Derived>();
            test_is_constructible<Base&&, Derived&&>();
            test_is_not_constructible<Derived&&, Base&&>();
            test_is_not_constructible<Derived&&, Base>();

            // test that T must also be destructible
            test_is_constructible<PrivateDtor&, PrivateDtor&>();
            test_is_not_constructible<PrivateDtor, int>();

            test_is_not_constructible<void() const, void() const>();
            test_is_not_constructible<void() const, void*>();

            test_is_constructible<int&, ImplicitTo<int&>>();
            test_is_constructible<const int&, ImplicitTo<int&&>>();
            test_is_constructible<int&&, ImplicitTo<int&&>>();
            test_is_constructible<const int&, ImplicitTo<int>>();

            test_is_not_constructible<Base&&, Base&>();
            test_is_not_constructible<Base&&, Derived&>();
            test_is_constructible<Base&&, ImplicitTo<Derived&&>>();
            test_is_constructible<Base&&, ImplicitTo<Derived&&>&>();
            test_is_constructible<const int&, ImplicitTo<int&>&>();
            test_is_constructible<const int&, ImplicitTo<int&>>();
            test_is_constructible<const int&, ExplicitTo<int&>&>();
            test_is_constructible<const int&, ExplicitTo<int&>>();

            test_is_constructible<const int&, ExplicitTo<int&>&>();
            test_is_constructible<const int&, ExplicitTo<int&>>();
            test_is_constructible<int&, ExplicitTo<int&>>();
            test_is_constructible<const int&, ExplicitTo<int&&>>();

            test_is_constructible<int&, ExplicitTo<int&>>();

            static_assert(dpl::is_constructible<int&&, ExplicitTo<int&&>>::value);
        });
    });
}

void
kernel_test2(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest2>([=]() {
            test_is_constructible<A, int>();
            test_is_constructible<A, int, double>();
            test_is_constructible<A, int, long, double>();

            test_is_not_constructible<A>();
            test_is_not_constructible<A, char>();
            test_is_not_constructible<A, void>();
            test_is_constructible<int&&, double&>();
            test_is_constructible<int&&, double&>();
            test_is_not_constructible<const int&, ExplicitTo<double&&>>();
            test_is_not_constructible<int&&, ExplicitTo<double&&>>();
        });
    });
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test1(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        kernel_test2(deviceQueue);
    }

    return TestUtils::done();
}
