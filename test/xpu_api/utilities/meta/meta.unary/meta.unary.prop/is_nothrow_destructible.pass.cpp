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
test_is_nothrow_destructible(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(dpl::is_nothrow_destructible<T>::value);
            static_assert(dpl::is_nothrow_destructible<const T>::value);
            static_assert(dpl::is_nothrow_destructible<volatile T>::value);
            static_assert(dpl::is_nothrow_destructible<const volatile T>::value);
            static_assert(dpl::is_nothrow_destructible_v<T>);
            static_assert(dpl::is_nothrow_destructible_v<const T>);
            static_assert(dpl::is_nothrow_destructible_v<volatile T>);
            static_assert(dpl::is_nothrow_destructible_v<const volatile T>);
        });
    });
}

template <class T>
void
test_is_not_nothrow_destructible(sycl::queue deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!dpl::is_nothrow_destructible<T>::value);
            static_assert(!dpl::is_nothrow_destructible<const T>::value);
            static_assert(!dpl::is_nothrow_destructible<volatile T>::value);
            static_assert(!dpl::is_nothrow_destructible<const volatile T>::value);
            static_assert(!dpl::is_nothrow_destructible_v<T>);
            static_assert(!dpl::is_nothrow_destructible_v<const T>);
            static_assert(!dpl::is_nothrow_destructible_v<volatile T>);
            static_assert(!dpl::is_nothrow_destructible_v<const volatile T>);
        });
    });
}

struct PublicDestructor
{
  public:
    ~PublicDestructor() {}
};

struct PublicDestructorT
{
  public:
    ~PublicDestructorT() noexcept(false) {}
};

struct ProtectedDestructor
{
  protected:
    ~ProtectedDestructor() {}
};

struct PrivateDestructor
{
  private:
    ~PrivateDestructor() {}
};

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

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_not_nothrow_destructible<void>(deviceQueue);
    test_is_not_nothrow_destructible<char[]>(deviceQueue);
    test_is_not_nothrow_destructible<char[][3]>(deviceQueue);

    test_is_nothrow_destructible<int&>(deviceQueue);
    test_is_nothrow_destructible<int>(deviceQueue);
    test_is_nothrow_destructible<int*>(deviceQueue);
    test_is_nothrow_destructible<const int*>(deviceQueue);
    test_is_nothrow_destructible<char[3]>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_nothrow_destructible<double>(deviceQueue);
    }

    test_is_not_nothrow_destructible<PublicDestructorT>(deviceQueue);

    // requires noexcept. These are all destructible.
    test_is_nothrow_destructible<PublicDestructor>(deviceQueue);
    test_is_nothrow_destructible<bit_zero>(deviceQueue);
    test_is_nothrow_destructible<Empty>(deviceQueue);
    test_is_nothrow_destructible<Union>(deviceQueue);

    // requires access control
    test_is_not_nothrow_destructible<ProtectedDestructor>(deviceQueue);
    test_is_not_nothrow_destructible<PrivateDestructor>(deviceQueue);
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
