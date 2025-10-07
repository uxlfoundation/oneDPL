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
test_is_destructible(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(dpl::is_destructible<T>::value);
            static_assert(dpl::is_destructible<const T>::value);
            static_assert(dpl::is_destructible<volatile T>::value);
            static_assert(dpl::is_destructible<const volatile T>::value);
            static_assert(dpl::is_destructible_v<T>);
            static_assert(dpl::is_destructible_v<const T>);
            static_assert(dpl::is_destructible_v<volatile T>);
            static_assert(dpl::is_destructible_v<const volatile T>);
        });
    });
}

template <class T>
void
test_is_not_destructible(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!dpl::is_destructible<T>::value);
            static_assert(!dpl::is_destructible<const T>::value);
            static_assert(!dpl::is_destructible<volatile T>::value);
            static_assert(!dpl::is_destructible<const volatile T>::value);
            static_assert(!dpl::is_destructible_v<T>);
            static_assert(!dpl::is_destructible_v<const T>);
            static_assert(!dpl::is_destructible_v<volatile T>);
            static_assert(!dpl::is_destructible_v<const volatile T>);
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

typedef void(Function)();

struct PublicDestructor
{
  public:
    ~PublicDestructor() {}
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

struct DeletedPublicDestructor
{
  public:
    ~DeletedPublicDestructor() = delete;
};
struct DeletedProtectedDestructor
{
  protected:
    ~DeletedProtectedDestructor() = delete;
};
struct DeletedPrivateDestructor
{
  private:
    ~DeletedPrivateDestructor() = delete;
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_destructible<int&>(deviceQueue);
    test_is_destructible<Union>(deviceQueue);
    test_is_destructible<Empty>(deviceQueue);
    test_is_destructible<int>(deviceQueue);
    test_is_destructible<int*>(deviceQueue);
    test_is_destructible<const int*>(deviceQueue);
    test_is_destructible<char[3]>(deviceQueue);
    test_is_destructible<bit_zero>(deviceQueue);
    test_is_destructible<int[3]>(deviceQueue);
    test_is_destructible<PublicDestructor>(deviceQueue);

    test_is_not_destructible<int[]>(deviceQueue);
    test_is_not_destructible<void>(deviceQueue);
    test_is_not_destructible<Function>(deviceQueue);

    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_destructible<double>(deviceQueue);
    }

    // Test access controlled destructors
    test_is_not_destructible<ProtectedDestructor>(deviceQueue);
    test_is_not_destructible<PrivateDestructor>(deviceQueue);

    // Test deleted constructors
    test_is_not_destructible<DeletedPublicDestructor>(deviceQueue);
    test_is_not_destructible<DeletedProtectedDestructor>(deviceQueue);
    test_is_not_destructible<DeletedPrivateDestructor>(deviceQueue);
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
