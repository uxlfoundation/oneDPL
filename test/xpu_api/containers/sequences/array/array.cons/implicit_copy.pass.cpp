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

// <array>
// implicitly generated array constructors / assignment operators

#include "support/test_config.h"

#include <oneapi/dpl/array>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

#define TEST_NOT_COPY_ASSIGNABLE(T) static_assert(!dpl::is_copy_assignable<T>::value)

struct NoDefault
{
    NoDefault() {}
    NoDefault(int) {}
};

int
main()
{
    {
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest1>([=]() {
                {
                    typedef dpl::array<float, 3> C;
                    C c = {1.1f, 2.2f, 3.3f};
                    C c2 = c;
                    c2 = c;
                    static_assert(dpl::is_copy_constructible<C>::value);
                    static_assert(dpl::is_copy_assignable<C>::value);
                }
                {
                    typedef dpl::array<const float, 3> C;
                    C c = {1.1f, 2.2f, 3.3f};
                    C c2 = c;
                    ((void)c2);
                    static_assert(dpl::is_copy_constructible<C>::value);
                    TEST_NOT_COPY_ASSIGNABLE(C);
                }
                {
                    typedef dpl::array<float, 0> C;
                    C c = {};
                    C c2 = c;
                    c2 = c;
                    static_assert(dpl::is_copy_constructible<C>::value);
                    static_assert(dpl::is_copy_assignable<C>::value);
                }
                {
                    // const arrays of size 0 should disable the implicit copy assignment
                    // operator.
                    typedef dpl::array<const float, 0> C;
                    const C c = {{}};
                    C c2 = c;
                    ((void)c2);
                    static_assert(dpl::is_copy_constructible<C>::value);
                }
                {
                    typedef dpl::array<NoDefault, 0> C;
                    C c = {};
                    C c2 = c;
                    c2 = c;
                    static_assert(dpl::is_copy_constructible<C>::value);
                    static_assert(dpl::is_copy_assignable<C>::value);
                }
                {
                    typedef dpl::array<const NoDefault, 0> C;
                    C c = {{}};
                    C c2 = c;
                    ((void)c2);
                    static_assert(dpl::is_copy_constructible<C>::value);
                }
            });
        });
    }

    return TestUtils::done();
}
