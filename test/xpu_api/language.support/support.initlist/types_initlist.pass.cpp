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

// template<class E>
// class initializer_list
// {
// public:
//     typedef E        value_type;
//     typedef const E& reference;
//     typedef const E& const_reference;
//     typedef size_t   size_type;
//
//     typedef const E* iterator;
//     typedef const E* const_iterator;

#include "support/test_config.h"

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/cstddef>

#include "support/test_macros.h"
#include "support/utils.h"

#include <initializer_list>

struct A
{
};

int
main()
{
    bool ret = true;
    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{1});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                static_assert(dpl::is_same<std::initializer_list<A>::value_type, A>::value);
                static_assert(dpl::is_same<std::initializer_list<A>::reference, const A&>::value);
                static_assert(dpl::is_same<std::initializer_list<A>::const_reference, const A&>::value);
                static_assert(dpl::is_same<std::initializer_list<A>::size_type, dpl::size_t>::value);
                static_assert(dpl::is_same<std::initializer_list<A>::iterator, const A*>::value);
                static_assert(dpl::is_same<std::initializer_list<A>::const_iterator, const A*>::value);
                acc[0] = true;
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result with initializer list");

    return TestUtils::done();
}
