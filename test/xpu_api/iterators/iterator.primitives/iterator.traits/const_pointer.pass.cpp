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

// <iterator>

// template<class T>
// struct iterator_traits<const T*>
// {
//   typedef ptrdiff_t                  difference_type;
//   typedef T                          value_type;
//   typedef const T*                   pointer;
//   typedef const T&                   reference;
//   typedef random_access_iterator_tag iterator_category;
// };

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/cstddef>

#include "support/utils.h"

struct A
{
};

void
kernelTest()
{
    sycl::queue q = TestUtils::get_test_queue();
    q.submit([&](sycl::handler& cgh) {
        cgh.single_task<class IteratorTest>([=]() {
            typedef dpl::iterator_traits<const A*> It;
            static_assert(dpl::is_same<It::difference_type, dpl::ptrdiff_t>::value);
            static_assert(dpl::is_same<It::value_type, A>::value);
            static_assert(dpl::is_same<It::pointer, const A*>::value);
            static_assert(dpl::is_same<It::reference, const A&>::value);
            static_assert(dpl::is_same<It::iterator_category, dpl::random_access_iterator_tag>::value);
        });
    });
}

int
main()
{
    kernelTest();

    return TestUtils::done();
}
