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

// template<class Iter>
// struct iterator_traits
// {
//   typedef typename Iter::difference_type difference_type;
//   typedef typename Iter::value_type value_type;
//   typedef typename Iter::pointer pointer;
//   typedef typename Iter::reference reference;
//   typedef typename Iter::iterator_category iterator_category;
// };

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

struct A
{
};

struct test_iterator
{
    typedef int difference_type;
    typedef A value_type;
    typedef A* pointer;
    typedef A& reference;
    typedef std::forward_iterator_tag iterator_category;
};

void
kernelTest()
{
    sycl::queue q = TestUtils::get_test_queue();
    q.submit([&](sycl::handler& cgh) {
        cgh.single_task<class IteratorTest>([=]() {
            typedef dpl::iterator_traits<test_iterator> It;
            static_assert(dpl::is_same<It::difference_type, int>::value);
            static_assert(dpl::is_same<It::value_type, A>::value);
            static_assert(dpl::is_same<It::pointer, A*>::value);
            static_assert(dpl::is_same<It::reference, A&>::value);
            static_assert(dpl::is_same<It::iterator_category, dpl::forward_iterator_tag>::value);
        });
    });
}

int
main()
{
    kernelTest();

    return TestUtils::done();
}
