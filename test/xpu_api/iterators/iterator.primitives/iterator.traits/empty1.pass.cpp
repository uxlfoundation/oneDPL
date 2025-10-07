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

// template<class NotAnIterator>
// struct iterator_traits
// {
// };

#include "support/test_config.h"

#include <oneapi/dpl/iterator>

#include "support/utils.h"

struct not_an_iterator
{
};

template <class T>
struct has_value_type
{
  private:
    struct two
    {
        char lx;
        char lxx;
    };
    template <class U>
    static two
    test(...);
    template <class U>
    static char
    test(typename U::value_type* = 0);

  public:
    static const bool value = sizeof(test<T>(0)) == 1;
};

void
kernelTest()
{
    sycl::queue q = TestUtils::get_test_queue();
    q.submit([&](sycl::handler& cgh) {
        cgh.single_task<class IteratorTest>([=]() {
            typedef dpl::iterator_traits<not_an_iterator> It;
            static_assert(!has_value_type<It>::value);
        });
    });
}

int
main()
{
    kernelTest();

    return TestUtils::done();
}
