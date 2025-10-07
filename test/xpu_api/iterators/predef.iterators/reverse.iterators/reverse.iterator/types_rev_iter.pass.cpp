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

// reverse_iterator

// Test nested types and data member:

// template <BidirectionalIterator Iter>
// class reverse_iterator {
// protected:
//   Iter current;
// public:
//   iterator<typename iterator_traits<Iterator>::iterator_category,
//   typename iterator_traits<Iterator>::value_type,
//   typename iterator_traits<Iterator>::difference_type,
//   typename iterator_traits<Iterator>::pointer,
//   typename iterator_traits<Iterator>::reference> {
// };

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/test_iterators.h"
#include "support/utils.h"

template <class It>
struct find_current : private dpl::reverse_iterator<It>
{
    void
    test()
    {
        ++(this->current);
    }
};

template <class Tt>
class kernelTest;

template <class It>
void
test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        sycl::range<1> numOfItems{1};
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<kernelTest<It>>([=]() {
                typedef dpl::reverse_iterator<It> R;
                typedef dpl::iterator_traits<It> T;
                find_current<It> q;
                q.test();
                static_assert(dpl::is_same<typename R::iterator_type, It>::value);
                static_assert(dpl::is_same<typename R::value_type, typename T::value_type>::value);
                static_assert(dpl::is_same<typename R::difference_type, typename T::difference_type>::value);
                static_assert(dpl::is_same<typename R::reference, typename T::reference>::value);
                static_assert(dpl::is_same<typename R::pointer, typename dpl::iterator_traits<It>::pointer>::value);
                static_assert(dpl::is_same<typename R::iterator_category, typename T::iterator_category>::value);
            });
        });
    }
}

int
main()
{
    test<bidirectional_iterator<char*>>();
    test<random_access_iterator<char*>>();
    test<char*>();

    return TestUtils::done();
}
