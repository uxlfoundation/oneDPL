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

// move_iterator

// Test nested types:

// template <InputIterator Iter>
// class move_iterator {
// public:
//   typedef Iter                  iterator_type;
//   typedef Iter::difference_type difference_type;
//   typedef Iter                  pointer;
//   typedef Iter::value_type      value_type;
//   typedef value_type&&          reference;
// };

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/cstddef>
#include <oneapi/dpl/functional>

#include "support/test_iterators.h"
#include "support/utils.h"

template <class ValueType, class Reference>
struct DummyIt
{
    typedef dpl::forward_iterator_tag iterator_category;
    typedef ValueType value_type;
    typedef dpl::ptrdiff_t difference_type;
    typedef ValueType* pointer;
    typedef Reference reference;

    // Definition of operator* is not required, only the return type is needed
    // This operator would only be used by std::iter_rvalue_reference to determine
    // move_iterator::reference type starting from C++20
    reference
    operator*();
};

template <class It>
void
test()
{
    typedef dpl::move_iterator<It> R;
    typedef dpl::iterator_traits<It> T;
    static_assert(dpl::is_same<typename R::iterator_type, It>::value);
    static_assert(dpl::is_same<typename R::difference_type, typename T::difference_type>::value);
    static_assert(dpl::is_same<typename R::pointer, It>::value);
    static_assert(dpl::is_same<typename R::value_type, typename T::value_type>::value);
    static_assert(dpl::is_same<typename R::reference, typename R::value_type&&>::value);
    static_assert(dpl::is_same<typename R::iterator_category, typename T::iterator_category>::value);
}

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        sycl::range<1> numOfItems{1};
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                test<input_iterator<char*>>();
                test<forward_iterator<char*>>();
                test<bidirectional_iterator<char*>>();
                test<random_access_iterator<char*>>();
                test<char*>();
                {
                    typedef DummyIt<int, int> T;
                    typedef dpl::move_iterator<T> It;
                    static_assert(dpl::is_same<It::reference, int>::value);
                }
                {
                    typedef DummyIt<int, dpl::reference_wrapper<int>> T;
                    typedef dpl::move_iterator<T> It;
                    static_assert(dpl::is_same<It::reference, dpl::reference_wrapper<int>>::value);
                }
                {
                    // Check that move_iterator uses whatever reference type it's given
                    // when it's not a reference.
                    typedef DummyIt<int, long> T;
                    typedef dpl::move_iterator<T> It;
                    static_assert(dpl::is_same<It::reference, long>::value);
                }
                {
                    typedef DummyIt<int, int&> T;
                    typedef dpl::move_iterator<T> It;
                    static_assert(dpl::is_same<It::reference, int&&>::value);
                }
                {
                    typedef DummyIt<int, int&&> T;
                    typedef dpl::move_iterator<T> It;
                    static_assert(dpl::is_same<It::reference, int&&>::value);
                }
            });
        });
    }
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
