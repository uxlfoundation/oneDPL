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

//   All of these became constexpr in C++17
//
// template <InputIterator Iter>
//   constexpr void advance(Iter& i, Iter::difference_type n);
//
// template <BidirectionalIterator Iter>
//   constexpr void advance(Iter& i, Iter::difference_type n);
//
// template <RandomAccessIterator Iter>
//   constexpr void advance(Iter& i, Iter::difference_type n);

#include "support/test_config.h"

#include <oneapi/dpl/iterator>

#include "support/test_iterators.h"
#include "support/utils.h"

template <class It>
bool
test(It i, typename std::iterator_traits<It>::difference_type n, It x)
{
    dpl::advance(i, n);
    return (i == x);
}

template <class It>
constexpr bool
constepxr_test(It i, typename std::iterator_traits<It>::difference_type n, It x)
{
    dpl::advance(i, n);
    return i == x;
}

bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = true;
    {
        sycl::range<1> numOfItems{1};
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                {
                    const char* s = "1234567890";
                    ret_access[0] &= test(input_iterator<const char*>(s), 10, input_iterator<const char*>(s + 10));
                    ret_access[0] &= test(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s + 10));
                    ret_access[0] &= test(bidirectional_iterator<const char*>(s + 5), 5,
                                          bidirectional_iterator<const char*>(s + 10));
                    ret_access[0] &=
                        test(bidirectional_iterator<const char*>(s + 5), -5, bidirectional_iterator<const char*>(s));
                    ret_access[0] &= test(random_access_iterator<const char*>(s + 5), 5,
                                          random_access_iterator<const char*>(s + 10));
                    ret_access[0] &=
                        test(random_access_iterator<const char*>(s + 5), -5, random_access_iterator<const char*>(s));
                    ret_access[0] &= test(s + 5, 5, s + 10);
                    ret_access[0] &= test(s + 5, -5, s);
                }

                {
                    constexpr const char* s = "1234567890";
                    static_assert(
                        constepxr_test(input_iterator<const char*>(s), 10, input_iterator<const char*>(s + 10)));
                    static_assert(
                        constepxr_test(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s + 10)));
                    static_assert(constepxr_test(bidirectional_iterator<const char*>(s + 5), 5,
                                                 bidirectional_iterator<const char*>(s + 10)));
                    static_assert(constepxr_test(bidirectional_iterator<const char*>(s + 5), -5,
                                                 bidirectional_iterator<const char*>(s)));
                    static_assert(constepxr_test(random_access_iterator<const char*>(s + 5), 5,
                                                 random_access_iterator<const char*>(s + 10)));
                    static_assert(constepxr_test(random_access_iterator<const char*>(s + 5), -5,
                                                 random_access_iterator<const char*>(s)));
                    static_assert(constepxr_test(s + 5, 5, s + 10));
                    static_assert(constepxr_test(s + 5, -5, s));
                }
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Error in work with dpl::advance in kernel_test()");

    return TestUtils::done();
}
