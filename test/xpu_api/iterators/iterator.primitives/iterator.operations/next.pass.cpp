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

// template <InputIterator Iter>
//   Iter next(Iter x, Iter::difference_type n = 1);

// LWG #2353 relaxed the requirement on next from ForwardIterator to
// InputIterator

#include "support/test_config.h"

#include <oneapi/dpl/iterator>

#include "support/test_iterators.h"
#include "support/utils.h"

template <class It>
bool
test(It i, typename std::iterator_traits<It>::difference_type n, It x)
{
    return (dpl::next(i, n) == x);
}

template <class It>
bool
test(It i, It x)
{
    return (dpl::next(i) == x);
}

template <class It>
constexpr bool
constexpr_test(It i, typename std::iterator_traits<It>::difference_type n, It x)
{
    return dpl::next(i, n) == x;
}

template <class It>
constexpr bool
constexpr_test(It i, It x)
{
    return dpl::next(i) == x;
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
                    ret_access[0] &=
                        test(bidirectional_iterator<const char*>(s), 10, bidirectional_iterator<const char*>(s + 10));
                    ret_access[0] &=
                        test(bidirectional_iterator<const char*>(s + 10), -10, bidirectional_iterator<const char*>(s));
                    ret_access[0] &=
                        test(random_access_iterator<const char*>(s), 10, random_access_iterator<const char*>(s + 10));
                    ret_access[0] &=
                        test(random_access_iterator<const char*>(s + 10), -10, random_access_iterator<const char*>(s));
                    ret_access[0] &= test(s, 10, s + 10);

                    ret_access[0] &= test(input_iterator<const char*>(s), input_iterator<const char*>(s + 1));
                    ret_access[0] &= test(forward_iterator<const char*>(s), forward_iterator<const char*>(s + 1));
                    ret_access[0] &=
                        test(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s + 1));
                    ret_access[0] &=
                        test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s + 1));
                    ret_access[0] &= test(s, s + 1);
                }

                {
                    constexpr const char* s = "1234567890";
                    static_assert(
                        constexpr_test(input_iterator<const char*>(s), 10, input_iterator<const char*>(s + 10)));
                    static_assert(
                        constexpr_test(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s + 10)));
                    static_assert(constexpr_test(bidirectional_iterator<const char*>(s), 10,
                                                 bidirectional_iterator<const char*>(s + 10)));
                    static_assert(constexpr_test(bidirectional_iterator<const char*>(s + 10), -10,
                                                 bidirectional_iterator<const char*>(s)));
                    static_assert(constexpr_test(random_access_iterator<const char*>(s), 10,
                                                 random_access_iterator<const char*>(s + 10)));
                    static_assert(constexpr_test(random_access_iterator<const char*>(s + 10), -10,
                                                 random_access_iterator<const char*>(s)));
                    static_assert(constexpr_test(s, 10, s + 10));

                    static_assert(constexpr_test(input_iterator<const char*>(s), input_iterator<const char*>(s + 1)));
                    static_assert(
                        constexpr_test(forward_iterator<const char*>(s), forward_iterator<const char*>(s + 1)));
                    static_assert(constexpr_test(bidirectional_iterator<const char*>(s),
                                                 bidirectional_iterator<const char*>(s + 1)));
                    static_assert(constexpr_test(random_access_iterator<const char*>(s),
                                                 random_access_iterator<const char*>(s + 1)));
                    static_assert(constexpr_test(s, s + 1));
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
    EXPECT_TRUE(ret, "Error in work with dpl::next in kernel_test()");

    return TestUtils::done();
}
