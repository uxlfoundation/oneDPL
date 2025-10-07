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

// template <RandomAccessIterator Iter1, RandomAccessIterator Iter2>
//   requires HasMinus<Iter1, Iter2>
//   auto
//   operator-(const move_iterator<Iter1>& x, const move_iterator<Iter2>& y)
//   -> decltype(x.base() - y.base());
//
//  constexpr in C++17

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_iterators.h"
#include "support/utils.h"

template <class It>
bool
test(It l, It r, typename dpl::iterator_traits<It>::difference_type x)
{
    const dpl::move_iterator<It> r1(l);
    const dpl::move_iterator<It> r2(r);
    return (r1 - r2 == x);
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
                char s[] = "1234567890";
                ret_access[0] &= test(random_access_iterator<char*>(s + 5), random_access_iterator<char*>(s), 5);
                ret_access[0] &= test(s + 5, s, 5);

                {
                    constexpr const char* p = "123456789";
                    typedef dpl::move_iterator<const char*> MI;
                    constexpr MI it1 = dpl::make_move_iterator(p);
                    constexpr MI it2 = dpl::make_move_iterator(p + 1);
                    static_assert(it1 - it2 == -1);
                    static_assert(it2 - it1 == 1);
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
    EXPECT_TRUE(ret, "Wrong result of operator-(...) in kernel_test()");

    return TestUtils::done();
}
