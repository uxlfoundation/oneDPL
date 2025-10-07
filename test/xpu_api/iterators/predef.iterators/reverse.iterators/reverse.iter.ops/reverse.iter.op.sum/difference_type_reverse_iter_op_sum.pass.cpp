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

// template <RandomAccessIterator Iterator>
//   constexpr reverse_iterator<Iter>
//   operator+(Iter::difference_type n, const reverse_iterator<Iter>& x);
//
// constexpr in C++17

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/test_iterators.h"
#include "support/utils.h"

template <class It>
bool
test(It i, typename dpl::iterator_traits<It>::difference_type n, It x)
{
    const dpl::reverse_iterator<It> r(i);
    dpl::reverse_iterator<It> rr = n + r;
    return (rr.base() == x);
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
                const char* s = "1234567890";
                ret_access[0] &=
                    test(random_access_iterator<const char*>(s + 5), 5, random_access_iterator<const char*>(s));
                ret_access[0] &= test(s + 5, 5, s);

                {
                    constexpr const char* p = "123456789";
                    typedef dpl::reverse_iterator<const char*> RI;
                    constexpr RI it1 = dpl::make_reverse_iterator(p);
                    constexpr RI it2 = dpl::make_reverse_iterator(p + 5);
                    constexpr RI it3 = 5 + it2;
                    static_assert(it1 != it2);
                    static_assert(it1 == it3);
                    static_assert(it2 != it3);
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
    EXPECT_TRUE(ret, "Wrong result of work with reverse iterator and summ in kernel_test()");

    return TestUtils::done();
}
