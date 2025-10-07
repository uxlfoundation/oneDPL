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

// template <RandomAccessIterator Iter1, RandomAccessIterator Iter2>
//   requires HasGreater<Iter1, Iter2>
//   constexpr bool
//   operator<=(const reverse_iterator<Iter1>& x, const reverse_iterator<Iter2>&
//   y);
//
//   constexpr in C++17

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/test_iterators.h"
#include "support/utils.h"

template <class It>
bool
test(It l, It r, bool x)
{
    const dpl::reverse_iterator<It> r1(l);
    const dpl::reverse_iterator<It> r2(r);
    return ((r1 <= r2) == x);
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
                    test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s), true);
                ret_access[0] &=
                    test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s + 1), false);
                ret_access[0] &=
                    test(random_access_iterator<const char*>(s + 1), random_access_iterator<const char*>(s), true);
                ret_access[0] &= test(s, s, true);
                ret_access[0] &= test(s, s + 1, false);
                ret_access[0] &= test(s + 1, s, true);

                {
                    constexpr const char* p = "123456789";
                    typedef dpl::reverse_iterator<const char*> RI;
                    constexpr RI it1 = dpl::make_reverse_iterator(p);
                    constexpr RI it2 = dpl::make_reverse_iterator(p);
                    constexpr RI it3 = dpl::make_reverse_iterator(p + 1);
                    static_assert(it1 <= it2);
                    static_assert(!(it1 <= it3));
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
    EXPECT_TRUE(ret, "Wrong result of work with reverse iterator and '<=' in kernel_test()");

    return TestUtils::done();
}
