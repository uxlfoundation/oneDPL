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

// move_iterator& operator--();
//
//  constexpr in C++17

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_iterators.h"
#include "support/utils.h"

template <class It>
bool
test(It i, It x)
{
    dpl::move_iterator<It> r(i);
    dpl::move_iterator<It>& rr = --r;
    auto ret = (r.base() == x);
    ret &= (&rr == &r);
    return ret;
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
                char s[] = "123";
                ret_access[0] &= test(bidirectional_iterator<char*>(s + 1), bidirectional_iterator<char*>(s));
                ret_access[0] &= test(random_access_iterator<char*>(s + 1), random_access_iterator<char*>(s));
                ret_access[0] &= test(s + 1, s);

                {
                    constexpr const char* p = "123456789";
                    typedef dpl::move_iterator<const char*> MI;
                    constexpr MI it1 = dpl::make_move_iterator(p);
                    constexpr MI it2 = dpl::make_move_iterator(p + 1);
                    static_assert(it1 != it2);
                    constexpr MI it3 = --dpl::make_move_iterator(p + 1);
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
    EXPECT_TRUE(ret, "Wrong result of move_iterator and operator--(int) in kernel_test()");

    return TestUtils::done();
}
