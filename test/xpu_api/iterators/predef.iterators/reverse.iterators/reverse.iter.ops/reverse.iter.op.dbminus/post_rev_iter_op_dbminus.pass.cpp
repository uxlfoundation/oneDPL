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

// constexpr reverse_iterator operator--(int);
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
test(It i, It x)
{
    dpl::reverse_iterator<It> r(i);
    dpl::reverse_iterator<It> rr = r--;
    auto ret = (r.base() == x);
    ret &= (rr.base() == i);
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
                const char* s = "123";
                ret_access[0] &=
                    test(bidirectional_iterator<const char*>(s + 1), bidirectional_iterator<const char*>(s + 2));
                ret_access[0] &=
                    test(random_access_iterator<const char*>(s + 1), random_access_iterator<const char*>(s + 2));
                ret_access[0] &= test(s + 1, s + 2);

                {
                    constexpr const char* p = "123456789";
                    typedef dpl::reverse_iterator<const char*> RI;
                    constexpr RI it1 = dpl::make_reverse_iterator(p);
                    constexpr RI it2 = dpl::make_reverse_iterator(p + 1);
                    static_assert(it1 != it2);
                    constexpr RI it3 = dpl::make_reverse_iterator(p)--;
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
    EXPECT_TRUE(ret, "Wrong result of work with reverse iterator and post decrement in kernel_test()");

    return TestUtils::done();
}
