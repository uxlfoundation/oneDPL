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

// template <InputIterator Iter>
//   move_iterator<Iter>
//   make_move_iterator(const Iter& i);
//
//  constexpr in C++17

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_iterators.h"
#include "support/utils.h"

#define TEST_IGNORE_NODISCARD (void)

template <class It>
bool
test(It i)
{
    const dpl::move_iterator<It> r(i);
    return (dpl::make_move_iterator(i) == r);
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
                    char s[] = "1234567890";
                    ret_access[0] &= test(input_iterator<char*>(s + 5));
                    ret_access[0] &= test(forward_iterator<char*>(s + 5));
                    ret_access[0] &= test(bidirectional_iterator<char*>(s + 5));
                    ret_access[0] &= test(random_access_iterator<char*>(s + 5));
                    ret_access[0] &= test(s + 5);
                }
                {
                    int a[] = {1, 2, 3, 4};
                    TEST_IGNORE_NODISCARD dpl::make_move_iterator(a + 4);
                    TEST_IGNORE_NODISCARD dpl::make_move_iterator(a); // test for LWG issue 2061
                }

                {
                    constexpr const char* p = "123456789";
                    constexpr auto iter = dpl::make_move_iterator<const char*>(p);
                    static_assert(iter.base() == p);
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
    EXPECT_TRUE(ret, "Wrong result of dpl::move_iterator / dpl::make_move_iterator in kernel_test");

    return TestUtils::done();
}
