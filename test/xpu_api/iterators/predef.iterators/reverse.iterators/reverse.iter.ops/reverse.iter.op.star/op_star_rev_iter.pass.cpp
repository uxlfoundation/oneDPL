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

// constexpr reference operator*() const;
//
// constexpr in C++17

// Be sure to respect LWG 198:
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#198
// LWG 198 was superseded by LWG 2360
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2360

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

class A
{
    int data_;

  public:
    A() : data_(1) {}
    ~A() { data_ = -1; }

    friend bool
    operator==(const A& x, const A& y)
    {
        return x.data_ == y.data_;
    }
};

template <class It>
bool
test(It i, typename dpl::iterator_traits<It>::value_type x)
{
    dpl::reverse_iterator<It> r(i);
    return (*r == x);
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
                A a;
                ret_access[0] &= test(&a + 1, A());

                {
                    constexpr const char* p = "123456789";
                    typedef dpl::reverse_iterator<const char*> RI;
                    constexpr RI it1 = dpl::make_reverse_iterator(p + 1);
                    constexpr RI it2 = dpl::make_reverse_iterator(p + 2);
                    static_assert(*it1 == p[0]);
                    static_assert(*it2 == p[1]);
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
    EXPECT_TRUE(ret, "Wrong result of work with reverse iterator and '*' in kernel_test()");

    return TestUtils::done();
}
