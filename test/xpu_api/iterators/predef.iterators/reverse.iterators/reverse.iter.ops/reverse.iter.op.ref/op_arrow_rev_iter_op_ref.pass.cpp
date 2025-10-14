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

// constexpr pointer operator->() const;
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

    int
    get() const
    {
        return data_;
    }

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
    return (r->get() == x.get());
}

class B
{
    int data_;

  public:
    B(int d = 1) : data_(d) {}
    ~B() { data_ = -1; }

    int
    get() const
    {
        return data_;
    }

    friend bool
    operator==(const B& x, const B& y)
    {
        return x.data_ == y.data_;
    }
    const B*
    operator&() const
    {
        return nullptr;
    }
    B*
    operator&()
    {
        return nullptr;
    }
};

class C
{
    int data_;

  public:
    constexpr C() : data_(1) {}

    constexpr int
    get() const
    {
        return data_;
    }

    friend constexpr bool
    operator==(const C& x, const C& y)
    {
        return x.data_ == y.data_;
    }
};

constexpr C gC;

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
                    typedef dpl::reverse_iterator<const C*> RI;
                    constexpr RI it1 = dpl::make_reverse_iterator(&gC + 1);

                    static_assert(it1->get() == gC.get());
                }
                {
                    ((void)gC);
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
    EXPECT_TRUE(ret, "Wrong result of work with reverse iterator and reference in kernel_test()");

    return TestUtils::done();
}
