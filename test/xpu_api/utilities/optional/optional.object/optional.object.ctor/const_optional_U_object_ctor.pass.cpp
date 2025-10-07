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

#include "support/test_config.h"

#include <oneapi/dpl/optional>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

using dpl::optional;

template <class KernelTest, class T, class U>
bool
kernel_test(const optional<U>& rhs)
{
    sycl::queue q = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        sycl::buffer<optional<U>, 1> buffer2(&rhs, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto rhs_access = buffer2.template get_access<sycl::access::mode::read>(cgh);
            cgh.single_task<KernelTest>([=]() {
                bool rhs_engaged = static_cast<bool>(rhs_access[0]);
                optional<T> lhs = rhs_access[0];
                ret_access[0] &= (static_cast<bool>(lhs) == rhs_engaged);
                if (rhs_engaged)
                    ret_access[0] &= (*lhs == *rhs_access[0]);
            });
        });
    }
    return ret;
}

// TODO required to unify
class X
{
    int i_;

  public:
    X(int i) : i_(i) {}
    X(const X& x) : i_(x.i_) {}
    ~X() { i_ = 0; }
    friend bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_;
    }
};

// TODO required to unify
class Y
{
    int i_;

  public:
    Y(int i) : i_(i) {}

    friend constexpr bool
    operator==(const Y& x, const Y& y)
    {
        return x.i_ == y.i_;
    }
};

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;
class KernelTest6;

bool
test()
{
    bool ret = true;
    {
        optional<short> rhs;
        ret &= kernel_test<KernelTest1, int>(rhs);
    }
    {
        optional<short> rhs(short{3});
        ret &= kernel_test<KernelTest2, int>(rhs);
    }
    {
        optional<int> rhs;
        ret &= kernel_test<KernelTest3, X>(rhs);
    }
    {
        optional<int> rhs(int{3});
        ret &= kernel_test<KernelTest4, X>(rhs);
    }
    {
        optional<int> rhs;
        ret &= kernel_test<KernelTest5, Y>(rhs);
    }
    {
        optional<int> rhs(int{3});
        ret &= kernel_test<KernelTest6, Y>(rhs);
    }
    return ret;
}

int
main()
{
    auto ret = test();
    EXPECT_TRUE(ret, "Wrong result of dpl::optional and operator '==' check");

    return TestUtils::done();
}
