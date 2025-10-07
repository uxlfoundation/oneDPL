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

#include "support/test_macros.h"
#include "support/utils.h"

using dpl::optional;

struct X
{
    int i_;

    constexpr X(int i) : i_(i) {}
};

constexpr bool
operator<=(const X& lhs, const X& rhs)
{
    return lhs.i_ <= rhs.i_;
}

bool
kernel_test()
{
    sycl::queue q = TestUtils::get_test_queue();
    bool ret = true;
    typedef optional<X> O;
    X val(2);
    O ia[3] = {O{}, O{1}, O{val}};
    sycl::range<1> numOfItems1{1};
    sycl::range<1> numOfItems2{3};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        sycl::buffer<O, 1> buffer2(ia, numOfItems2);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto ia_acc = buffer2.get_access<sycl::access::mode::read>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                {

                    ret_access[0] &= ((ia_acc[0] <= X(1)));
                    ret_access[0] &= ((ia_acc[1] <= X(1))); // equal
                    ret_access[0] &= (!(ia_acc[2] <= X(1)));
                    ret_access[0] &= ((ia_acc[1] <= val));
                    ret_access[0] &= ((ia_acc[2] <= val)); // equal
                    ret_access[0] &= ((ia_acc[2] <= X(3)));

                    ret_access[0] &= (!(X(1) <= ia_acc[0]));
                    ret_access[0] &= ((X(1) <= ia_acc[1])); // equal
                    ret_access[0] &= ((X(1) <= ia_acc[2]));
                    ret_access[0] &= (!(val <= ia_acc[1]));
                    ret_access[0] &= ((val <= ia_acc[2])); // equal
                    ret_access[0] &= (!(X(3) <= ia_acc[2]));
                }
                {
                    constexpr optional<int> o1(42);
                    static_assert(o1 <= 42l);
                    static_assert(!(101l <= o1));
                }
                {
                    constexpr optional<const int> o1(42);
                    static_assert(o1 <= 42);
                    static_assert(!(101 <= o1));
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
    EXPECT_TRUE(ret, "Wrong result of less value check");

    return TestUtils::done();
}
