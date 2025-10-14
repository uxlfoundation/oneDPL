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

#include <oneapi/dpl/tuple>
#include <oneapi/dpl/functional>

#include "support/test_macros.h"
#include "support/utils.h"

class A
{
    int i_;
    char c_;

  public:
    A(int i, char c) : i_(i), c_(c) {}
    int
    get_i() const
    {
        return i_;
    }
    char
    get_c() const
    {
        return c_;
    }
};

using B = std::tuple<float, unsigned, unsigned>;

class KernelPairTest;
void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelPairTest>([=]() {
            dpl::pair<A, B> p(dpl::piecewise_construct, dpl::make_tuple(4, 'a'), dpl::make_tuple(3.5f, 6u, 2u));
            ret_access[0] = (p.first.get_i() == 4);
            ret_access[0] &= (p.first.get_c() == 'a');
            ret_access[0] &= (std::get<0>(p.second) == 3.5f);
            ret_access[0] &= (std::get<1>(p.second) == 6u);
            ret_access[0] &= (std::get<2>(p.second) == 2u);
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::piecewise_construct check");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
