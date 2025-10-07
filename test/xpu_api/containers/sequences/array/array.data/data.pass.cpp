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

// <array>
// T *data();

#include "support/test_config.h"

#include <oneapi/dpl/array>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

class Test1;

struct NoDefault
{
    NoDefault() {}
    NoDefault(int) {}
};

int
main()
{
    bool ret = true;

    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{1});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto ret_acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<Test1>([=]() {
                {
                    typedef float T;
                    typedef dpl::array<T, 3> C;
                    C c = {1.f, 2.f, 3.5f};
                    T* p = c.data();
                    ret_acc[0] &= (p[0] == 1.f);
                    ret_acc[0] &= (p[1] == 2.f);
                    ret_acc[0] &= (p[2] == 3.5f);
                }
                {
                    typedef float T;
                    typedef dpl::array<T, 0> C;
                    C c = {};
                    [[maybe_unused]] T* p = c.data();
                }
                {
                    typedef float T;
                    typedef dpl::array<const T, 0> C;
                    C c = {{}};
                    [[maybe_unused]] const T* p = c.data();
                    static_assert(dpl::is_same<decltype(c.data()), const T*>::value);
                }
                {
                    typedef NoDefault T;
                    typedef dpl::array<T, 0> C;
                    C c = {};
                    [[maybe_unused]] T* p = c.data();
                }
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with dpl::array::data");

    return TestUtils::done();
}
