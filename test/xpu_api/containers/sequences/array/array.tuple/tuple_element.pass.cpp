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
// tuple_element<I, array<T, N>>::type

#include "support/test_config.h"

#include <oneapi/dpl/array>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

class KernelTest1;

template <class T>
void
test()
{
    {
        typedef T Exp;
        typedef dpl::array<T, 3> C;
        static_assert(dpl::is_same<typename std::tuple_element<0, C>::type, Exp>::value);
        static_assert(dpl::is_same<typename std::tuple_element<1, C>::type, Exp>::value);
        static_assert(dpl::is_same<typename std::tuple_element<2, C>::type, Exp>::value);

        static_assert(dpl::is_same<std::tuple_element_t<0, C>, Exp>::value);
        static_assert(dpl::is_same<std::tuple_element_t<1, C>, Exp>::value);
        static_assert(dpl::is_same<std::tuple_element_t<2, C>, Exp>::value);
    }
    {
        typedef T const Exp;
        typedef dpl::array<T, 3> const C;
        static_assert(dpl::is_same<typename std::tuple_element<0, C>::type, Exp>::value);
        static_assert(dpl::is_same<typename std::tuple_element<1, C>::type, Exp>::value);
        static_assert(dpl::is_same<typename std::tuple_element<2, C>::type, Exp>::value);

        static_assert(dpl::is_same<std::tuple_element_t<0, C>, Exp>::value);
        static_assert(dpl::is_same<std::tuple_element_t<1, C>, Exp>::value);
        static_assert(dpl::is_same<std::tuple_element_t<2, C>, Exp>::value);
    }
    {
        typedef T volatile Exp;
        typedef dpl::array<T, 3> volatile C;
        static_assert(dpl::is_same<typename std::tuple_element<0, C>::type, Exp>::value);
        static_assert(dpl::is_same<typename std::tuple_element<1, C>::type, Exp>::value);
        static_assert(dpl::is_same<typename std::tuple_element<2, C>::type, Exp>::value);

        static_assert(dpl::is_same<std::tuple_element_t<0, C>, Exp>::value);
        static_assert(dpl::is_same<std::tuple_element_t<1, C>, Exp>::value);
        static_assert(dpl::is_same<std::tuple_element_t<2, C>, Exp>::value);
    }
    {
        typedef T const volatile Exp;
        typedef dpl::array<T, 3> const volatile C;
        static_assert(dpl::is_same<typename std::tuple_element<0, C>::type, Exp>::value);
        static_assert(dpl::is_same<typename std::tuple_element<1, C>::type, Exp>::value);
        static_assert(dpl::is_same<typename std::tuple_element<2, C>::type, Exp>::value);

        static_assert(dpl::is_same<std::tuple_element_t<0, C>, Exp>::value);
        static_assert(dpl::is_same<std::tuple_element_t<1, C>, Exp>::value);
        static_assert(dpl::is_same<std::tuple_element_t<2, C>, Exp>::value);
    }
}

int
main()
{
    bool ret = false;
    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{1});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto ret_acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelTest1>([=]() {
                test<float>();
                test<int>();
                ret_acc[0] = true;
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with std::tuple_element");

    return TestUtils::done();
}
