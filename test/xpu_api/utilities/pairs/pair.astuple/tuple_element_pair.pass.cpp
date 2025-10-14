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

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

template <typename T1, typename T2>
bool
test()
{

    bool result = false;
    {
        typedef dpl::pair<T1, T2> P;
        static_assert(dpl::is_same<typename std::tuple_element<0, P>::type, T1>::value);
        static_assert(dpl::is_same<typename std::tuple_element<1, P>::type, T2>::value);

        static_assert(dpl::is_same<std::tuple_element_t<0, P>, T1>::value);
        static_assert(dpl::is_same<std::tuple_element_t<1, P>, T2>::value);

        result = (dpl::is_same<typename std::tuple_element<0, P>::type, T1>::value);
        result &= (dpl::is_same<typename std::tuple_element<1, P>::type, T2>::value);
    }
    {
        typedef dpl::pair<T1, T2> const P;
        static_assert(dpl::is_same<typename std::tuple_element<0, P>::type, const T1>::value);
        static_assert(dpl::is_same<typename std::tuple_element<1, P>::type, const T2>::value);

        static_assert(dpl::is_same<std::tuple_element_t<0, P>, const T1>::value);
        static_assert(dpl::is_same<std::tuple_element_t<1, P>, const T2>::value);

        result &= (dpl::is_same<typename std::tuple_element<0, P>::type, const T1>::value);
        result &= (dpl::is_same<typename std::tuple_element<1, P>::type, const T2>::value);
    }
    {
        typedef dpl::pair<T1, T2> volatile P;
        static_assert(dpl::is_same<typename std::tuple_element<0, P>::type, volatile T1>::value);
        static_assert(dpl::is_same<typename std::tuple_element<1, P>::type, volatile T2>::value);

        static_assert(dpl::is_same<std::tuple_element_t<0, P>, volatile T1>::value);
        static_assert(dpl::is_same<std::tuple_element_t<1, P>, volatile T2>::value);

        result &= (dpl::is_same<typename std::tuple_element<0, P>::type, volatile T1>::value);
        result &= (dpl::is_same<typename std::tuple_element<1, P>::type, volatile T2>::value);
    }
    {
        typedef dpl::pair<T1, T2> const volatile P;
        static_assert(dpl::is_same<typename std::tuple_element<0, P>::type, const volatile T1>::value);
        static_assert(dpl::is_same<typename std::tuple_element<1, P>::type, const volatile T2>::value);

        static_assert(dpl::is_same<std::tuple_element_t<0, P>, const volatile T1>::value);
        static_assert(dpl::is_same<std::tuple_element_t<1, P>, const volatile T2>::value);

        result &= (dpl::is_same<typename std::tuple_element<0, P>::type, const volatile T1>::value);
        result &= (dpl::is_same<typename std::tuple_element<1, P>::type, const volatile T2>::value);
    }

    return result;
}

class KernelPairTest1;
class KernelPairTest2;

template <typename T1, typename T2, typename KC>
void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<KC>([=]() { ret_access[0] = test<T1, T2>(); });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::pair elements check: should be std::tuple_element");
}

int
main()
{
    kernel_test<int, short, KernelPairTest1>();
    kernel_test<int*, char, KernelPairTest2>();

    return TestUtils::done();
}
