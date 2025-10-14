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
#include <oneapi/dpl/utility>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

class KernelGetConstRvTest;

#if !_PSTL_TEST_GCC7_RVALUE_TUPLE_GET_BROKEN
void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelGetConstRvTest>([=]() {
            {
                const dpl::tuple<int> t(3);
                static_assert(dpl::is_same<const int&&, decltype(dpl::get<0>(dpl::move(t)))>::value);
                ASSERT_NOEXCEPT(dpl::get<0>(dpl::move(t)));
                const int&& i = dpl::get<0>(dpl::move(t));
                ret_access[0] = (i == 3);
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::get(dpl::tuple&&) check");
}
#endif // !_PSTL_TEST_GCC7_RVALUE_TUPLE_GET_BROKEN

int
main()
{
    bool bProcessed = false;

#if !_PSTL_TEST_GCC7_RVALUE_TUPLE_GET_BROKEN
    kernel_test();
    bProcessed = true;
#endif // !_PSTL_TEST_GCC7_RVALUE_TUPLE_GET_BROKEN

    return TestUtils::done(bProcessed);
}
