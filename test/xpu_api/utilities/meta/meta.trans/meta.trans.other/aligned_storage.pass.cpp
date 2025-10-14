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

#include "support/test_macros.h"
#include "support/utils.h"

bool
kernel_test()
{
    {
        typedef dpl::aligned_storage<10, 1>::type T1;
        ASSERT_SAME_TYPE(T1, dpl::aligned_storage_t<10, 1>);

#if TEST_STD_VER == 17
        static_assert(dpl::is_pod<T1>::value);
#endif
        static_assert(dpl::is_trivial<T1>::value);
        static_assert(dpl::is_standard_layout<T1>::value);
        static_assert(dpl::alignment_of<T1>::value == 1);
        static_assert(sizeof(T1) == 10);
    }
    {
        typedef dpl::aligned_storage<10, 2>::type T1;
        ASSERT_SAME_TYPE(T1, dpl::aligned_storage_t<10, 2>);

#if TEST_STD_VER == 17
        static_assert(dpl::is_pod<T1>::value);
#endif
        static_assert(dpl::is_trivial<T1>::value);
        static_assert(dpl::is_standard_layout<T1>::value);
        static_assert(dpl::alignment_of<T1>::value == 2);
        static_assert(sizeof(T1) == 10);
    }
    {
        typedef dpl::aligned_storage<10, 4>::type T1;
        ASSERT_SAME_TYPE(T1, dpl::aligned_storage_t<10, 4>);

#if TEST_STD_VER == 17
        static_assert(dpl::is_pod<T1>::value);
#endif
        static_assert(dpl::is_trivial<T1>::value);
        static_assert(dpl::is_standard_layout<T1>::value);
        static_assert(dpl::alignment_of<T1>::value == 4);
        static_assert(sizeof(T1) == 12);
    }
    {
        typedef dpl::aligned_storage<10, 8>::type T1;
        ASSERT_SAME_TYPE(T1, dpl::aligned_storage_t<10, 8>);

#if TEST_STD_VER == 17
        static_assert(dpl::is_pod<T1>::value);
#endif
        static_assert(dpl::is_trivial<T1>::value);
        static_assert(dpl::is_standard_layout<T1>::value);
        static_assert(dpl::alignment_of<T1>::value == 8);
        static_assert(sizeof(T1) == 16);
    }
    return true;
}
class KernelTest;

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with dpl::aligned_storage");

    return TestUtils::done();
}
