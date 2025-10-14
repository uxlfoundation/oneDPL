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

/*
 1) Warning  'argument_type' is deprecated: warning STL4007: Many result_type typedefs and all argument_type, first_argument_type, and second_argument_type typedefs are deprecated in C++17.
    You can define _SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING or _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS to suppress this warning.
 2) Warning  'result_type' is deprecated: warning STL4007: Many result_type typedefs and all argument_type, first_argument_type, and second_argument_type typedefs are deprecated in C++17.
    You can define _SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING or _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS to suppress this warning.
 */
#define _SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING

#include "support/test_config.h"

#include <oneapi/dpl/functional>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"
#include "support/test_macros.h"

class KernelBitNotTest;

void
kernel_test()
{
#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelBitNotTest>([=]() {
            typedef dpl::bit_not<int> F;
            const F f = F();
#if TEST_STD_VER < 20
            static_assert(dpl::is_same<F::argument_type, int>::value);
            static_assert(dpl::is_same<F::result_type, int>::value);
#endif // TEST_STD_VER < 20
            ret_access[0] = ((f(0xEA95) & 0xFFFF) == 0x156A);
            ret_access[0] &= ((f(0x58D3) & 0xFFFF) == 0xA72C);
            ret_access[0] &= ((f(0) & 0xFFFF) == 0xFFFF);
            ret_access[0] &= ((f(0xFFFF) & 0xFFFF) == 0);

            const dpl::bit_not<long> f2;
            ret_access[0] &= ((f2(0xEA95L) & 0xFFFF) == 0x156A);
            ret_access[0] &= ((f2(0x58D3L) & 0xFFFF) == 0xA72C);
            ret_access[0] &= ((f2(0L) & 0xFFFF) == 0xFFFF);
            ret_access[0] &= ((f2(0xFFFFL) & 0xFFFF) == 0);
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with dpl::bit_not");

#ifdef __clang__
#    pragma clang diagnostic pop
#endif
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
