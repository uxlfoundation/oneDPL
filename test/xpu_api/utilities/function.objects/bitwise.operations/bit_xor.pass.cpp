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
 1) Warning  'first_argument_type' is deprecated: warning STL4007: Many result_type typedefs and all argument_type, first_argument_type, and second_argument_type typedefs are deprecated in C++17.
    You can define _SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING or _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS to suppress this warning.
 2) Warning  'second_argument_type' is deprecated: warning STL4007: Many result_type typedefs and all argument_type, first_argument_type, and second_argument_type typedefs are deprecated in C++17.
    You can define _SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING or _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS to suppress this warning.
 */
#define _SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING

#include "support/test_config.h"

#include <oneapi/dpl/functional>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"
#include "support/test_macros.h"

class KernelBitXorTest;

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
        cgh.single_task<class KernelBitXorTest>([=]() {
            {
                typedef dpl::bit_xor<int> F;
                const F f = F();
#if TEST_STD_VER < 20
                static_assert(dpl::is_same<int, F::first_argument_type>::value);
                static_assert(dpl::is_same<int, F::second_argument_type>::value);
                static_assert(dpl::is_same<int, F::result_type>::value);
#endif // TEST_STD_VER < 20
                ret_access[0] = (f(0xEA95, 0xEA95) == 0);
                ret_access[0] &= (f(0xEA95, 0x58D3) == 0xB246);
                ret_access[0] &= (f(0x58D3, 0xEA95) == 0xB246);
                ret_access[0] &= (f(0x58D3, 0) == 0x58D3);
                ret_access[0] &= (f(0xFFFF, 0x58D3) == 0xA72C);
            }

            {
                const dpl::bit_xor<long> f;
                ret_access[0] &= (f(0xEA95L, 0xEA95) == 0);
                ret_access[0] &= (f(0xEA95, 0xEA95L) == 0);

                ret_access[0] &= (f(0xEA95L, 0x58D3) == 0xB246);
                ret_access[0] &= (f(0xEA95, 0x58D3L) == 0xB246);

                ret_access[0] &= (f(0x58D3L, 0xEA95) == 0xB246);
                ret_access[0] &= (f(0x58D3, 0xEA95L) == 0xB246);

                ret_access[0] &= (f(0x58D3L, 0) == 0x58D3);
                ret_access[0] &= (f(0x58D3, 0L) == 0x58D3);

                ret_access[0] &= (f(0xFFFF, 0x58D3) == 0xA72C);
                ret_access[0] &= (f(0xFFFFL, 0x58D3) == 0xA72C);
                ret_access[0] &= (f(0xFFFF, 0x58D3L) == 0xA72C);
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with dpl::bit_xor");

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
