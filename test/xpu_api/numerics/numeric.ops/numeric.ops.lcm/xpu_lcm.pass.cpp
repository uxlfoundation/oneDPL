// -*- C++ -*-
//===-- xpu_lcm.pass.cpp --------------------------------------------===//
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

#include <oneapi/dpl/numeric>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

#include <cassert>
#include <iostream>

using oneapi::dpl::is_same_v;
using oneapi::dpl::lcm;

template <typename T1, typename T2>
class KernelName;

template <typename Input1, typename Input2, typename Output>
bool
test0(int in1, int in2, int out)
{
    auto value1 = static_cast<Input1>(in1);
    auto value2 = static_cast<Input2>(in2);
    static_assert(is_same_v<Output, decltype(lcm(value1, value2))>);
    static_assert(is_same_v<Output, decltype(lcm(value2, value1))>);
    return static_cast<Output>(out) == lcm(value1, value2);
}

template <typename Input1, typename Input2 = Input1>
void
do_test(sycl::queue& deviceQueue)
{
    bool res = true;
    sycl::range<1> numOfItems1{1};

    {
        sycl::buffer<bool, 1> buffer1(&res, numOfItems1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto out = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelName<Input1, Input2>>([=]() {
                constexpr struct
                {
                    int x;
                    int y;
                    int expect;
                } cases[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0},   {1, 1, 1},
                             {2, 3, 6}, {2, 4, 4}, {3, 17, 51}, {36, 18, 36}};
                using S1 = oneapi::dpl::make_signed_t<Input1>;
                using S2 = oneapi::dpl::make_signed_t<Input2>;
                using U1 = oneapi::dpl::make_unsigned_t<Input1>;
                using U2 = oneapi::dpl::make_unsigned_t<Input2>;
                for (auto tc : cases)
                {
                    { // Test with two signed types
                        using Output = oneapi::dpl::common_type_t<S1, S2>;
                        out[0] &= test0<S1, S2, Output>(tc.x, tc.y, tc.expect);
                        out[0] &= test0<S1, S2, Output>(-tc.x, tc.y, tc.expect);
                        out[0] &= test0<S1, S2, Output>(tc.x, -tc.y, tc.expect);
                        out[0] &= test0<S1, S2, Output>(-tc.x, -tc.y, tc.expect);
                        out[0] &= test0<S2, S1, Output>(tc.x, tc.y, tc.expect);
                        out[0] &= test0<S2, S1, Output>(-tc.x, tc.y, tc.expect);
                        out[0] &= test0<S2, S1, Output>(tc.x, -tc.y, tc.expect);
                        out[0] &= test0<S2, S1, Output>(-tc.x, -tc.y, tc.expect);
                    }

                    { // test with two unsigned types
                        using Output = oneapi::dpl::common_type_t<U1, U2>;
                        out[0] &= test0<U1, U2, Output>(tc.x, tc.y, tc.expect);
                        out[0] &= test0<U2, U1, Output>(tc.x, tc.y, tc.expect);
                    }
                    { // Test with mixed signs
                        using Output = oneapi::dpl::common_type_t<S1, U2>;
                        out[0] &= test0<S1, U2, Output>(tc.x, tc.y, tc.expect);
                        out[0] &= test0<U2, S1, Output>(tc.x, tc.y, tc.expect);
                        out[0] &= test0<S1, U2, Output>(-tc.x, tc.y, tc.expect);
                        out[0] &= test0<U2, S1, Output>(tc.x, -tc.y, tc.expect);
                    }
                    { // Test with mixed signs
                        using Output = oneapi::dpl::common_type_t<S2, U1>;
                        out[0] &= test0<S2, U1, Output>(tc.x, tc.y, tc.expect);
                        out[0] &= test0<U1, S2, Output>(tc.x, tc.y, tc.expect);
                        out[0] &= test0<S2, U1, Output>(-tc.x, tc.y, tc.expect);
                        out[0] &= test0<U1, S2, Output>(tc.x, -tc.y, tc.expect);
                    }
                }
                {
                    auto res1 = oneapi::dpl::lcm(static_cast<std::int64_t>(1234), INT32_MIN);
                    out[0] &= (res1 == 1324997410816LL);
                }
            });
        }).wait();
    }

    assert(res);
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();

// TODO: remove the macro guard once L0 backend fixes the issue
#if defined(_WIN32)
    return TestUtils::done(0);
#else
    do_test<signed char>(deviceQueue);
    do_test<short>(deviceQueue);
    do_test<int>(deviceQueue);
    do_test<long>(deviceQueue);
    do_test<long long>(deviceQueue);
    do_test<std::int8_t>(deviceQueue);
    do_test<std::int16_t>(deviceQueue);
    do_test<std::int32_t>(deviceQueue);
    do_test<std::int64_t>(deviceQueue);
    do_test<signed char, int>(deviceQueue);
    do_test<int, signed char>(deviceQueue);
    do_test<short, int>(deviceQueue);
    do_test<int, short>(deviceQueue);
    do_test<int, long>(deviceQueue);
    do_test<long, int>(deviceQueue);
    do_test<int, long long>(deviceQueue);
    do_test<long long, int>(deviceQueue);
#endif
    return TestUtils::done();
}
