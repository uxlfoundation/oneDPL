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

#include <oneapi/dpl/cmath>

#include "support/utils.h"
#include "support/utils_invoke.h"

#include <iostream>
#include <vector>

template <typename KernelClass, typename Function, typename ValueType>
void
test(Function fnc, const std::vector<ValueType>& args, const char* message)
{
    const auto args_count = args.size();

    std::vector<ValueType> output(args_count);

    sycl::range<1> numOfItems1{args_count};
    sycl::range<1> numOfItems2{args_count};

    // Evaluate results in Kernel
    {
        sycl::queue deviceQueue = TestUtils::get_test_queue();

        if (!TestUtils::has_type_support<ValueType>(deviceQueue.get_device()))
        {
            std::cout << deviceQueue.get_device().template get_info<sycl::info::device::name>() << " does not support "
                      << typeid(ValueType).name() << " type,"
                      << " affected test case have been skipped" << std::endl;
            return;
        }

        sycl::buffer<ValueType> buffer1(args.data(), args_count);
        sycl::buffer<ValueType> buffer2(output.data(), args_count);

        deviceQueue.submit([&](sycl::handler& cgh) {
            auto in = buffer1.template get_access<sycl::access::mode::read>(cgh);
            auto out = buffer2.template get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelClass>([=]() {
                for (size_t i = 0; i < args_count; ++i)
                    out[i] = fnc(in[i]);
            });
        });
    }

    // Check results: compare results evaluated in Kernel and on host
    for (size_t i = 0; i < args_count; ++i)
    {
        auto host_result = fnc(args[i]);
        EXPECT_EQ(host_result, output[i], message);
    }
}

class Test;

int
main()
{
    // functions from https://en.cppreference.com/w/cpp/numeric/math/nearbyint :
    //     float  nearbyint (float arg);
    //     float  nearbyintf(float arg);
    //     double nearbyint (double arg);

    ////////////////////////////////////////////////////////
    // float nearbyint(float arg);
    auto f_nearbyint_float = [](float arg) -> float { return oneapi::dpl::nearbyint(arg); };
    const std::vector<float> f_args_float = {+2.3, +2.5, +3.5, -2.3, -2.5, -3.5};
    test<TestUtils::unique_kernel_name<Test, 1>>(f_nearbyint_float, f_args_float, "float nearbyint(float)");

    ////////////////////////////////////////////////////////
    // float nearbyintf(float arg);
    auto f_nearbyintf_float = [](float arg) -> float { return oneapi::dpl::nearbyintf(arg); };
    test<TestUtils::unique_kernel_name<Test, 11>>(f_nearbyintf_float, f_args_float, "float nearbyintf(float)");

    ////////////////////////////////////////////////////////
    // double nearbyint(double arg);
    auto f_nearbyint_double = [](double arg) -> double { return oneapi::dpl::nearbyint(arg); };
    const std::vector<double> f_args_double = {+2.3, +2.5, +3.5, -2.3, -2.5, -3.5};
    test<TestUtils::unique_kernel_name<Test, 2>>(f_nearbyint_double, f_args_double, "double nearbyint(double)");

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
