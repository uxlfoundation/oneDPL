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

#ifndef _TEST_COMPLEX_H
#define _TEST_COMPLEX_H

#include "test_config.h"

#include <oneapi/dpl/complex>

#include "utils.h"
#include "utils_invoke.h"

#include <type_traits>
#include <cassert>
#include <limits>

#if !_PSTL_MSVC_LESS_THAN_CPP20_COMPLEX_CONSTEXPR_BROKEN
#    define STD_COMPLEX_TESTS_STATIC_ASSERT(arg) static_assert(arg)
#else
#    define STD_COMPLEX_TESTS_STATIC_ASSERT(arg) assert(arg)
#endif // !_PSTL_MSVC_LESS_THAN_CPP20_COMPLEX_CONSTEXPR_BROKEN

constexpr bool
is_fast_math_switched_on()
{
#if defined(__FAST_MATH__)
    return true;
#else
    return false;
#endif
}

#define ONEDPL_TEST_NUM_MAIN                                                                          \
template <typename HasDoubleSupportInRuntime, typename HasLongDoubleSupportInCompiletime>             \
int                                                                                                   \
run_test();                                                                                           \
                                                                                                      \
int main(int, char**)                                                                                 \
{                                                                                                     \
    static_assert(!is_fast_math_switched_on(),                                                        \
                  "Tests of std::complex are not compatible with -ffast-math compiler option.");      \
                                                                                                      \
    run_test<std::true_type, std::true_type>();                                                       \
                                                                                                      \
    /* Sometimes we may start test on device, which don't support type double. */                     \
    /* In this case generates run-time error.                                  */                     \
    /* This two types allow us to avoid this situation.                        */                     \
    using HasDoubleTypeSupportInRuntime = std::true_type;                                             \
    using HasntDoubleTypeSupportInRuntime = std::false_type;                                          \
                                                                                                      \
    /* long double type generate compile-time error in Kernel code             */                     \
    /* and we never can use this type inside Kernel                            */                     \
    using HasntLongDoubleSupportInCompiletime = std::false_type;                                      \
                                                                                                      \
    TestUtils::run_test_in_kernel(                                                                    \
        /* lambda for the case when we have support of double type on device */                       \
        [&]() { run_test<HasDoubleTypeSupportInRuntime, HasntLongDoubleSupportInCompiletime>(); },    \
        /* lambda for the case when we haven't support of double type on device */                    \
        [&]() { run_test<HasntDoubleTypeSupportInRuntime, HasntLongDoubleSupportInCompiletime>(); }); \
                                                                                                      \
    return TestUtils::done();                                                                         \
}                                                                                                     \
                                                                                                      \
template <typename HasDoubleSupportInRuntime, typename HasLongDoubleSupportInCompiletime>             \
int                                                                                                   \
run_test()

// We should use this macros to avoid runtime-error if type double doesn't supported on device.
//
// Example:
//     template <class T>
//     void
//     test(T x, std::enable_if_t<std::is_integral_v<T>>* = 0)
//     {
//         static_assert((std::is_same_v<decltype(dpl::conj(x)), dpl::complex<double>>));
//
//         // HERE IS THE CODE WHICH CALL WE SHOULD AVOID IF DOUBLE IS NOT SUPPORTED ON DEVICE
//         assert(dpl::conj(x) == dpl::conj(dpl::complex<double>(x, 0)));
//     }
//
//     template <class T>
//     void test()
//     {
//         // ...
//         test<T>(1);
//         // ...
//     }
//
//     ONEDPL_TEST_NUM_MAIN
//     {
//         // ...
//         IF_DOUBLE_SUPPORT(test<int>())
//         // ...
//     }
#define IF_DOUBLE_SUPPORT(...)                                                                        \
    if constexpr (HasDoubleSupportInRuntime{})                                                        \
    {                                                                                                 \
        auto __fnc = []() { __VA_ARGS__; };                                                           \
        __fnc();                                                                                      \
    }

// We should use this macros to avoid compile-time error in code with long double type in Kernel.
#define IF_LONG_DOUBLE_SUPPORT(...)                                                                   \
    if constexpr (HasLongDoubleSupportInCompiletime{})                                                \
    {                                                                                                 \
        auto __fnc = []() { __VA_ARGS__; };                                                           \
        __fnc();                                                                                      \
    }

namespace TestUtils
{
    template <typename T>
    static constexpr float infinity_val = std::numeric_limits<T>::infinity();

    // Run test in Kernel as single task
    template <typename TFncDoubleHasSupportInRuntime, typename TFncDoubleHasntSupportInRuntime>
    void
    run_test_in_kernel(TFncDoubleHasSupportInRuntime fncDoubleHasSupportInRuntime,
                       TFncDoubleHasntSupportInRuntime fncDoubleHasntSupportInRuntime)
    {
#if TEST_DPCPP_BACKEND_PRESENT
        try
        {
            sycl::queue deviceQueue = TestUtils::get_test_queue();

            const auto device = deviceQueue.get_device();

            // We should run fncDoubleHasSupportInRuntime and fncDoubleHasntSupportInRuntime
            // in two separate Kernels to have ability compile these Kernels separately
            // by using Intel(R) oneAPI DPC++/C++ Compiler option -fsycl-device-code-split=per_kernel
            // which described at
            // https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/compilation/jitting.html
            if (has_type_support<double>(device))
            {
                deviceQueue.submit(
                    [&](sycl::handler& cgh) {
                        cgh.single_task<TestUtils::new_kernel_name<class TestType, 0>>(
                            [fncDoubleHasSupportInRuntime]() { fncDoubleHasSupportInRuntime(); });
                    });
            }
            else
            {
                deviceQueue.submit(
                    [&](sycl::handler& cgh) {
                        cgh.single_task<TestUtils::new_kernel_name<class TestType, 1>>(
                            [fncDoubleHasntSupportInRuntime]() { fncDoubleHasntSupportInRuntime(); });
                    });
            }
            deviceQueue.wait_and_throw();
        }
        catch (const std::exception& exc)
        {
            std::stringstream str;

            str << "Exception occurred";
            if (exc.what())
                str << " : " << exc.what();

            TestUtils::issue_error_message(str);
        }
#endif // TEST_DPCPP_BACKEND_PRESENT
    }

} /* namespace TestUtils */

#endif // _TEST_COMPLEX_H
