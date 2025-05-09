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
#ifndef _UTILS_SYCL_H
#define _UTILS_SYCL_H

// File contains common utilities for SYCL that tests rely on

#include "test_config.h"

// Do not #include <algorithm>, because if we do we will not detect accidental dependencies.
#include <iterator>

#if TEST_DPCPP_BACKEND_PRESENT
#include "utils_sycl_defs.h"
#endif // TEST_DPCPP_BACKEND_PRESENT

#include _PSTL_TEST_HEADER(iterator)
#include "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h"
#include "iterator_utils.h"
#include "utils_invoke.h"
#include "utils_test_base.h"

#ifdef ONEDPL_USE_PREDEFINED_POLICIES
#  define TEST_USE_PREDEFINED_POLICIES ONEDPL_USE_PREDEFINED_POLICIES
#else
#  define TEST_USE_PREDEFINED_POLICIES 1
#endif
#include _PSTL_TEST_HEADER(execution)

namespace TestUtils
{

#define PRINT_DEBUG(message) ::TestUtils::print_debug(message)

inline void
print_debug(const char*
#if _ONEDPL_DEBUG_SYCL
                message
#endif
)
{
#if _ONEDPL_DEBUG_SYCL
    ::std::cout << message << ::std::endl;
#endif
}

// Check values in sequence
template <typename Iterator, typename T>
bool
check_values(Iterator first, Iterator last, const T& val)
{
    return ::std::all_of(first, last, [&val](const T& x) { return x == val; });
}

auto async_handler = [](sycl::exception_list ex_list) {
    for (auto& ex : ex_list)
    {
        try
        {
            ::std::rethrow_exception(ex);
        }
        catch (sycl::exception& ex)
        {
            ::std::cerr << ex.what() << ::std::endl;
            ::std::exit(EXIT_FAILURE);
        }
    }
};

#if ONEDPL_FPGA_DEVICE
inline auto default_selector =
#    if ONEDPL_FPGA_EMULATOR
        sycl::ext::intel::fpga_emulator_selector_v;
#    else
        sycl::ext::intel::fpga_selector{};
#    endif // ONEDPL_FPGA_EMULATOR

inline auto&& default_dpcpp_policy =
#    if TEST_USE_PREDEFINED_POLICIES
        oneapi::dpl::execution::dpcpp_fpga;
#    else
        TestUtils::make_fpga_policy(sycl::queue{default_selector});
#    endif
#else
inline auto default_selector =
#    if TEST_LIBSYCL_VERSION >= 60000
        sycl::default_selector_v;
#    else
        sycl::default_selector{};
#    endif
inline auto&& default_dpcpp_policy =
#    if TEST_USE_PREDEFINED_POLICIES
        oneapi::dpl::execution::dpcpp_default;
#    else
        TestUtils::make_device_policy(sycl::queue{default_selector});
#    endif
#endif     // ONEDPL_FPGA_DEVICE

inline
sycl::queue get_test_queue()
{
    try
    {
        // create the queue with custom asynchronous exceptions handler
        static sycl::queue my_queue(default_selector, async_handler);
        return my_queue;
    }
    catch (const std::exception& exc)
    {
        std::cerr << "Exception occurred in get_test_queue()";
        if (exc.what())
            std::cerr << ": " << exc.what();
        std::cerr << std::endl;

        throw;
    }
}

template <sycl::usm::alloc alloc_type>
constexpr bool
required_test_sycl_buffer()
{
    return alloc_type == sycl::usm::alloc::shared;
}

template <sycl::usm::alloc alloc_type, typename TestValueType, typename TestName,
          bool TestSyclBuffer = required_test_sycl_buffer<alloc_type>()>
void
test1buffer(float ScaleStep = 1.0f, float ScaleMax = 1.0f)
{
    sycl::queue queue = get_test_queue(); // usm and allocator requires queue
    const size_t local_max_n = max_n * ScaleMax;
    const size_t incr_by_one_max = 16 * ScaleMax;
    const size_t local_step = 3.1415 * ScaleStep;
#if _PSTL_SYCL_TEST_USM
    { // USM
        // 1. allocate usm memory
        using TestBaseData = test_base_data_usm<alloc_type, TestValueType>;
        TestBaseData test_base_data(queue, { { local_max_n, inout1_offset } });

        // 2. create a pointer at first+offset
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);

        // 3. run algorithms
        for (size_t n = 1; n <= local_max_n; n = n <= incr_by_one_max ? n + 1 : size_t(local_step * n))
        {
#    if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#    endif
            invoke_on_all_hetero_policies<0>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               n);
        }
    }
#endif

    if constexpr (TestSyclBuffer)
    {
        // sycl::buffer
        // 1. create buffers
        using TestBaseData = test_base_data_buffer<TestValueType>;
        TestBaseData test_base_data({ { local_max_n, inout1_offset } });

        // 2. create iterators over buffers
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);

        // 3. run algorithms
        for (size_t n = 1; n <= local_max_n; n = n <= incr_by_one_max ? n + 1 : size_t(local_step * n))
        {
#if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#endif
            invoke_on_all_hetero_policies<1>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               n);
        }
    }
}

template <sycl::usm::alloc alloc_type, typename TestValueType, typename TestName,
          bool TestSyclBuffer = required_test_sycl_buffer<alloc_type>()>
void
test2buffers(float ScaleStep = 1.0f, float ScaleMax = 1.0f)
{
    sycl::queue queue = get_test_queue(); // usm and allocator requires queue
    const size_t local_max_n = max_n * ScaleMax;
    const size_t incr_by_one_max = 16 * ScaleMax;
    const size_t local_step = 3.1415 * ScaleStep;

#if _PSTL_SYCL_TEST_USM
    { // USM
        // 1. allocate usm memory
        using TestBaseData = test_base_data_usm<alloc_type, TestValueType>;
        TestBaseData test_base_data(queue, { { local_max_n, inout1_offset },
                                             { local_max_n, inout2_offset } });

        // 2. create pointers at first+offset
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);
        auto inout2_offset_first = test_base_data.get_start_from(UDTKind::eVals);

        // 3. run algorithms
        for (size_t n = 1; n <= local_max_n; n = n <= incr_by_one_max ? n + 1 : size_t(local_step * n))
        {
#    if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#    endif
            invoke_on_all_hetero_policies<0>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n,
                                               n);
        }
    }
#endif

    if constexpr (TestSyclBuffer)
    {
        // sycl::buffer
        // 1. create buffers
        using TestBaseData = test_base_data_buffer<TestValueType>;
        TestBaseData test_base_data({ { local_max_n, inout1_offset },
                                      { local_max_n, inout2_offset } });

        // 2. create iterators over buffers
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);
        auto inout2_offset_first = test_base_data.get_start_from(UDTKind::eVals);

        // 3. run algorithms
        for (size_t n = 1; n <= local_max_n; n = n <= incr_by_one_max ? n + 1 : size_t(local_step * n))
        {
#if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#endif
            invoke_on_all_hetero_policies<1>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n,
                                               n);
        }
    }
}

template <sycl::usm::alloc alloc_type, typename TestValueType, typename TestName,
          bool TestSyclBuffer = required_test_sycl_buffer<alloc_type>()>
void
test3buffers(int mult = kDefaultMultValue, float ScaleStep = 1.0f, float ScaleMax = 1.0f)
{
    sycl::queue queue = get_test_queue(); // usm requires queue
    const size_t local_max_n = max_n * ScaleMax;
    const size_t incr_by_one_max = 16 * ScaleMax;
    const size_t local_step = 3.1415 * ScaleStep;

#if _PSTL_SYCL_TEST_USM
    { // USM

        // 1. allocate usm memory
        using TestBaseData = test_base_data_usm<alloc_type, TestValueType>;
        TestBaseData test_base_data(queue, { { local_max_n,        inout1_offset },
                                             { local_max_n,        inout2_offset },
                                             { local_max_n * mult, inout3_offset } });

        // 2. create pointers at first+offset
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);
        auto inout2_offset_first = test_base_data.get_start_from(UDTKind::eVals);
        auto inout3_offset_first = test_base_data.get_start_from(UDTKind::eRes);

        // 3. run algorithms
        for (size_t n = 1; n <= local_max_n; n = (n <= incr_by_one_max ? n + 1 : size_t(local_step * n)))
        {
#    if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#    endif
            invoke_on_all_hetero_policies<0>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n,
                                               inout3_offset_first, inout3_offset_first + n * mult,
                                               n);
        }
    }
#endif

    if constexpr (TestSyclBuffer)
    {
        // sycl::buffer
        // 1. create buffers
        using TestBaseData = test_base_data_buffer<TestValueType>;
        TestBaseData test_base_data({ { local_max_n,        inout1_offset },
                                      { local_max_n,        inout2_offset },
                                      { local_max_n * mult, inout3_offset } });

        // 2. create iterators over buffers
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);
        auto inout2_offset_first = test_base_data.get_start_from(UDTKind::eVals);
        auto inout3_offset_first = test_base_data.get_start_from(UDTKind::eRes);

        // 3. run algorithms
        for (size_t n = 1; n <= local_max_n; n = (n <= incr_by_one_max ? n + 1 : size_t(local_step * n)))
        {
#if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#endif
            invoke_on_all_hetero_policies<1>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n,
                                               inout3_offset_first, inout3_offset_first + n * mult,
                                               n);
        }
    }
}

template <sycl::usm::alloc alloc_type, typename TestValueType, typename TestName,
          bool TestSyclBuffer = required_test_sycl_buffer<alloc_type>()>
void
test4buffers(int mult = kDefaultMultValue, float ScaleStep = 1.0f, float ScaleMax = 1.0f)
{
    sycl::queue queue = get_test_queue(); // usm requires queue
    const size_t local_max_n = max_n * ScaleMax;
    const size_t incr_by_one_max = 16 * ScaleMax;
    const size_t local_step = 3.1415 * ScaleStep;

#if _PSTL_SYCL_TEST_USM
    { // USM

        // 1. allocate usm memory
        using TestBaseData = test_base_data_usm<alloc_type, TestValueType>;
        TestBaseData test_base_data(queue, { { local_max_n,        inout1_offset },
                                             { local_max_n,        inout2_offset },
                                             { local_max_n * mult, inout3_offset },
                                             { local_max_n * mult, inout4_offset } });

        // 2. create pointers at first+offset
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);
        auto inout2_offset_first = test_base_data.get_start_from(UDTKind::eVals);
        auto inout3_offset_first = test_base_data.get_start_from(UDTKind::eRes);
        auto inout4_offset_first = test_base_data.get_start_from(UDTKind::eRes2);

        // 3. run algorithms
        for (size_t n = 1; n <= local_max_n; n = (n <= incr_by_one_max ? n + 1 : size_t(local_step * n)))
        {
#    if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#    endif
            invoke_on_all_hetero_policies<0>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n,
                                               inout3_offset_first, inout3_offset_first + n * mult,
                                               inout4_offset_first, inout4_offset_first + n * mult,
                                               n);
        }
    }
#endif

    if constexpr (TestSyclBuffer)
    {
        // sycl::buffer
        // 1. create buffers
        using TestBaseData = test_base_data_buffer<TestValueType>;
        TestBaseData test_base_data({ { local_max_n,        inout1_offset },
                                      { local_max_n,        inout2_offset },
                                      { local_max_n * mult, inout3_offset },
                                      { local_max_n * mult, inout4_offset } });

        // 2. create iterators over buffers
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);
        auto inout2_offset_first = test_base_data.get_start_from(UDTKind::eVals);
        auto inout3_offset_first = test_base_data.get_start_from(UDTKind::eRes);
        auto inout4_offset_first = test_base_data.get_start_from(UDTKind::eRes2);

        // 3. run algorithms
        for (size_t n = 1; n <= local_max_n; n = (n <= incr_by_one_max ? n + 1 : size_t(local_step * n)))
        {
#if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#endif
            invoke_on_all_hetero_policies<1>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n,
                                               inout3_offset_first, inout3_offset_first + n * mult,
                                               inout4_offset_first, inout4_offset_first + n * mult,
                                               n);
        }
    }
}

template <sycl::usm::alloc alloc_type, typename TestName,
          bool TestSyclBuffer = required_test_sycl_buffer<alloc_type>()>
::std::enable_if_t<::std::is_base_of_v<test_base<typename TestName::UsedValueType>, TestName>>
test1buffer()
{
    test1buffer<alloc_type, typename TestName::UsedValueType, TestName, TestSyclBuffer>(TestName::ScaleStep, TestName::ScaleMax);
}

template <sycl::usm::alloc alloc_type, typename TestName,
          bool TestSyclBuffer = required_test_sycl_buffer<alloc_type>()>
::std::enable_if_t<::std::is_base_of_v<test_base<typename TestName::UsedValueType>, TestName>>
test2buffers()
{
    test2buffers<alloc_type, typename TestName::UsedValueType, TestName, TestSyclBuffer>(TestName::ScaleStep, TestName::ScaleMax);
}

template <sycl::usm::alloc alloc_type, typename TestName,
          bool TestSyclBuffer = required_test_sycl_buffer<alloc_type>()>
::std::enable_if_t<::std::is_base_of_v<test_base<typename TestName::UsedValueType>, TestName>>
test3buffers(int mult = kDefaultMultValue)
{
    test3buffers<alloc_type, typename TestName::UsedValueType, TestName, TestSyclBuffer>(mult, TestName::ScaleStep, TestName::ScaleMax);
}

template <sycl::usm::alloc alloc_type, typename TestName,
          bool TestSyclBuffer = required_test_sycl_buffer<alloc_type>()>
::std::enable_if_t<::std::is_base_of_v<test_base<typename TestName::UsedValueType>, TestName>>
test4buffers(int mult = kDefaultMultValue)
{
    test4buffers<alloc_type, typename TestName::UsedValueType, TestName, TestSyclBuffer>(mult, TestName::ScaleStep, TestName::ScaleMax);
}

} /* namespace TestUtils */
#endif // _UTILS_SYCL_H
