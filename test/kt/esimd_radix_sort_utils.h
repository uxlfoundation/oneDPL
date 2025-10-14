
// -*- C++ -*-
//===-- esimd_radix_sort_test_utils.h -----------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ESIMD_RADIX_SORT_TEST_UTILS_H
#define _ESIMD_RADIX_SORT_TEST_UTILS_H

#include <string>
#include <tuple>
#include <random>
#include <cmath>
#include <limits>
#include <iostream>
#include <cstdint>
#include <vector>
#include <type_traits>

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#ifndef LOG_TEST_INFO
#    define LOG_TEST_INFO 0
#endif

template <typename KernelParam, typename KeyT, typename ValueT = void>
bool can_run_test(sycl::queue q, KernelParam param)
{
    const auto max_slm_size = q.get_device().template get_info<sycl::info::device::local_mem_size>();
    std::size_t slm_alloc_size = sizeof(KeyT) * param.data_per_workitem * param.workgroup_size;
    if constexpr (!std::is_void_v<ValueT>)
        slm_alloc_size += sizeof(ValueT) * param.data_per_workitem * param.workgroup_size;

    // skip tests with error: LLVM ERROR: SLM size exceeds target limits
    // TODO: get rid of that check: it is useless for AOT case. Proper configuration must be provided at compile time.
    return slm_alloc_size < max_slm_size;
}

inline const std::vector<std::size_t> sort_sizes = {
    1,       6,         16,      43,        256,           316,           2048,
    5072,    8192,      14001,   1 << 14,   (1 << 14) + 1, 50000,         67543,
    100'000, 1 << 17,   179'581, 250'000,   1 << 18,       (1 << 18) + 1, 500'000,
    888'235, 1'000'000, 1 << 20, 10'000'000};

template <typename T, bool Order>
struct Compare : public std::less<T>
{
};

template <typename T>
struct Compare<T, false> : public std::greater<T>
{
};

template<bool Order>
struct CompareKey
{
    template<typename T, typename U>
    bool operator()(const T& lhs, const U& rhs) const
    {
        return std::get<0>(lhs) < std::get<0>(rhs);
    }
};

template<>
struct CompareKey<false>
{
    template<typename T, typename U>
    bool operator()(const T& lhs, const U& rhs) const
    {
        return std::get<0>(lhs) > std::get<0>(rhs);
    }
};

constexpr bool Ascending = true;
constexpr bool Descending = false;
constexpr std::uint8_t TestRadixBits = 8;

#if LOG_TEST_INFO
struct TypeInfo
{
    template <typename T>
    const std::string&
    name()
    {
        static const std::string TypeName = "unknown type name";
        return TypeName;
    }

    template <>
    const std::string&
    name<char>()
    {
        static const std::string TypeName = "char";
        return TypeName;
    }

    template <>
    const std::string&
    name<int8_t>()
    {
        static const std::string TypeName = "int8_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<uint8_t>()
    {
        static const std::string TypeName = "uint8_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<int16_t>()
    {
        static const std::string TypeName = "int16_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<uint16_t>()
    {
        static const std::string TypeName = "uint16_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<uint32_t>()
    {
        static const std::string TypeName = "uint32_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<uint64_t>()
    {
        static const std::string TypeName = "uint64_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<int64_t>()
    {
        static const std::string TypeName = "int64_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<int>()
    {
        static const std::string TypeName = "int";
        return TypeName;
    }

    template <>
    const std::string&
    name<float>()
    {
        static const std::string TypeName = "float";
        return TypeName;
    }

    template <>
    const std::string&
    name<double>()
    {
        static const std::string TypeName = "double";
        return TypeName;
    }
};

struct USMAllocPresentation
{
    template <sycl::usm::alloc>
    const std::string&
    name()
    {
        static const std::string USMAllocTypeName = "unknown";
        return USMAllocTypeName;
    }

    template <>
    const std::string&
    name<sycl::usm::alloc::host>()
    {
        static const std::string USMAllocTypeName = "sycl::usm::alloc::host";
        return USMAllocTypeName;
    }

    template <>
    const std::string&
    name<sycl::usm::alloc::device>()
    {
        static const std::string USMAllocTypeName = "sycl::usm::alloc::device";
        return USMAllocTypeName;
    }

    template <>
    const std::string&
    name<sycl::usm::alloc::shared>()
    {
        static const std::string USMAllocTypeName = "sycl::usm::alloc::shared";
        return USMAllocTypeName;
    }

    template <>
    const std::string&
    name<sycl::usm::alloc::unknown>()
    {
        static const std::string USMAllocTypeName = "sycl::usm::alloc::unknown";
        return USMAllocTypeName;
    }
};
#endif // LOG_TEST_INFO

template <typename Container1, typename Container2>
void
print_data(const Container1& expected, const Container2& actual, std::size_t first, std::size_t n = 0)
{
    if (expected.size() <= first)
        return;
    if (n == 0 || expected.size() < first + n)
        n = expected.size() - first;

    if constexpr (std::is_floating_point_v<typename Container1::value_type>)
        std::cout << std::hexfloat;
    else
        std::cout << std::hex;

    for (std::size_t i = first; i < first + n; ++i)
    {
        std::cout << actual[i] << " --- " << expected[i] << std::endl;
    }

    if constexpr (std::is_floating_point_v<typename Container1::value_type>)
        std::cout << std::defaultfloat << std::endl;
    else
        std::cout << std::dec << std::endl;
}

#endif // _ESIMD_RADIX_SORT_TEST_UTILS_H
