
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
#include <algorithm>

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#ifndef LOG_TEST_INFO
#    define LOG_TEST_INFO 0
#endif

// Check if SYCL radix sort KT is available (matches criteria from radix_sort_utils.h in include/)
#if defined(SYCL_EXT_ONEAPI_FORWARD_PROGRESS) && defined(SYCL_EXT_ONEAPI_ROOT_GROUP) &&                                \
    (!defined(__INTEL_LLVM_COMPILER) || __INTEL_LLVM_COMPILER >= 20250100)
#    define TEST_SYCL_RADIX_SORT_KT_AVAILABLE 1
#endif

// Namespace aliases for kernel template APIs
#ifdef TEST_KT_BACKEND_ESIMD
namespace kt_ns = oneapi::dpl::experimental::kt::gpu::esimd;
namespace kt_deprecated_ns = oneapi::dpl::experimental::kt::esimd;
#elif defined(TEST_KT_BACKEND_SYCL)
namespace kt_ns = oneapi::dpl::experimental::kt::gpu;
#endif

// Helper to calculate SLM usage based on backend
template <typename KernelParam, typename KeyT, typename ValueT = void>
std::size_t
calculate_slm_size(KernelParam param)
{
#ifdef TEST_KT_BACKEND_ESIMD
    std::size_t slm_alloc_size = sizeof(KeyT) * param.data_per_workitem * param.workgroup_size;
    if constexpr (!std::is_void_v<ValueT>)
        slm_alloc_size += sizeof(ValueT) * param.data_per_workitem * param.workgroup_size;
    return slm_alloc_size;
#else // defined(TEST_KT_BACKEND_SYCL)
    using LocOffsetT = std::uint16_t;
    using GlobOffsetT = std::uint32_t;

    constexpr std::uint32_t radix_bits = 8;
    constexpr std::uint32_t bin_count = 1 << radix_bits;
    constexpr std::uint32_t sub_group_size = 32;

    const std::uint32_t num_sub_groups = param.workgroup_size / sub_group_size;
    const std::uint32_t work_item_all_hists_size = num_sub_groups * bin_count * sizeof(LocOffsetT);
    const std::uint32_t group_hist_size = bin_count * sizeof(LocOffsetT);
    const std::uint32_t global_hist_size = bin_count * sizeof(GlobOffsetT);

    std::uint32_t reorder_size = sizeof(KeyT) * param.data_per_workitem * param.workgroup_size;
    if constexpr (!std::is_void_v<ValueT>)
        reorder_size += sizeof(ValueT) * param.data_per_workitem * param.workgroup_size;

    const std::uint32_t slm_size =
        std::max(work_item_all_hists_size, reorder_size) + group_hist_size + global_hist_size;

    return slm_size;
#endif
}

template <typename KernelParam, typename KeyT, typename ValueT = void>
bool
can_run_test(sycl::queue q, KernelParam param)
{
    const auto max_slm_size = q.get_device().template get_info<sycl::info::device::local_mem_size>();
    std::size_t slm_alloc_size = calculate_slm_size<KernelParam, KeyT, ValueT>(param);

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

template <bool Order>
struct CompareKey
{
    template <typename T, typename U>
    bool
    operator()(const T& lhs, const U& rhs) const
    {
        return std::get<0>(lhs) < std::get<0>(rhs);
    }
};

template <>
struct CompareKey<false>
{
    template <typename T, typename U>
    bool
    operator()(const T& lhs, const U& rhs) const
    {
        return std::get<0>(lhs) > std::get<0>(rhs);
    }
};

constexpr bool Ascending = true;
constexpr bool Descending = false;
constexpr std::uint8_t TestRadixBits = 8;

// Fill `data` with distinct values from a narrow range so that the upper radix stages observe a single
// bin and exercise the single-bin direct-copy optimization, while the lower stages still reorder (forcing
// the optimization to scatter distinct keys to distinct output positions). Integral types use [0, 1000].
//
// Floating-point types use [1.0, 1.2). The IEEE-754 layout is [sign | exponent | mantissa]:
//   - sign     = 0           (all values positive)
//   - exponent = bias        (encodes 2^0, since 1.0 <= x < 2.0)
//   - mantissa: 1.0 <= x < 1.2 means the fraction is in [0, 0.2), so the top two mantissa bits are 00
//               and only the lower mantissa bits vary.
// This fixes the sign, the entire exponent, and the top two mantissa bits, so the most-significant byte is
// constant across all three IEEE formats:
//   - float64: 1 sign + 11 exp bits -> MSB holds sign + 7 exp bits (all fixed); next byte holds 4 exp + top
//              4 mantissa bits (top 2 fixed as 00). Top stage (top byte) is single-bin.
//   - float32: 1 sign + 8 exp bits  -> MSB holds sign + 7 exp bits (all fixed). Top stage is single-bin.
//   - sycl::half (float16): 1 sign + 5 exp bits + 10 mantissa bits -> MSB holds sign + 5 exp + top 2
//              mantissa bits (top 2 fixed as 00). Top stage is single-bin.
// The upper bound is 1.2 rather than 1.25 because values are generated in float and narrowed to T: with
// sycl::half's 10-bit mantissa, floats just under 1.25 round up to exactly 1.25, whose top two mantissa
// bits are 01, spilling into a second top-stage bin. 1.2 stays below that rounding cliff for all three types.
// All generated values are finite (no NaN/inf).
template <typename T>
void
generate_constrained_range_data(T* data, std::size_t size, std::uint32_t seed)
{
    std::default_random_engine gen{seed};
    if constexpr (std::is_integral_v<T>)
    {
        // For single byte type, just use a constant value and only vart first stage bits for other
        // types.
        std::uniform_int_distribution<std::uint32_t> dist;
        if constexpr (sizeof(T) == 8)
        {
            dist = std::uniform_int_distribution<std::uint32_t>{0, 42};
        }
        else
        {
            dist = std::uniform_int_distribution<std::uint32_t>{0, 256};
        }
        std::generate(data, data + size, [&] { return static_cast<T>(dist(gen)); });
    }
    else
    {
        // sycl::half is not directly usable with std::uniform_real_distribution; generate in float and cast.
        std::uniform_real_distribution<float> dist(1.0f, 1.2f);
        std::generate(data, data + size, [&] { return static_cast<T>(dist(gen)); });
    }
}

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
