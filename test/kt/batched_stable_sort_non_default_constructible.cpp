// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Keys (for merge_wg_resident_tag) and values (for every sort configuration) which are device copyable but not default
// constructible. Segment sizes leave partial sub-group blocks and tiles, whose padding slots must not need a value.

#include "../support/test_config.h"

#include <oneapi/dpl/experimental/kernel_templates>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include "../support/utils.h"
#include "../support/utils_sycl.h"

namespace kt = oneapi::dpl::experimental::kt;

struct Key
{
    std::uint32_t v;
    Key() = delete;
    explicit Key(std::uint32_t x) : v(x) {}
};

struct Value
{
    std::uint32_t idx;
    Value() = delete;
    explicit Value(std::uint32_t x) : idx(x) {}
};

static_assert(!std::is_default_constructible_v<Key> && !std::is_default_constructible_v<Value>);
static_assert(sycl::is_device_copyable_v<Key> && sycl::is_device_copyable_v<Value>);

// Orders keys by their value modulo 7 only, so many distinct keys are equivalent and stability is visible in the keys
struct KeyMod7Less
{
    bool
    operator()(const Key& lhs, const Key& rhs) const
    {
        return lhs.v % 7 < rhs.v % 7;
    }
};

inline std::uint32_t
key_id(const Key& k)
{
    return k.v;
}

inline std::uint32_t
key_id(std::uint32_t k)
{
    return k;
}

// Few distinct keys, so stability is also visible through the values, which hold the original index
template <typename KeyT>
std::vector<KeyT>
make_keys(std::size_t n)
{
    std::vector<KeyT> keys;
    keys.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        keys.emplace_back(std::uint32_t((i * 2654435761u) >> 7) % 50);
    return keys;
}

inline std::vector<Value>
make_indices(std::size_t n)
{
    std::vector<Value> vals;
    vals.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        vals.emplace_back(std::uint32_t(i));
    return vals;
}

template <typename T>
std::vector<T>
copy_to_host(sycl::queue q, const T* ptr, std::size_t n)
{
    // The host vector needs constructed elements to copy into
    std::vector<T> out(n, T(0u));
    q.copy(ptr, out.data(), n).wait();
    return out;
}

template <typename T>
T*
copy_to_device(sycl::queue q, const std::vector<T>& v)
{
    T* ptr = sycl::malloc_device<T>(v.size(), q);
    q.copy(v.data(), ptr, v.size()).wait();
    return ptr;
}

// Sorts n / segment_size segments of keys (and indices as values) and checks them against a per-segment
// std::stable_sort with HostLess
template <typename SortConfig, typename KernelParam, typename KeyT, bool ByKey, typename HostLess>
void
test_case(sycl::queue q, std::size_t segment_size, std::size_t segment_count, HostLess less)
{
    const std::size_t n = segment_size * segment_count;
    const std::vector<KeyT> keys_in = make_keys<KeyT>(n);
    const std::vector<Value> vals_in = make_indices(n);

    KeyT* d_keys = copy_to_device(q, keys_in);
    Value* d_vals = copy_to_device(q, vals_in);
    if constexpr (ByKey)
        kt::gpu::batched_stable_sort_by_key<SortConfig>(q, d_keys, d_keys + n, d_vals, segment_size, KernelParam{})
            .wait();
    else
        kt::gpu::batched_stable_sort<SortConfig>(q, d_keys, d_keys + n, segment_size, KernelParam{}).wait();
    const std::vector<KeyT> keys_out = copy_to_host(q, d_keys, n);
    const std::vector<Value> vals_out = copy_to_host(q, d_vals, n);
    sycl::free(d_keys, q);
    sycl::free(d_vals, q);

    // The expected order, as indices into the input
    std::vector<std::uint32_t> order(n);
    for (std::size_t i = 0; i < n; ++i)
        order[i] = std::uint32_t(i);
    for (std::size_t begin = 0; begin < n; begin += segment_size)
        std::stable_sort(order.begin() + begin, order.begin() + begin + segment_size,
                         [&](std::uint32_t a, std::uint32_t b) { return less(keys_in[a], keys_in[b]); });

    bool ok = true;
    for (std::size_t i = 0; i < n && ok; ++i)
    {
        ok = key_id(keys_out[i]) == key_id(keys_in[order[i]]);
        if constexpr (ByKey)
            ok = ok && vals_out[i].idx == order[i];
    }
    const std::string what = std::string(ByKey ? "by key" : "keys only") +
                             ", segment size: " + std::to_string(segment_size) +
                             ", segment count: " + std::to_string(segment_count);
    EXPECT_TRUE(ok, what.c_str());
}

template <typename SortConfig, typename KernelParam, typename KeyT, bool ByKey, typename HostLess>
void
test_sizes(sycl::queue q, const std::vector<std::size_t>& segment_sizes, HostLess less)
{
    for (std::size_t segment_size : segment_sizes)
        for (std::size_t segment_count : {std::size_t(1), std::size_t(3), std::size_t(64)})
            test_case<SortConfig, KernelParam, KeyT, ByKey>(q, segment_size, segment_count, less);
}

int
main()
{
    auto q = TestUtils::get_test_queue();
    try
    {
        // Tile of 1024 elements; sizes leave partial sub-group blocks (32 * 4 elements) and partial work-items
        using WgResidentParam = kt::kernel_param<4, 256>;
        const std::vector<std::size_t> wg_resident_sizes = {1, 7, 100, 317, 1000, 1024};
        // Tile of 2048 elements; sizes leave partial tiles, within a single tile and after full ones
        using MultiWgParam = kt::kernel_param<4, 512>;
        const std::vector<std::size_t> multi_wg_sizes = {1, 7, 1000, 2048, 5000};

        auto key_less = [](const Key& a, const Key& b) { return KeyMod7Less{}(a, b); };
        auto bits_less = [](std::uint32_t a, std::uint32_t b) { return a < b; };

        using MergeConfig = kt::gpu::merge_wg_resident_tag<KeyMod7Less>;
        test_sizes<MergeConfig, WgResidentParam, Key, /*ByKey=*/false>(q, wg_resident_sizes, key_less);
        test_sizes<MergeConfig, WgResidentParam, Key, /*ByKey=*/true>(q, wg_resident_sizes, key_less);

#if _ONEDPL_ENABLE_SYCL_WG_RESIDENT_RADIX_SORT_KT
        test_sizes<kt::gpu::radix_wg_resident_tag<>, WgResidentParam, std::uint32_t, /*ByKey=*/true>(
            q, wg_resident_sizes, bits_less);
#endif

#if _ONEDPL_ENABLE_SYCL_ONESWEEP_KT
        test_sizes<kt::gpu::radix_multi_wg_tag<>, MultiWgParam, std::uint32_t, /*ByKey=*/true>(q, multi_wg_sizes,
                                                                                               bits_less);
#endif
    }
    catch (const std::exception& exc)
    {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return EXIT_FAILURE;
    }
    return TestUtils::done();
}
