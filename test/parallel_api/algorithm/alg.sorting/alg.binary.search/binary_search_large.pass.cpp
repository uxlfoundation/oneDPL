// -*- C++ -*-
//===-- binary_search_large.pass.cpp --------------------------------------===//
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

// lower_bound / upper_bound / binary_search take a batched code path once the key count is large
// enough for __parallel_for's large submitter, which no other test reaches: the gate scales with the
// device's compute unit count, so on a large GPU it sits above a million keys. This test derives the
// gate from the device and sizes itself to cross it, over the value type sizes that select each
// distinct batch geometry, and over two key distributions.

#include "support/test_config.h"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include <algorithm>
#    include <cstdint>
#    include <iostream>
#    include <limits>
#    include <random>
#    include <string>
#    include <vector>

// A three-byte key, to select an iterations-per-item count that is not a multiple of the number of
// searches kept in flight, so that a batch and a shorter trailing batch both run.
struct Key3
{
    std::uint8_t __b[3];

    std::uint32_t
    value() const
    {
        return std::uint32_t(__b[0]) | (std::uint32_t(__b[1]) << 8) | (std::uint32_t(__b[2]) << 16);
    }
    bool
    operator<(const Key3& __o) const
    {
        return value() < __o.value();
    }
    bool
    operator==(const Key3& __o) const
    {
        return value() == __o.value();
    }
};

template <typename KeyT>
struct key_traits
{
    static constexpr std::uint32_t max_value = std::numeric_limits<KeyT>::max();
    static KeyT
    make(std::uint32_t __v)
    {
        return KeyT(__v);
    }
};

template <>
struct key_traits<Key3>
{
    static constexpr std::uint32_t max_value = (std::uint32_t(1) << 24) - 1;
    static Key3
    make(std::uint32_t __v)
    {
        return Key3{{std::uint8_t(__v), std::uint8_t(__v >> 8), std::uint8_t(__v >> 16)}};
    }
};

template <typename KeyT, typename ResT, int Idx>
class policy_name;

// How the keys are drawn. A uniform draw over the haystack's range leaves the classes that only a
// key outside that range produces -- a search that runs off either end of the haystack -- to a
// handful of lanes, so absent_heavy draws a quarter of the lanes past the last element and a quarter
// below the first, per lane, so that the classes mix within a work item's batch whatever its stride.
enum class key_mix
{
    uniform,
    absent_heavy
};

// Mirrors __parallel_for_large_submitter's dispatch gate.
std::size_t
batched_path_min_keys(sycl::queue __q, std::size_t __min_type_size)
{
    const std::size_t __wg =
        std::min<std::size_t>(512, __q.get_device().get_info<sycl::info::device::max_work_group_size>());
    const std::size_t __cu = __q.get_device().get_info<sycl::info::device::max_compute_units>();
    const std::size_t __iters_per_item = std::max<std::size_t>(1, 16 / __min_type_size);
    return __wg * __iters_per_item * __cu;
}

template <typename KeyT, typename ResT, typename Invoke>
void
run_and_check(const std::vector<KeyT>& __hay, const std::vector<KeyT>& __keys, const std::vector<ResT>& __ref,
              Invoke __invoke, const std::string& __what)
{
    std::vector<ResT> __out(__keys.size(), ResT(0));
    {
        sycl::buffer<KeyT> __hay_buf(const_cast<KeyT*>(__hay.data()), sycl::range<1>(__hay.size()));
        sycl::buffer<KeyT> __key_buf(const_cast<KeyT*>(__keys.data()), sycl::range<1>(__keys.size()));
        sycl::buffer<ResT> __out_buf(__out.data(), sycl::range<1>(__out.size()));

        auto __out_begin = oneapi::dpl::begin(__out_buf);
        auto __ret = __invoke(oneapi::dpl::begin(__hay_buf), oneapi::dpl::end(__hay_buf),
                             oneapi::dpl::begin(__key_buf), oneapi::dpl::end(__key_buf), __out_begin);
        EXPECT_EQ(std::ptrdiff_t(__keys.size()), std::distance(__out_begin, __ret),
                  (__what + ": wrong return value").c_str());
    }

    std::size_t __bad = 0;
    for (std::size_t __i = 0; __i != __keys.size(); ++__i)
    {
        if (__out[__i] != __ref[__i])
        {
            if (__bad == 0)
                std::cout << __what << ": first mismatch at " << __i << ", expected " << std::int64_t(__ref[__i])
                          << ", got " << std::int64_t(__out[__i]) << std::endl;
            ++__bad;
        }
    }
    EXPECT_EQ(std::size_t(0), __bad, (__what + ": wrong effect").c_str());
}

template <typename KeyT, typename ResT>
void
run_case(sycl::queue __q, std::size_t __n_keys, key_mix __mix, const std::string& __label)
{
    // An odd haystack length keeps the search range off a power of two.
    const std::size_t __n_hay = __n_keys | 1;
    const std::uint32_t __span = std::uint32_t(std::min<std::size_t>(4 * __n_hay, key_traits<KeyT>::max_value));
    // Lifting the haystack off zero makes keys below its first element representable.
    const std::uint32_t __base = (__mix == key_mix::absent_heavy) ? 8 : 0;

    std::vector<KeyT> __hay(__n_hay), __keys(__n_keys);
    for (std::size_t __i = 0; __i != __n_hay; ++__i)
        __hay[__i] = key_traits<KeyT>::make(__base + std::uint32_t(std::uint64_t(__i) * (__span - __base) / __n_hay));

    std::mt19937 __gen(777);
    std::uniform_int_distribution<std::uint32_t> __dist(0, __span);
    for (std::size_t __i = 0; __i != __n_keys; ++__i)
        __keys[__i] = key_traits<KeyT>::make(__dist(__gen));
    if (__mix == key_mix::absent_heavy)
    {
        const std::uint32_t __hay_max =
            __base + std::uint32_t(std::uint64_t(__n_hay - 1) * (__span - __base) / __n_hay);
        std::uniform_int_distribution<std::uint32_t> __past(__hay_max + 1, __span);
        std::uniform_int_distribution<std::uint32_t> __below(0, __base - 1);
        std::uniform_int_distribution<std::size_t> __element(0, __n_hay - 1);
        for (std::size_t __i = 0; __i != __n_keys; ++__i)
        {
            switch (__gen() & 3u)
            {
            case 0:
                __keys[__i] = __hay[__element(__gen)];
                break;
            case 1:
                __keys[__i] = key_traits<KeyT>::make(__past(__gen));
                break;
            case 2:
                __keys[__i] = key_traits<KeyT>::make(__below(__gen));
                break;
            default:
                break;
            }
        }
    }
    // Pin the ends: a key below every element and one above every element.
    __keys[0] = key_traits<KeyT>::make(0);
    __keys[__n_keys - 1] = key_traits<KeyT>::make(__span);

    std::vector<ResT> __ref_lb(__n_keys), __ref_ub(__n_keys), __ref_bs(__n_keys);
    for (std::size_t __i = 0; __i != __n_keys; ++__i)
    {
        const std::size_t __lb = std::lower_bound(__hay.begin(), __hay.end(), __keys[__i]) - __hay.begin();
        __ref_lb[__i] = ResT(__lb);
        __ref_ub[__i] = ResT(std::upper_bound(__hay.begin(), __hay.end(), __keys[__i]) - __hay.begin());
        __ref_bs[__i] = ResT(__lb != __n_hay && __hay[__lb] == __keys[__i]);
    }

    using namespace oneapi::dpl::execution;
    run_and_check(__hay, __keys, __ref_lb,
                  [__q](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                      return oneapi::dpl::lower_bound(make_device_policy<policy_name<KeyT, ResT, 0>>(__q), __f, __l,
                                                      __vf, __vl, __r);
                  },
                  __label + " lower_bound");
    run_and_check(__hay, __keys, __ref_lb,
                  [__q](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                      return oneapi::dpl::lower_bound(make_device_policy<policy_name<KeyT, ResT, 1>>(__q), __f, __l,
                                                      __vf, __vl, __r, TestUtils::IsLess<KeyT>{});
                  },
                  __label + " lower_bound with comparator");
    run_and_check(__hay, __keys, __ref_ub,
                  [__q](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                      return oneapi::dpl::upper_bound(make_device_policy<policy_name<KeyT, ResT, 2>>(__q), __f, __l,
                                                      __vf, __vl, __r);
                  },
                  __label + " upper_bound");
    run_and_check(__hay, __keys, __ref_ub,
                  [__q](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                      return oneapi::dpl::upper_bound(make_device_policy<policy_name<KeyT, ResT, 3>>(__q), __f, __l,
                                                      __vf, __vl, __r, TestUtils::IsLess<KeyT>{});
                  },
                  __label + " upper_bound with comparator");
    run_and_check(__hay, __keys, __ref_bs,
                  [__q](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                      return oneapi::dpl::binary_search(make_device_policy<policy_name<KeyT, ResT, 4>>(__q), __f, __l,
                                                        __vf, __vl, __r);
                  },
                  __label + " binary_search");
    run_and_check(__hay, __keys, __ref_bs,
                  [__q](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                      return oneapi::dpl::binary_search(make_device_policy<policy_name<KeyT, ResT, 5>>(__q), __f, __l,
                                                        __vf, __vl, __r, TestUtils::IsLess<KeyT>{});
                  },
                  __label + " binary_search with comparator");
}

template <typename KeyT, typename ResT>
void
run_type(sycl::queue __q, const std::string& __type_label)
{
    const std::size_t __min_keys = batched_path_min_keys(__q, std::min(sizeof(KeyT), sizeof(ResT)));
    // Keep the test bounded on a device whose compute unit count puts the gate out of reach.
    if (__min_keys > (std::size_t(1) << 25))
    {
        std::cout << "Skipping " << __type_label << ": batched path needs " << __min_keys << " keys" << std::endl;
        return;
    }
    std::cout << __type_label << ": batched path from " << __min_keys << " keys" << std::endl;

    // At the gate every work item has its full complement of keys; three keys past it the trailing
    // item is partial, which is a separate code path in the batched brick.
    run_case<KeyT, ResT>(__q, __min_keys, key_mix::uniform, __type_label + " full");
    run_case<KeyT, ResT>(__q, __min_keys + 3, key_mix::uniform, __type_label + " partial");
    run_case<KeyT, ResT>(__q, __min_keys, key_mix::absent_heavy, __type_label + " full, absent heavy");
    run_case<KeyT, ResT>(__q, __min_keys + 3, key_mix::absent_heavy, __type_label + " partial, absent heavy");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue __q = TestUtils::get_test_queue();

    // The value type sizes below select iterations-per-item 2, 4, 8 and 5, which is every distinct
    // batch geometry: one short batch, one full batch, two full batches, and a batch plus a tail.
    run_type<std::uint64_t, std::uint64_t>(__q, "uint64");
    run_type<std::uint32_t, std::uint32_t>(__q, "uint32");
    // A 16-bit result would take the expected and actual indices mod 65536 and compare them blind.
    run_type<std::uint16_t, std::uint32_t>(__q, "uint16");
    run_type<Key3, std::int32_t>(__q, "key3");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
