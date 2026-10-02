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

// lower_bound / upper_bound / binary_search take a batched code path once the key count is large enough
// for __parallel_for's large submitter. The gate scales with the device's compute unit count, so on a
// large GPU it sits above a million keys, which no other test in this directory reaches there. This test
// derives the gate from the device and sizes itself at twice it, so that it still crosses a gate that moves up.

#include "support/test_config.h"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include "support/utils.h"
#include "support/sycl_alloc_utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include <algorithm>
#    include <cstdint>
#    include <iostream>
#    include <limits>
#    include <memory>
#    include <random>
#    include <string>
#    include <type_traits>
#    include <vector>

// A three-byte key, to select an iterations-per-item count that is not a multiple of the number of
// searches kept in flight, so that a batch and a shorter trailing batch both run. It is also not
// default constructible, which none of these algorithms requires of a key or haystack value type.
struct Key3
{
    std::uint8_t __b[3];

    Key3() = delete;
    explicit Key3(std::uint32_t __v) : __b{std::uint8_t(__v), std::uint8_t(__v >> 8), std::uint8_t(__v >> 16)} {}

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
static_assert(!std::is_default_constructible_v<Key3>);

// A ten-byte key. The cap on the bytes a work item keeps in flight allows three such searches, which
// divides none of the iterations-per-item counts its result types select, so a trailing batch runs.
struct Key10
{
    std::uint16_t __w[5];

    explicit Key10(std::uint32_t __v)
        : __w{std::uint16_t(__v >> 16), std::uint16_t(__v), std::uint16_t(~__v >> 16), std::uint16_t(~__v), 0x5a5a}
    {
    }

    std::uint32_t
    value() const
    {
        return (std::uint32_t(__w[0]) << 16) | __w[1];
    }
    bool
    operator<(const Key10& __o) const
    {
        return value() < __o.value();
    }
    // All five words, so that a key copied short does not compare equal.
    bool
    operator==(const Key10& __o) const
    {
        bool __eq = true;
        for (int __i = 0; __i != 5; ++__i)
            __eq = __eq && __w[__i] == __o.__w[__i];
        return __eq;
    }
};
static_assert(sizeof(Key10) == 10);

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
        return Key3(__v);
    }
};

template <>
struct key_traits<Key10>
{
    static constexpr std::uint32_t max_value = std::numeric_limits<std::uint32_t>::max();
    static Key10
    make(std::uint32_t __v)
    {
        return Key10(__v);
    }
};

template <typename KeyT, typename ResT, typename Comp, int Idx>
class policy_name;

template <int Idx>
class mixed_policy_name;

// How the keys are drawn. A uniform draw over the haystack's range almost never lands outside it, so the
// searches that run off either end would reach only a handful of lanes; absent_heavy forces them.
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

// Twice the gate, so that the inputs still cross it if the gate they mirror moves up; 0 to skip.
std::size_t
test_key_count(sycl::queue __q, std::size_t __min_type_size, const std::string& __label)
{
    const std::size_t __min_keys = batched_path_min_keys(__q, __min_type_size);
    const std::size_t __n = 2 * __min_keys;
    // Safety valve on the allocation, not a device bound: 2^25 8-byte keys is already ~800 MB.
    if (__n > (std::size_t(1) << 25))
    {
        std::cout << "Skipping " << __label << ": batched path needs " << __min_keys << " keys" << std::endl;
        return 0;
    }
    std::cout << __label << ": batched path from " << __min_keys << " keys, testing " << __n << std::endl;
    return __n;
}

template <typename HayT, typename KeyT, typename ResT, typename Invoke>
void
run_and_check(sycl::queue __q, const std::vector<HayT>& __hay, const std::vector<KeyT>& __keys,
              const std::vector<ResT>& __ref, Invoke __invoke, const std::string& __what)
{
    // The key and result ranges stop one element short of their allocations. A lane past the end that
    // stores anyway overwrites the sentinel with the result for the last key, which differs from it.
    // Device USM, because a buffer opened no_init leaves the elements past the accessed range undefined.
    std::vector<KeyT> __keys_ext(__keys);
    __keys_ext.push_back(__keys.back());
    const ResT __sentinel = ResT(__ref.back() == ResT(0));
    std::unique_ptr<ResT[]> __actual(new ResT[__keys.size() + 1]);
    std::fill_n(__actual.get(), __keys.size(), ResT(0));
    __actual[__keys.size()] = __sentinel;

    using TestUtils::usm_data_transfer;
    usm_data_transfer<sycl::usm::alloc::device, HayT> __hay_dev(__q, const_cast<HayT*>(__hay.data()), __hay.size());
    usm_data_transfer<sycl::usm::alloc::device, KeyT> __key_dev(__q, __keys_ext.data(), __keys_ext.size());
    usm_data_transfer<sycl::usm::alloc::device, ResT> __out_dev(__q, __actual.get(), __keys.size() + 1);
    HayT* __hay_begin = __hay_dev.get_data();
    KeyT* __key_begin = __key_dev.get_data();
    ResT* __out_begin = __out_dev.get_data();

    ResT* __ret =
        __invoke(__hay_begin, __hay_begin + __hay.size(), __key_begin, __key_begin + __keys.size(), __out_begin);
    EXPECT_EQ(std::ptrdiff_t(__keys.size()), __ret - __out_begin, (__what + ": wrong return value").c_str());
    __out_dev.retrieve_data(__actual.get());
    EXPECT_TRUE(__actual[__keys.size()] == __sentinel, (__what + ": stored past the end").c_str());

    std::size_t __bad = 0;
    for (std::size_t __i = 0; __i != __keys.size(); ++__i)
    {
        if (__actual[__i] != __ref[__i])
        {
            if (__bad == 0)
                std::cout << __what << ": first mismatch at " << __i << ", expected " << std::int64_t(__ref[__i])
                          << ", got " << std::int64_t(__actual[__i]) << std::endl;
            ++__bad;
        }
    }
    EXPECT_EQ(std::size_t(0), __bad, (__what + ": wrong effect").c_str());
}

// binary_search writes BsResT, which may select a different iterations-per-item count from ResT.
template <typename KeyT, typename ResT, typename BsResT, typename Comp>
void
run_case(sycl::queue __q, std::size_t __n_keys, key_mix __mix, Comp __comp, const std::string& __label)
{
    // An odd haystack length keeps the search range off a power of two.
    const std::size_t __n_hay = __n_keys | 1;
    const std::uint32_t __span = std::uint32_t(std::min<std::size_t>(4 * __n_hay, key_traits<KeyT>::max_value));
    // Lifting the haystack off zero makes keys below its first element representable.
    const std::uint32_t __base = (__mix == key_mix::absent_heavy) ? 8 : 0;

    const KeyT __fill = key_traits<KeyT>::make(0);
    std::vector<KeyT> __hay(__n_hay, __fill), __keys(__n_keys, __fill);
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
    std::sort(__hay.begin(), __hay.end(), __comp);

    std::vector<ResT> __ref_lb(__n_keys), __ref_ub(__n_keys);
    std::vector<BsResT> __ref_bs(__n_keys);
    for (std::size_t __i = 0; __i != __n_keys; ++__i)
    {
        const std::size_t __lb = std::lower_bound(__hay.begin(), __hay.end(), __keys[__i], __comp) - __hay.begin();
        __ref_lb[__i] = ResT(__lb);
        __ref_ub[__i] = ResT(std::upper_bound(__hay.begin(), __hay.end(), __keys[__i], __comp) - __hay.begin());
        __ref_bs[__i] = BsResT(__lb != __n_hay && __hay[__lb] == __keys[__i]);
    }

    using namespace oneapi::dpl::execution;
    if constexpr (std::is_same_v<Comp, TestUtils::IsLess<KeyT>>)
    {
        run_and_check(__q, __hay, __keys, __ref_lb,
                      [__q](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                          return oneapi::dpl::lower_bound(make_device_policy<policy_name<KeyT, ResT, Comp, 0>>(__q),
                                                          __f, __l, __vf, __vl, __r);
                      },
                      __label + " lower_bound");
        run_and_check(__q, __hay, __keys, __ref_ub,
                      [__q](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                          return oneapi::dpl::upper_bound(make_device_policy<policy_name<KeyT, ResT, Comp, 1>>(__q),
                                                          __f, __l, __vf, __vl, __r);
                      },
                      __label + " upper_bound");
        run_and_check(__q, __hay, __keys, __ref_bs,
                      [__q](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                          return oneapi::dpl::binary_search(
                              make_device_policy<policy_name<KeyT, BsResT, Comp, 2>>(__q), __f, __l, __vf, __vl, __r);
                      },
                      __label + " binary_search");
    }
    run_and_check(__q, __hay, __keys, __ref_lb,
                  [__q, __comp](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                      return oneapi::dpl::lower_bound(make_device_policy<policy_name<KeyT, ResT, Comp, 3>>(__q), __f,
                                                      __l, __vf, __vl, __r, __comp);
                  },
                  __label + " lower_bound with comparator");
    run_and_check(__q, __hay, __keys, __ref_ub,
                  [__q, __comp](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                      return oneapi::dpl::upper_bound(make_device_policy<policy_name<KeyT, ResT, Comp, 4>>(__q), __f,
                                                      __l, __vf, __vl, __r, __comp);
                  },
                  __label + " upper_bound with comparator");
    run_and_check(__q, __hay, __keys, __ref_bs,
                  [__q, __comp](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                      return oneapi::dpl::binary_search(make_device_policy<policy_name<KeyT, BsResT, Comp, 5>>(__q),
                                                        __f, __l, __vf, __vl, __r, __comp);
                  },
                  __label + " binary_search with comparator");
}

template <typename KeyT, typename ResT, typename BsResT = ResT, typename Comp = TestUtils::IsLess<KeyT>>
void
run_type(sycl::queue __q, const std::string& __type_label)
{
    const std::size_t __n = test_key_count(__q, std::min({sizeof(KeyT), sizeof(ResT), sizeof(BsResT)}), __type_label);
    if (__n == 0)
        return;

    // __n keys give every work item its full complement. One fewer leaves the last lane of the last
    // work group one key short, so that one batch mixes in-range and out-of-range keys; three more
    // leave a trailing work group with three keys.
    run_case<KeyT, ResT, BsResT>(__q, __n, key_mix::uniform, Comp{}, __type_label + " full");
    run_case<KeyT, ResT, BsResT>(__q, __n - 1, key_mix::uniform, Comp{}, __type_label + " partial lane");
    run_case<KeyT, ResT, BsResT>(__q, __n, key_mix::absent_heavy, Comp{}, __type_label + " full, absent heavy");
    run_case<KeyT, ResT, BsResT>(__q, __n + 3, key_mix::absent_heavy, Comp{},
                                 __type_label + " partial group, absent heavy");
}

// The haystack type differs from the key type: every other float element lies half way between two
// integers, so an int32_t key equals no such element, but would if the element were converted to int.
void
run_mixed_types(sycl::queue __q)
{
    const std::size_t __n = test_key_count(__q, sizeof(bool), "float haystack, int32 keys");
    if (__n == 0)
        return;

    const std::size_t __n_keys = __n - 1;
    const std::size_t __n_hay = __n_keys | 1;
    std::vector<float> __hay(__n_hay);
    for (std::size_t __i = 0; __i != __n_hay; ++__i)
        __hay[__i] = float(__i) + ((__i & 1) ? 0.5f : 0.f);
    std::mt19937 __gen(777);
    std::uniform_int_distribution<std::int32_t> __dist(-2, std::int32_t(__n_hay) + 2);
    std::vector<std::int32_t> __keys(__n_keys);
    for (auto& __k : __keys)
        __k = __dist(__gen);

    std::vector<std::uint32_t> __ref_lb(__n_keys), __ref_ub(__n_keys);
    std::vector<bool> __ref_bs(__n_keys);
    for (std::size_t __i = 0; __i != __n_keys; ++__i)
    {
        const std::size_t __lb = std::lower_bound(__hay.begin(), __hay.end(), __keys[__i]) - __hay.begin();
        __ref_lb[__i] = __lb;
        __ref_ub[__i] = std::upper_bound(__hay.begin(), __hay.end(), __keys[__i]) - __hay.begin();
        __ref_bs[__i] = __lb != __n_hay && __hay[__lb] == __keys[__i];
    }

    using namespace oneapi::dpl::execution;
    run_and_check(__q, __hay, __keys, __ref_lb,
                  [__q](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                      return oneapi::dpl::lower_bound(make_device_policy<mixed_policy_name<0>>(__q), __f, __l, __vf,
                                                      __vl, __r);
                  },
                  "float haystack, int32 keys lower_bound");
    run_and_check(__q, __hay, __keys, __ref_ub,
                  [__q](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                      return oneapi::dpl::upper_bound(make_device_policy<mixed_policy_name<1>>(__q), __f, __l, __vf,
                                                      __vl, __r);
                  },
                  "float haystack, int32 keys upper_bound");
    run_and_check(__q, __hay, __keys, __ref_bs,
                  [__q](auto __f, auto __l, auto __vf, auto __vl, auto __r) {
                      return oneapi::dpl::binary_search(make_device_policy<mixed_policy_name<2>>(__q), __f, __l, __vf,
                                                        __vl, __r);
                  },
                  "float haystack, int32 keys binary_search");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue __q = TestUtils::get_test_queue();

    // The value type sizes below select iterations-per-item 2, 4, 8 and 5: one short batch, one full
    // batch, two full batches, and a batch plus a tail. A bool result selects 16.
    run_type<std::uint64_t, std::uint64_t>(__q, "uint64");
    run_type<std::uint32_t, std::uint32_t>(__q, "uint32");
    // A 16-bit result would take the expected and actual indices mod 65536 and compare them blind.
    run_type<std::uint16_t, std::uint32_t>(__q, "uint16");
    run_type<Key3, std::int32_t>(__q, "key3");
    run_type<Key10, std::uint32_t, bool>(__q, "key10");
    // A comparator that reverses the order, and 8-byte keys with a bool binary_search result.
    run_type<std::uint64_t, std::uint64_t, bool, TestUtils::IsGreat<std::uint64_t>>(__q, "uint64 descending");
    run_mixed_types(__q);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
