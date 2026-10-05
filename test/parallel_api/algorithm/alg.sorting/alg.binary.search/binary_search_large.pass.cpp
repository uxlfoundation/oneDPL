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

// Covers the batched lower_bound / upper_bound / binary_search path, taken above __parallel_for's
// large-submitter gate, which scales with the device.

#include "support/test_config.h"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include "support/utils.h"
#include "support/sycl_alloc_utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include <algorithm>
#    include <cstdint>
#    include <functional>
#    include <iostream>
#    include <limits>
#    include <memory>
#    include <random>
#    include <string>
#    include <tuple>
#    include <type_traits>
#    include <vector>

// 5 iterations per item: a full batch plus a tail.
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

// The byte cap allows 3 searches of this size, so 4 per item run a batch plus a tail.
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

// Overloads unary operator&, so the batch must take the address of its copies with std::addressof.
struct KeyAddr
{
    std::uint32_t __v;

    explicit KeyAddr(std::uint32_t __x) : __v(__x) {}

    KeyAddr*
    operator&()
    {
        return this + 1;
    }
    const KeyAddr*
    operator&() const
    {
        return this + 1;
    }
    bool
    operator<(const KeyAddr& __o) const
    {
        return __v < __o.__v;
    }
    bool
    operator==(const KeyAddr& __o) const
    {
        return __v == __o.__v;
    }
};

// A class-specific operator new hides the global placement form.
struct KeyNew
{
    std::uint32_t __v;

    explicit KeyNew(std::uint32_t __x) : __v(__x) {}

    static void*
    operator new(std::size_t __n)
    {
        return ::operator new(__n);
    }
    static void
    operator delete(void* __p)
    {
        ::operator delete(__p);
    }
    bool
    operator<(const KeyNew& __o) const
    {
        return __v < __o.__v;
    }
    bool
    operator==(const KeyNew& __o) const
    {
        return __v == __o.__v;
    }
};

struct KeyDtor
{
    std::uint32_t __v;

    explicit KeyDtor(std::uint32_t __x) : __v(__x) {}
    KeyDtor(const KeyDtor&) = default;
    KeyDtor&
    operator=(const KeyDtor&) = default;
    ~KeyDtor() {}

    bool
    operator<(const KeyDtor& __o) const
    {
        return __v < __o.__v;
    }
    bool
    operator==(const KeyDtor& __o) const
    {
        return __v == __o.__v;
    }
};

static_assert(!std::is_trivially_destructible_v<KeyDtor>);

// Copyable only from a const lvalue.
struct KeyConstCopy
{
    std::uint32_t __v;

    explicit KeyConstCopy(std::uint32_t __x) : __v(__x) {}
    KeyConstCopy(const KeyConstCopy&) = default;
    KeyConstCopy(KeyConstCopy&) = delete;
    KeyConstCopy&
    operator=(const KeyConstCopy&) = default;

    bool
    operator<(const KeyConstCopy& __o) const
    {
        return __v < __o.__v;
    }
    bool
    operator==(const KeyConstCopy& __o) const
    {
        return __v == __o.__v;
    }
};

static_assert(std::is_copy_constructible_v<KeyConstCopy>);
static_assert(!std::is_constructible_v<KeyConstCopy, KeyConstCopy&>);

struct Key16
{
    std::uint64_t __w[2];
};

struct Key24
{
    std::uint64_t __w[3];
};

using KeyMoveOnly = TestUtils::MoveOnlyWrapper<std::uint32_t>;

using oneapi::dpl::internal::search_algorithm;

// A pointer to (haystack, key, result) tuples stands in for the zip view the brick receives.
template <typename KeyT, std::uint8_t NumStrides = 4, typename HaystackT = KeyT,
          search_algorithm Func = search_algorithm::lower_bound>
constexpr bool takes_batched_path = oneapi::dpl::__par_backend_hetero::__brick_is_batched_v<
    oneapi::dpl::internal::__custom_brick<std::less<KeyT>, std::ptrdiff_t, Func>, NumStrides,
    oneapi::dpl::__par_backend_hetero::__pfor_params_simple, std::tuple<HaystackT, KeyT, std::uint32_t>*>;
static_assert(takes_batched_path<KeyAddr>);
static_assert(takes_batched_path<KeyNew>);
static_assert(takes_batched_path<KeyDtor>);
static_assert(takes_batched_path<KeyConstCopy>);
static_assert(!takes_batched_path<KeyMoveOnly>);
static_assert(takes_batched_path<std::uint16_t, 8>);
static_assert(!takes_batched_path<std::uint8_t, 16>);
// Only binary_search copies a haystack element.
static_assert(takes_batched_path<std::uint32_t, 4, KeyMoveOnly>);
static_assert(!takes_batched_path<std::uint32_t, 4, KeyMoveOnly, search_algorithm::binary_search>);
// The byte cap must leave at least 2 searches in flight on both index paths: 32 bytes per search does, 48 does not.
static_assert(takes_batched_path<Key16>);
static_assert(!takes_batched_path<Key24>);

// Whether NumStrides per item run a full batch plus a tail, on the 32-bit index path.
template <typename KeyT, std::uint8_t NumStrides>
constexpr bool batch_plus_tail = [] {
    using Brick = oneapi::dpl::internal::__custom_brick<std::less<KeyT>, std::ptrdiff_t, search_algorithm::lower_bound>;
    constexpr std::size_t batch =
        Brick::template __batch_size<NumStrides, Brick::max_in_flight_32, std::tuple<KeyT, KeyT, std::uint32_t>*>;
    return takes_batched_path<KeyT, NumStrides> && NumStrides > batch && NumStrides % batch != 0;
}();
static_assert(batch_plus_tail<Key3, 5>);
static_assert(batch_plus_tail<Key10, 4>);

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

template <typename KeyT>
struct key_traits_u32
{
    static constexpr std::uint32_t max_value = std::numeric_limits<std::uint32_t>::max();
    static KeyT
    make(std::uint32_t __v)
    {
        return KeyT(__v);
    }
};

template <>
struct key_traits<KeyAddr> : key_traits_u32<KeyAddr>
{
};

template <>
struct key_traits<KeyNew> : key_traits_u32<KeyNew>
{
};

template <>
struct key_traits<KeyDtor> : key_traits_u32<KeyDtor>
{
};

template <>
struct key_traits<KeyConstCopy> : key_traits_u32<KeyConstCopy>
{
};

template <>
struct key_traits<KeyMoveOnly> : key_traits_u32<KeyMoveOnly>
{
};

template <typename KeyT, typename ResT, typename Comp, int Idx>
class policy_name;

template <int Idx>
class mixed_policy_name;

// A uniform draw rarely leaves the haystack range; absent_heavy forces searches off either end.
enum class key_mix
{
    uniform,
    absent_heavy
};

// The key count from which __parallel_for takes the large submitter for these value types.
template <typename HayT, typename KeyT, typename ResT>
std::size_t
large_submitter_min_keys(sycl::queue __q)
{
    namespace __bknd = oneapi::dpl::__par_backend_hetero;
    using __submitter = __bknd::__parallel_for_large_submitter<__bknd::__internal::__optional_kernel_name<>>;
    using __params = __bknd::__pfor_params<std::tuple<HayT, KeyT, ResT>*>;
    return __submitter::__minimal_useful_size(__q, __params::__iters_per_item);
}

// Twice the gate; 0 to skip.
std::size_t
test_key_count(std::size_t __min_keys, const std::string& __label)
{
    const std::size_t __n = 2 * __min_keys;
    // Caps the allocation; not a device bound.
    if (__n > (std::size_t(1) << 25))
    {
        std::cout << "Skipping " << __label << ": large submitter needs " << __min_keys << " keys" << std::endl;
        return 0;
    }
    std::cout << __label << ": large submitter from " << __min_keys << " keys, testing " << __n << std::endl;
    return __n;
}

template <typename HayT, typename KeyT, typename ResT, typename Invoke>
void
run_and_check(sycl::queue __q, const std::vector<HayT>& __hay, const std::vector<KeyT>& __keys,
              const std::vector<ResT>& __ref, Invoke __invoke, const std::string& __what)
{
    // The key and result ranges stop one element short of their allocations, so a store past the end
    // overwrites the sentinel.
    const ResT __sentinel = ResT(__ref.back() == ResT(0));
    std::unique_ptr<ResT[]> __actual(new ResT[__keys.size() + 1]);
    std::fill_n(__actual.get(), __keys.size(), ResT(0));
    __actual[__keys.size()] = __sentinel;

    using TestUtils::usm_data_transfer;
    usm_data_transfer<sycl::usm::alloc::device, HayT> __hay_dev(__q, const_cast<HayT*>(__hay.data()), __hay.size());
    usm_data_transfer<sycl::usm::alloc::device, KeyT> __key_dev(__q, __keys.size() + 1);
    KeyT* __keys_host = const_cast<KeyT*>(__keys.data());
    __key_dev.update_data(__keys_host, 0, __keys.size());
    __key_dev.update_data(__keys_host + __keys.size() - 1, __keys.size(), 1);
    usm_data_transfer<sycl::usm::alloc::device, ResT> __out_dev(__q, __actual.get(), __keys.size() + 1);
    HayT* __hay_begin = __hay_dev.get_data();
    KeyT* __key_begin = __key_dev.get_data();
    ResT* __out_begin = __out_dev.get_data();

    ResT* __ret =
        __invoke(__hay_begin, __hay_begin + __hay.size(), __key_begin, __key_begin + __keys.size(), __out_begin);
    EXPECT_EQ(std::ptrdiff_t(__keys.size()), __ret - __out_begin, (__what + ": wrong return value").c_str());
    __out_dev.retrieve_data(__actual.get());
    EXPECT_TRUE(__actual[__keys.size()] == __sentinel, (__what + ": stored past the end").c_str());

    EXPECT_EQ_N(__ref.begin(), __actual.get(), __keys.size(), (__what + ": wrong effect").c_str());
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

    auto __hay_value = [=](std::size_t __i) {
        return __base + std::uint32_t(std::uint64_t(__i) * (__span - __base) / __n_hay);
    };

    // Built without copies, so that move-only types run too.
    std::vector<KeyT> __hay, __keys;
    __hay.reserve(__n_hay);
    __keys.reserve(__n_keys);
    for (std::size_t __i = 0; __i != __n_hay; ++__i)
        __hay.push_back(key_traits<KeyT>::make(__hay_value(__i)));

    std::mt19937 __gen(777);
    std::uniform_int_distribution<std::uint32_t> __dist(0, __span);
    for (std::size_t __i = 0; __i != __n_keys; ++__i)
        __keys.push_back(key_traits<KeyT>::make(__dist(__gen)));
    if (__mix == key_mix::absent_heavy)
    {
        const std::uint32_t __hay_max = __hay_value(__n_hay - 1);
        std::uniform_int_distribution<std::uint32_t> __past(__hay_max + 1, __span);
        std::uniform_int_distribution<std::uint32_t> __below(0, __base - 1);
        std::uniform_int_distribution<std::size_t> __element(0, __n_hay - 1);
        for (std::size_t __i = 0; __i != __n_keys; ++__i)
        {
            switch (__gen() & 3u)
            {
            case 0:
                __keys[__i] = key_traits<KeyT>::make(__hay_value(__element(__gen)));
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
    if constexpr (std::is_same_v<Comp, TestUtils::IsLess<KeyT>> || std::is_same_v<Comp, std::less<KeyT>>)
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
    const std::size_t __n = test_key_count(
        std::max(large_submitter_min_keys<KeyT, KeyT, ResT>(__q), large_submitter_min_keys<KeyT, KeyT, BsResT>(__q)),
        __type_label);
    if (__n == 0)
        return;

    // __n fills every work item; __n - 1 leaves one batch partly out of range; __n + 3 adds a trailing group.
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
    const std::size_t __n =
        test_key_count(large_submitter_min_keys<float, std::int32_t, std::uint32_t>(__q), "float haystack, int32 keys");
    if (__n == 0)
        return;

    const std::size_t __n_keys = __n - 1;
    // Below 2^23, where a float still holds every half-integer.
    const std::size_t __n_hay = std::min<std::size_t>(__n_keys, std::size_t(1) << 22) | 1;
    std::vector<float> __hay(__n_hay);
    for (std::size_t __i = 0; __i != __n_hay; ++__i)
        __hay[__i] = float(__i) + ((__i & 1) ? 0.5f : 0.f);
    std::mt19937 __gen(777);
    std::uniform_int_distribution<std::int32_t> __dist(-2, std::int32_t(__n_hay) + 2);
    std::vector<std::int32_t> __keys(__n_keys);
    for (auto& __k : __keys)
        __k = __dist(__gen);

    std::vector<std::uint32_t> __ref_lb(__n_keys), __ref_ub(__n_keys);
    std::vector<std::uint32_t> __ref_bs(__n_keys);
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
    // batch, two full batches, and a batch plus a tail. A bool result selects 16, which keeps the strided loop.
    run_type<std::uint64_t, std::uint64_t>(__q, "uint64");
    run_type<std::uint32_t, std::uint32_t>(__q, "uint32");
    run_type<std::uint32_t, std::uint32_t, bool>(__q, "uint32, bool result");
    // A 16-bit result would take the expected and actual indices mod 65536 and compare them blind.
    run_type<std::uint16_t, std::uint32_t>(__q, "uint16");
    run_type<Key3, std::int32_t>(__q, "key3");
    run_type<Key10, std::uint32_t>(__q, "key10");
    run_type<std::uint64_t, std::uint64_t, std::uint64_t, TestUtils::IsGreat<std::uint64_t>>(__q, "uint64 descending");
    run_mixed_types(__q);
    run_type<KeyAddr, std::uint32_t>(__q, "overloaded operator&");
    run_type<KeyNew, std::uint32_t>(__q, "class operator new");
    run_type<KeyDtor, std::uint32_t>(__q, "non-trivial destructor");
    run_type<KeyConstCopy, std::uint32_t, std::uint32_t, std::less<KeyConstCopy>>(__q, "copy from const only");
    // IsLess takes its arguments by value.
    run_type<KeyMoveOnly, std::uint32_t, std::uint32_t, std::less<KeyMoveOnly>>(__q, "move-only");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
